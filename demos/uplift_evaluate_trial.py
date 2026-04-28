"""Run the three evaluation agents (Judge, XAI, Policy) for one trial.

The trial must have been executed by Carson's training agent first.
Expected layout inside the trial directory:

    artifacts/uplift/<run_dir>/
        uplift_scores.csv     client_id | uplift | treatment_flg | target
        trial_meta.json       learner_family, hypothesis_text, spec_id, …
        models/               optional — model_t.pkl + model_c.pkl (TwoModels)
                                          model.pkl (SoloModel / ClassTransformation)
        features.csv          optional — feature columns for XAI

Usage
-----
    # With Ollama (local, no key needed):
    python demos/uplift_evaluate_trial.py \\
        --trial-dir artifacts/uplift/baseline_runs/runs/UT-abc123 \\
        --ledger    artifacts/uplift/baseline_runs/uplift_ledger.jsonl \\
        --provider  ollama --model qwen2.5-coder:7b

    # With Gemini free tier:
    python demos/uplift_evaluate_trial.py \\
        --trial-dir artifacts/uplift/baseline_runs/runs/UT-abc123 \\
        --ledger    artifacts/uplift/baseline_runs/uplift_ledger.jsonl \\
        --provider  gemini --model gemini-2.0-flash --api-key $GEMINI_API_KEY
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import make_chat_llm
from src.uplift.evaluation_agents import run_evaluation_phase


def _load_trial(trial_dir: Path) -> tuple[dict, pd.DataFrame, Path | None, pd.DataFrame | None]:
    meta_path   = trial_dir / "trial_meta.json"
    scores_path = trial_dir / "uplift_scores.csv"
    models_dir  = trial_dir / "models" if (trial_dir / "models").exists() else None
    feats_path  = trial_dir / "features.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"trial_meta.json not found in {trial_dir}")
    if not scores_path.exists():
        raise FileNotFoundError(f"uplift_scores.csv not found in {trial_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    scores_df = pd.read_csv(scores_path)
    required  = {"client_id", "uplift", "treatment_flg", "target"}
    missing   = required - set(scores_df.columns)
    if missing:
        raise ValueError(f"uplift_scores.csv missing columns: {missing}")

    features_df = pd.read_csv(feats_path) if feats_path.exists() else None
    if features_df is not None:
        features_df = features_df.drop(
            columns=["client_id", "uplift", "treatment_flg", "target"], errors="ignore"
        )

    return meta, scores_df, models_dir, features_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run uplift evaluation agents for one trial.")
    parser.add_argument("--trial-dir", required=True,  help="Path to the trial run directory")
    parser.add_argument("--ledger",    required=True,  help="Path to uplift_ledger.jsonl")
    parser.add_argument("--provider",  default="ollama", choices=["ollama","gemini","claude","openai","stub"])
    parser.add_argument("--model",     default=None,   help="LLM model name (uses provider default if omitted)")
    parser.add_argument("--api-key",   default=None,   help="API key (not needed for Ollama/stub)")
    parser.add_argument("--coupon-cost",  type=float, default=1.0)
    parser.add_argument("--revenue",      type=float, default=10.0,
                        help="Revenue per incremental conversion")
    parser.add_argument("--budget",       type=float, default=None,
                        help="Optional total coupon budget for constrained simulation")
    args = parser.parse_args()

    trial_dir = Path(args.trial_dir)
    ledger    = UpliftLedger(args.ledger)
    llm       = make_chat_llm(args.provider, args.model, args.api_key)

    print(f"\n{'='*60}")
    print(f"  Evaluation — {trial_dir.name}")
    print(f"  Provider   : {args.provider} / {args.model or 'default'}")
    print(f"{'='*60}\n")

    meta, scores_df, models_dir, features_df = _load_trial(trial_dir)
    print(f"  Loaded {len(scores_df):,} scored customers | model: {meta.get('learner_family')}\n")

    result = run_evaluation_phase(
        trial_meta             = meta,
        scores_df              = scores_df,
        ledger                 = ledger,
        llm                    = llm,
        model_dir              = models_dir,
        features_df            = features_df,
        coupon_cost            = args.coupon_cost,
        revenue_per_conversion = args.revenue,
        budget                 = args.budget,
    )

    out_path = trial_dir / "evaluation.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    verdict   = result["judge"].get("verdict", "unknown")
    threshold = result["policy"].get("recommended_threshold", "?")
    op_verdict= result["policy"].get("operational_verdict", "?")

    print(f"\n{'='*60}")
    print(f"  Verdict          : {verdict.upper()}")
    print(f"  Policy threshold : top {threshold}%  ({op_verdict})")
    next_hyp = result["policy"].get("new_hypothesis")
    if next_hyp:
        print(f"  Next hypothesis  : {next_hyp}")
    print(f"  Output written   : {out_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
