#!/usr/bin/env python3
"""Generate a deterministic agentic tuning dry-run plan from an AutoLift ledger."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.uplift.ledger import UpliftLedger  # noqa: E402
from src.uplift.llm_client import make_chat_llm  # noqa: E402
from src.uplift.tuning import (  # noqa: E402
    build_agentic_tuning_plan,
    write_agentic_tuning_plan,
)

DEFAULT_RUN_DIR = ROOT / "results" / "run_20260430_best"


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key_for_provider(provider: str) -> str | None:
    key_by_provider = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_name = key_by_provider.get(provider)
    if env_name is None:
        return None
    value = os.getenv(env_name)
    if not value or value.startswith("PASTE_"):
        raise RuntimeError(f"Set {env_name} in .env before using provider={provider!r}.")
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        default=str(DEFAULT_RUN_DIR / "uplift_ledger.jsonl"),
        help="AutoLift ledger JSONL to use for internal candidate selection.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write the dry-run JSON plan. Defaults beside the ledger.",
    )
    parser.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "stub"))
    parser.add_argument("--model", default=os.getenv("LLM_MODEL") or None)
    parser.add_argument("--tuning-seed", type=int, default=20260501)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--budget-multiplier", type=int, default=4)
    parser.add_argument("--max-trials-per-candidate", type=int, default=16)
    parser.add_argument("--env-file", default=".env")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_local_env(ROOT / args.env_file)

    ledger_path = Path(args.ledger)
    if not ledger_path.is_absolute():
        ledger_path = ROOT / ledger_path
    output_path = (
        Path(args.output)
        if args.output
        else ledger_path.parent / "agentic_tuning_plan.json"
    )
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    provider = args.provider.strip().lower()
    llm = make_chat_llm(provider, args.model, _api_key_for_provider(provider))
    records = UpliftLedger(ledger_path).load()
    plan = build_agentic_tuning_plan(
        records,
        llm=llm,
        tuning_seed=args.tuning_seed,
        top_k=args.top_k,
        budget_multiplier=args.budget_multiplier,
        max_trials_per_candidate=args.max_trials_per_candidate,
    )
    written = write_agentic_tuning_plan(output_path, plan)
    summary = {
        "ledger_path": str(ledger_path),
        "output_path": written,
        "provider": provider,
        "tuning_seed": plan.tuning_seed,
        "top_k": plan.top_k,
        "n_candidates": len(plan.candidates),
        "n_trial_specs": len(plan.trial_specs),
        "candidate_templates": [candidate.template_name for candidate in plan.candidates],
        "budget_rule": plan.budget_rule,
        "warnings": plan.warnings,
    }
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
