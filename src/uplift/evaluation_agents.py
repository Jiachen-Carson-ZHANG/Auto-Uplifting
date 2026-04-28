"""Phase V — Evaluation Agents.

Agents (run in parallel after training):
    UpliftEvaluationJudge   — supported | refuted | inconclusive verdict
    UpliftXAIReasoner       — SHAP interpretation + leakage check
    UpliftPolicyAdvisor     — targeting policy + next hypothesis proposal

All three accept ChatLLM = Callable[[str, str], str] from llm_client.py.
Pure-computation tools (metrics, SHAP, policy math) are in:
    src/uplift/metrics.py, src/uplift/xai.py, src/uplift/policy.py

Adapted from Sherlyn's evaluation/ module to use:
    - repo's UpliftExperimentRecord + UpliftLedger (not a custom JSONL schema)
    - repo's metrics (qini_auc_score, uplift_auc_score, uplift_at_k)
    - repo's column conventions: client_id, treatment_flg, target, uplift
    - ChatLLM pattern from llm_client.py (not env-var provider selection)
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.models.uplift import UpliftExperimentRecord
from src.uplift.ledger import UpliftLedger
from src.uplift.metrics import (
    evaluate_uplift_predictions,
    qini_auc_score,
    uplift_auc_score,
    uplift_at_k,
)
from src.uplift.xai import (
    run_shap_two_model,
    run_shap_solo_model,
    check_leakage_signals,
)
from src.uplift.policy import build_policy_summary
from src.uplift.llm_client import ChatLLM

_SKILLS_DIR = Path(__file__).parent / "skills"


def _load_skill(name: str) -> str:
    path = _SKILLS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*\n+```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


def _scores_to_arrays(scores_df: pd.DataFrame):
    """Return (y, treatment, uplift) numpy arrays from a scores DataFrame."""
    return (
        scores_df["target"].values,
        scores_df["treatment_flg"].values,
        scores_df["uplift"].values,
    )


# ── Agent 1: UpliftEvaluationJudge ───────────────────────────────────────────

class UpliftEvaluationJudge:
    """Decides whether the tested hypothesis is supported | refuted | inconclusive."""

    _SKILL = _load_skill("evaluation_judge")

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm

    def run(
        self,
        trial_meta: dict,
        scores_df: pd.DataFrame,
        prior_champion: Optional[UpliftExperimentRecord] = None,
        stability_score: float = 1.0,
    ) -> dict:
        """
        Args:
            trial_meta:      dict from trial_meta.json (or UpliftTrialSpec fields)
            scores_df:       DataFrame with [client_id, uplift, treatment_flg, target]
            prior_champion:  best prior UpliftExperimentRecord (None on first run)
            stability_score: Jaccard ranking stability across seeds (1.0 if only one seed)

        Returns dict with verdict, reasoning, champion_comparison, confidence, key_evidence.
        """
        y, t, u = _scores_to_arrays(scores_df)

        metrics = {
            "qini_auc":       qini_auc_score(y, t, u),
            "uplift_auc":     uplift_auc_score(y, t, u),
            "uplift_at_5pct": uplift_at_k(y, t, u, k=0.05),
            "uplift_at_10pct":uplift_at_k(y, t, u, k=0.10),
            "uplift_at_20pct":uplift_at_k(y, t, u, k=0.20),
        }

        champion_block = (
            {
                "qini_auc":   prior_champion.qini_auc,
                "uplift_auc": prior_champion.uplift_auc,
                "verdict":    prior_champion.verdict,
            }
            if prior_champion else "None — first run."
        )

        user_msg = json.dumps({
            "trial_meta":        trial_meta,
            "computed_metrics":  metrics,
            "stability_score":   stability_score,
            "prior_champion":    champion_block,
        }, indent=2)

        raw    = self._llm(self._SKILL, user_msg)
        result = _parse_json(raw)
        result["computed_metrics"] = metrics
        result["trial_id"]         = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        return result


# ── Agent 2: UpliftXAIReasoner ───────────────────────────────────────────────

class UpliftXAIReasoner:
    """Runs SHAP analysis on the uplift model and interprets the results."""

    _SKILL = _load_skill("xai_reasoning")

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm

    def run(
        self,
        trial_meta: dict,
        features_df: pd.DataFrame,
        model_dir: Optional[Path] = None,
        judge_verdict: Optional[dict] = None,
    ) -> dict:
        """
        Args:
            trial_meta:    dict from trial_meta.json
            features_df:   feature columns only (no client_id / target / treatment_flg)
            model_dir:     directory containing model_t.pkl + model_c.pkl or model.pkl
            judge_verdict: output of UpliftEvaluationJudge.run() (optional context)
        """
        model_type = trial_meta.get("learner_family", "solo_model")
        shap_result = self._try_shap(model_type, features_df, model_dir)

        if shap_result is None:
            return {
                "skipped":  True,
                "reason":   "Model files not available — XAI skipped.",
                "trial_id": trial_meta.get("trial_id") or trial_meta.get("spec_id"),
            }

        leakage = check_leakage_signals(shap_result)

        user_msg = json.dumps({
            "trial_meta":        trial_meta,
            "hypothesis_text":   trial_meta.get("hypothesis_text", "Not specified"),
            "shap_result":       shap_result,
            "leakage_auto_flag": leakage,
            "judge_verdict":     judge_verdict,
        }, indent=2)

        raw    = self._llm(self._SKILL, user_msg)
        result = _parse_json(raw)
        result["shap_raw"]  = shap_result
        result["trial_id"]  = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        result["skipped"]   = False
        return result

    def _try_shap(self, model_type: str, features_df: pd.DataFrame, model_dir: Optional[Path]) -> Optional[dict]:
        if model_dir is None:
            return None
        try:
            if model_type == "two_model":
                mt = model_dir / "model_t.pkl"
                mc = model_dir / "model_c.pkl"
                if not (mt.exists() and mc.exists()):
                    return None
                return run_shap_two_model(mt, mc, features_df)
            else:
                mp = model_dir / "model.pkl"
                if not mp.exists():
                    return None
                return run_shap_solo_model(mp, features_df)
        except Exception as e:
            print(f"[XAI] SHAP failed: {e}")
            return None


# ── Agent 3: UpliftPolicyAdvisor ─────────────────────────────────────────────

class UpliftPolicyAdvisor:
    """Converts uplift scores into targeting policies and proposes next hypothesis."""

    _SKILL = _load_skill("policy_simulation")

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm

    def run(
        self,
        trial_meta: dict,
        scores_df: pd.DataFrame,
        xai_result: Optional[dict] = None,
        coupon_cost: float = 1.0,
        revenue_per_conversion: float = 10.0,
        budget: Optional[float] = None,
    ) -> dict:
        """
        Args:
            trial_meta:             dict from trial_meta.json
            scores_df:              DataFrame with [client_id, uplift, treatment_flg, target]
            xai_result:             output of UpliftXAIReasoner.run() (optional context)
            coupon_cost:            cost per coupon
            revenue_per_conversion: revenue per incremental conversion
            budget:                 optional total coupon budget
        """
        policy_data = build_policy_summary(
            scores_df,
            coupon_cost=coupon_cost,
            revenue_per_conversion=revenue_per_conversion,
            budget=budget,
        )
        elbow = policy_data["elbow_threshold_pct"]

        user_msg = json.dumps({
            "trial_meta":         trial_meta,
            "targeting_results":  policy_data["targeting_results"],
            "budget_result":      policy_data["budget_result"],
            "segment_summary":    policy_data["segment_summary"],
            "elbow_threshold_pct":elbow,
            "xai_findings":       xai_result,
        }, indent=2)

        raw    = self._llm(self._SKILL, user_msg)
        result = _parse_json(raw)
        # Attach computed data so caller has both LLM reasoning and raw numbers
        result["policy_data"]  = policy_data
        result["trial_id"]     = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        return result


# ── Convenience: run all three and produce a ledger-ready summary ─────────────

def run_evaluation_phase(
    trial_meta: dict,
    scores_df: pd.DataFrame,
    ledger: UpliftLedger,
    llm: ChatLLM,
    model_dir: Optional[Path] = None,
    features_df: Optional[pd.DataFrame] = None,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
    budget: Optional[float] = None,
) -> dict:
    """Run all three evaluation agents and return a combined result dict.

    The returned dict is structured so callers can append to UpliftLedger
    via ledger.append_result().
    """
    # Load prior champion from ledger
    records = ledger.load()
    champion = None
    if records:
        champion = max(
            (r for r in records if r.status == "success" and r.qini_auc is not None),
            key=lambda r: r.qini_auc,
            default=None,
        )

    judge  = UpliftEvaluationJudge(llm)
    xai    = UpliftXAIReasoner(llm)
    policy = UpliftPolicyAdvisor(llm)

    print("[Evaluation] Running Judge Agent…")
    judge_result  = judge.run(trial_meta, scores_df, champion)
    print(f"           -> verdict: {judge_result.get('verdict')}")

    print("[Evaluation] Running XAI Agent...")
    feat_df = features_df if features_df is not None else pd.DataFrame()
    xai_result = xai.run(trial_meta, feat_df, model_dir, judge_result)
    if xai_result.get("skipped"):
        print("           -> skipped (no model files)")
    else:
        print(f"           -> stability: {xai_result.get('stability')} | leakage: {xai_result.get('leakage_detected')}")

    print("[Evaluation] Running Policy Agent...")
    policy_result = policy.run(trial_meta, scores_df, xai_result, coupon_cost, revenue_per_conversion, budget)
    print(f"           -> recommended threshold: top {policy_result.get('recommended_threshold')}%")

    return {
        "judge":  judge_result,
        "xai":    xai_result,
        "policy": policy_result,
    }
