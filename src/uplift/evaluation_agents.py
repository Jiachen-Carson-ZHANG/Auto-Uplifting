"""PR2-style post-training evaluation phase."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.models.uplift import UpliftExperimentRecord
from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import ChatLLM
from src.uplift.metrics import (
    normalized_qini_auc_score,
    qini_auc_score,
    uplift_at_k,
    uplift_auc_score,
)
from src.uplift.policy import build_policy_summary
from src.uplift.xai import (
    check_leakage_signals,
    diagnose_xai_feature_semantics,
    explain_cached_uplift_model,
    explain_score_feature_associations,
    run_shap_solo_model,
    run_shap_two_model,
)

_SKILLS_DIR = Path(__file__).parent / "skills"


def _load_skill(name: str) -> str:
    """Load a skill prompt strictly. Missing files are a configuration bug."""
    path = _SKILLS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Required skill prompt is missing: {path}. "
            f"Add the file or remove the agent that requires it."
        )
    text = path.read_text(encoding="utf-8")
    match = re.search(r"## System Prompt\s*\n+```[^\n]*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _parse_json_strict(text: str) -> dict:
    """Strict JSON parse: returns a dict or raises ValueError. No silent fallback."""
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", stripped, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object found in LLM output: {exc}") from exc
        try:
            payload = json.loads(match.group())
        except json.JSONDecodeError as inner:
            raise ValueError(f"invalid JSON in LLM output: {inner}") from inner
    if not isinstance(payload, dict):
        raise ValueError("LLM JSON payload must be an object")
    return payload


def _call_llm_strict(
    llm: ChatLLM,
    system: str,
    user: str,
    *,
    max_retries: int = 1,
) -> dict:
    """Call ``llm`` and parse strict JSON; retry once on parse failure."""
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return _parse_json_strict(llm(system, user))
        except ValueError as exc:
            last_error = exc
    raise ValueError(
        f"LLM did not return valid JSON after {max_retries + 1} attempts: {last_error}"
    ) from last_error


_VERDICT_RANK: dict[str, int] = {"contradicted": 0, "inconclusive": 1, "supported": 2}


def _verdict_ceiling(
    metrics: dict,
    trial_status: str = "success",
    prior_champion: "UpliftExperimentRecord | None" = None,
) -> str:
    """Most permissive verdict deterministic evidence allows for the trial.

    Failed trials provide no evidence either way, so the ceiling is ``inconclusive``.
    Negative qini metrics cap the ceiling at ``contradicted``. A clearly positive
    qini allows ``supported`` only if the trial did not regress on uplift_auc versus
    the prior champion — a regression on either primary metric caps at ``inconclusive``.
    """
    if trial_status != "success":
        return "inconclusive"
    qini = metrics.get("normalized_qini_auc", metrics.get("qini_auc"))
    if not isinstance(qini, (int, float)) or (isinstance(qini, float) and qini != qini):
        return "inconclusive"
    if qini <= -0.01:
        return "contradicted"
    if qini >= 0.05:
        # Deterministic comparison: trial must also beat the prior champion on raw
        # Qini — a positive absolute Qini alone is insufficient if the search has
        # already found a better model. This prevents "supported" on a lateral move.
        surface = metrics.get("evaluation_surface", "validation")
        prior_qini = None
        prior_uplift_auc = None
        if prior_champion is not None:
            prior_qini = (
                prior_champion.held_out_qini_auc
                if surface == "held_out" and prior_champion.held_out_qini_auc is not None
                else prior_champion.qini_auc
            )
            prior_uplift_auc = (
                prior_champion.held_out_uplift_auc
                if surface == "held_out" and prior_champion.held_out_uplift_auc is not None
                else prior_champion.uplift_auc
            )
        if prior_qini is not None:
            raw_qini = metrics.get("qini_auc")
            if raw_qini is not None and raw_qini <= prior_qini:
                return "inconclusive"
        if prior_uplift_auc is not None:
            trial_uplift_auc = metrics.get("uplift_auc")
            if (
                trial_uplift_auc is not None
                and trial_uplift_auc < prior_uplift_auc - 0.001
            ):
                return "inconclusive"
        return "supported"
    return "inconclusive"


def _bound_verdict(llm_verdict: str, ceiling: str, trial_status: str = "success") -> str:
    """Clamp the LLM verdict to be no more optimistic than deterministic evidence."""
    if trial_status != "success":
        # Failed trials carry no evidence; force inconclusive regardless of LLM claim.
        return "inconclusive"
    if llm_verdict not in _VERDICT_RANK:
        return ceiling
    if _VERDICT_RANK[llm_verdict] <= _VERDICT_RANK[ceiling]:
        return llm_verdict
    return ceiling


def _scores_to_arrays(scores_df: pd.DataFrame):
    return (
        scores_df["target"].values,
        scores_df["treatment_flg"].values,
        scores_df["uplift"].values,
    )


def _score_metrics(scores_df: pd.DataFrame, *, surface: str) -> dict:
    y_true, treatment, uplift = _scores_to_arrays(scores_df)
    return {
        "qini_auc": qini_auc_score(y_true, treatment, uplift),
        "normalized_qini_auc": normalized_qini_auc_score(y_true, treatment, uplift),
        "uplift_auc": uplift_auc_score(y_true, treatment, uplift),
        "uplift_at_5pct": uplift_at_k(y_true, treatment, uplift, k=0.05),
        "uplift_at_10pct": uplift_at_k(y_true, treatment, uplift, k=0.10),
        "uplift_at_20pct": uplift_at_k(y_true, treatment, uplift, k=0.20),
        "evaluation_surface": surface,
    }


class UpliftEvaluationJudge:
    """Decide whether the tested hypothesis is supported, refuted, or inconclusive."""

    _SKILL = _load_skill("evaluation_judge")

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(
        self,
        trial_meta: dict,
        scores_df: pd.DataFrame,
        prior_champion: UpliftExperimentRecord | None = None,
        stability_score: float = 1.0,
        held_out_scores_df: pd.DataFrame | None = None,
        allow_held_out_metrics: bool = False,
    ) -> dict:
        validation_metrics = _score_metrics(scores_df, surface="validation")
        held_out_metrics = (
            _score_metrics(held_out_scores_df, surface="held_out")
            if allow_held_out_metrics
            and held_out_scores_df is not None
            and not held_out_scores_df.empty
            else None
        )
        metrics = held_out_metrics or validation_metrics
        trial_status = trial_meta.get("trial_status", "success")
        champion_block = (
            {
                "qini_auc": prior_champion.qini_auc,
                "uplift_auc": prior_champion.uplift_auc,
                "verdict": prior_champion.verdict,
            }
            if prior_champion is not None
            else "None - first run."
        )
        try:
            parsed = _call_llm_strict(
                self.llm,
                self._SKILL,
                json.dumps(
                    {
                        "trial_meta": trial_meta,
                        "computed_metrics": metrics,
                        "validation_metrics": validation_metrics,
                        "held_out_metrics": held_out_metrics,
                        "stability_score": stability_score,
                        "prior_champion": champion_block,
                    },
                    sort_keys=True,
                ),
            )
        except ValueError:
            # Judge narrative is supporting context. The deterministic ceiling
            # below remains authoritative even when the LLM is unavailable.
            parsed = {}
        # Deterministic constraint: the LLM cannot promote a verdict beyond
        # what the trial's qini_auc supports. Failed trials force inconclusive.
        ceiling = _verdict_ceiling(metrics, trial_status, prior_champion)
        proposed = parsed.get("verdict", ceiling)
        parsed["verdict"] = _bound_verdict(proposed, ceiling, trial_status)
        parsed["deterministic_verdict_ceiling"] = ceiling
        parsed["computed_metrics"] = metrics
        parsed["validation_metrics"] = validation_metrics
        parsed["held_out_metrics"] = held_out_metrics
        parsed["trial_id"] = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        return parsed


class UpliftXAIReasoner:
    """Run optional XAI and summarize leakage/plausibility evidence."""

    _SKILL = _load_skill("xai_reasoning")

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(
        self,
        trial_meta: dict,
        features_df: pd.DataFrame,
        model_dir: Path | None = None,
        judge_verdict: dict | None = None,
        scores_df: pd.DataFrame | None = None,
    ) -> dict:
        cached_model_result = self._try_cached_model_xai(
            features_df,
            model_dir,
            scores_df,
        )
        trial_id = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        if cached_model_result is not None:
            cached_model_result["trial_id"] = trial_id
            cached_model_result["skipped"] = False
            cached_model_result["leakage_auto_flag"] = check_leakage_signals(
                {"top_features": cached_model_result["global_top_features"]}
            )
            cached_model_result["feature_semantics_diagnostic"] = (
                diagnose_xai_feature_semantics(
                    cached_model_result["global_top_features"]
                )
            )
            return cached_model_result

        shap_result = self._try_shap(
            trial_meta.get("learner_family", "solo_model"),
            features_df,
            model_dir,
        )
        if shap_result is None:
            if scores_df is not None and not features_df.empty:
                fallback = explain_score_feature_associations(features_df, scores_df)
                fallback["trial_id"] = trial_id
                fallback["skipped"] = False
                fallback["leakage_auto_flag"] = check_leakage_signals(
                    {"top_features": fallback["global_top_features"]}
                )
                fallback["feature_semantics_diagnostic"] = (
                    diagnose_xai_feature_semantics(fallback["global_top_features"])
                )
                return fallback
            return {
                "skipped": True,
                "reason": "Model files not available; XAI skipped.",
                "trial_id": trial_id,
            }

        leakage = check_leakage_signals(shap_result)
        try:
            parsed = _call_llm_strict(
                self.llm,
                self._SKILL,
                json.dumps(
                    {
                        "trial_meta": trial_meta,
                        "hypothesis_text": trial_meta.get("hypothesis_text", ""),
                        "shap_result": shap_result,
                        "leakage_auto_flag": leakage,
                        "judge_verdict": judge_verdict,
                    },
                    sort_keys=True,
                ),
            )
        except ValueError:
            # XAI narrative is informational; the deterministic SHAP/leakage
            # signals below remain authoritative.
            parsed = {}
        parsed["shap_raw"] = shap_result
        parsed["leakage_auto_flag"] = leakage
        parsed["feature_semantics_diagnostic"] = diagnose_xai_feature_semantics(
            shap_result.get("global_top_features", [])
            or shap_result.get("top_features", [])
        )
        parsed["trial_id"] = trial_id
        parsed["skipped"] = False
        return parsed

    def _try_shap(
        self,
        model_type: str,
        features_df: pd.DataFrame,
        model_dir: Path | None,
    ) -> dict | None:
        if model_dir is None or features_df.empty:
            return None
        try:
            if model_type == "two_model":
                treatment_model = model_dir / "model_t.pkl"
                control_model = model_dir / "model_c.pkl"
                if not (treatment_model.exists() and control_model.exists()):
                    return None
                return run_shap_two_model(treatment_model, control_model, features_df)
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                return None
            return run_shap_solo_model(model_path, features_df)
        except Exception:
            return None

    def _try_cached_model_xai(
        self,
        features_df: pd.DataFrame,
        model_dir: Path | None,
        scores_df: pd.DataFrame | None,
    ) -> dict | None:
        if model_dir is None or features_df.empty:
            return None
        model_path = model_dir / "model.pkl"
        if not model_path.exists():
            return None
        try:
            return explain_cached_uplift_model(
                model_path,
                features_df,
                scores_df,
            )
        except Exception:
            return None


class UpliftPolicyAdvisor:
    """Convert uplift scores into targeting policy recommendations."""

    _SKILL = _load_skill("policy_simulation")

    def __init__(self, llm: ChatLLM) -> None:
        self.llm = llm

    def run(
        self,
        trial_meta: dict,
        scores_df: pd.DataFrame,
        xai_result: dict | None = None,
        coupon_cost: float = 1.0,
        revenue_per_conversion: float = 10.0,
        budget: float | None = None,
    ) -> dict:
        policy_data = build_policy_summary(
            scores_df,
            coupon_cost=coupon_cost,
            revenue_per_conversion=revenue_per_conversion,
            budget=budget,
        )
        try:
            parsed = _call_llm_strict(
                self.llm,
                self._SKILL,
                json.dumps(
                    {
                        "trial_meta": trial_meta,
                        "targeting_results": policy_data["targeting_results"],
                        "budget_result": policy_data["budget_result"],
                        "segment_summary": policy_data["segment_summary"],
                        "elbow_threshold_pct": policy_data["elbow_threshold_pct"],
                        "xai_findings": xai_result,
                    },
                    sort_keys=True,
                ),
            )
        except ValueError:
            # Policy narrative is informational; the deterministic policy_data
            # below remains the source of truth for threshold and segments.
            parsed = {}
        parsed.setdefault("recommended_threshold", policy_data["elbow_threshold_pct"])
        parsed["policy_data"] = policy_data
        parsed["trial_id"] = trial_meta.get("trial_id") or trial_meta.get("spec_id")
        return parsed


def run_evaluation_phase(
    trial_meta: dict,
    scores_df: pd.DataFrame,
    ledger: UpliftLedger,
    llm: ChatLLM,
    model_dir: Path | None = None,
    features_df: pd.DataFrame | None = None,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
    budget: float | None = None,
    trial_status: str = "success",
    held_out_scores_df: pd.DataFrame | None = None,
    allow_held_out_metrics: bool = False,
) -> dict:
    """Run PR2 Judge, XAI, and Policy agents for one completed trial.

    The judge's verdict is bounded by the deterministic ``trial_status`` and
    the qini-based ceiling computed from ``scores_df``. Failed trials always
    receive an ``inconclusive`` verdict regardless of the LLM narrative.
    """
    trial_meta = {**trial_meta, "trial_status": trial_status}
    current_trial_id = trial_meta.get("trial_id") or trial_meta.get("spec_id")
    records = ledger.load()
    champion = max(
        (
            record
            for record in records
            if record.status == "success"
            and record.qini_auc is not None
            and record.hypothesis_id != current_trial_id
            and record.run_id != current_trial_id
        ),
        key=lambda record: record.qini_auc if record.qini_auc is not None else float("-inf"),
        default=None,
    )
    judge = UpliftEvaluationJudge(llm)
    xai = UpliftXAIReasoner(llm)
    policy = UpliftPolicyAdvisor(llm)

    judgment_scores = (
        held_out_scores_df
        if allow_held_out_metrics
        and held_out_scores_df is not None
        and not held_out_scores_df.empty
        else scores_df
    )
    trial_meta = {
        **trial_meta,
        "evaluation_surface": "held_out"
        if allow_held_out_metrics
        and held_out_scores_df is not None
        and not held_out_scores_df.empty
        else "validation",
    }
    judge_result = judge.run(
        trial_meta,
        scores_df,
        champion,
        held_out_scores_df=held_out_scores_df,
        allow_held_out_metrics=allow_held_out_metrics,
    )
    xai_result = xai.run(
        trial_meta,
        features_df if features_df is not None else pd.DataFrame(),
        model_dir,
        judge_result,
        scores_df,
    )
    policy_result = policy.run(
        trial_meta,
        judgment_scores,
        xai_result,
        coupon_cost,
        revenue_per_conversion,
        budget,
    )
    return {"judge": judge_result, "xai": xai_result, "policy": policy_result}
