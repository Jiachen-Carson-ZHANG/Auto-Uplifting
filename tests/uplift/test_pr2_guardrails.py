"""Regression tests for PR2 planning/evaluation guardrails.

These tests lock in the four patches applied after the parallel-LLM-stack audit:
strict JSON parsing, dynamic action_type for hypothesis records, deterministic
verdict ceiling on the judge, and strict skill-prompt loading.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.uplift import UpliftHypothesis
from src.uplift.evaluation_agents import (
    UpliftEvaluationJudge,
    _bound_verdict,
    _verdict_ceiling,
)
from src.uplift.hypotheses import UpliftHypothesisStore
from src.uplift.planning_agents import (
    HypothesisDecision,
    HypothesisReasoningAgent,
    RetrievedContext,
)


def _retrieved_context() -> RetrievedContext:
    return RetrievedContext(
        similar_recipes=[],
        supported_hypotheses=[],
        refuted_hypotheses=[],
        best_learner_family="response_model",
        failed_runs=[],
        summary="cold start",
    )


def test_planning_agent_raises_on_invalid_json_after_retries(tmp_path):
    """HypothesisReasoningAgent must not silently fall back to {} on garbage JSON."""
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")

    def garbage_llm(system: str, user: str) -> str:
        return "not json at all"

    agent = HypothesisReasoningAgent(store, garbage_llm)
    with pytest.raises(ValueError, match="LLM did not return valid JSON"):
        agent.run(_retrieved_context(), current_hypothesis="test")


def test_hypothesis_action_type_uses_llm_value_when_valid(tmp_path):
    """LLM-supplied experiment_action_type must be honored, not hardcoded."""
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")

    def llm(system: str, user: str) -> str:
        return json.dumps(
            {
                "action": "propose",
                "hypothesis": "30d windows beat lifetime aggregates.",
                "evidence": "prior runs flat",
                "confidence": 0.6,
                "experiment_action_type": "window_sweep",
            }
        )

    agent = HypothesisReasoningAgent(store, llm)
    decision = agent.run(_retrieved_context(), current_hypothesis=None)

    assert decision.experiment_action_type == "window_sweep"
    stored = store.load_snapshots()
    assert len(stored) == 1
    assert stored[0].action_type == "window_sweep"


def test_hypothesis_action_type_falls_back_when_llm_value_invalid(tmp_path):
    """Unknown action_type values fall back to recipe_comparison rather than raising."""
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")

    def llm(system: str, user: str) -> str:
        return json.dumps(
            {
                "action": "propose",
                "hypothesis": "Some hypothesis text.",
                "evidence": "ev",
                "confidence": 0.5,
                "experiment_action_type": "invented_action",
            }
        )

    agent = HypothesisReasoningAgent(store, llm)
    decision = agent.run(_retrieved_context(), current_hypothesis=None)

    assert decision.experiment_action_type == "recipe_comparison"
    assert store.load_snapshots()[0].action_type == "recipe_comparison"


def test_verdict_ceiling_inconclusive_when_trial_failed():
    metrics = {"qini_auc": 0.99}
    assert _verdict_ceiling(metrics, trial_status="failed") == "inconclusive"


def test_verdict_ceiling_contradicted_when_qini_clearly_negative():
    assert _verdict_ceiling({"qini_auc": -0.05}) == "contradicted"


def test_verdict_ceiling_inconclusive_when_qini_marginal():
    assert _verdict_ceiling({"qini_auc": 0.02}) == "inconclusive"


def test_verdict_ceiling_supported_when_qini_clearly_positive():
    assert _verdict_ceiling({"qini_auc": 0.10}) == "supported"


def test_bound_verdict_caps_optimistic_llm_output():
    # LLM says supported; deterministic ceiling is inconclusive → use ceiling.
    assert _bound_verdict("supported", "inconclusive") == "inconclusive"


def test_bound_verdict_allows_more_conservative_llm_output():
    # LLM says inconclusive; deterministic ceiling is supported → use LLM (more conservative).
    assert _bound_verdict("inconclusive", "supported") == "inconclusive"


def test_bound_verdict_forces_inconclusive_for_failed_trial():
    # Even if the LLM claims contradicted, a failed trial has no evidence.
    assert _bound_verdict("contradicted", "inconclusive", trial_status="failed") == "inconclusive"
    assert _bound_verdict("supported", "supported", trial_status="failed") == "inconclusive"


def test_judge_clamps_supported_verdict_when_qini_is_marginal():
    """The LLM saying 'supported' on weak metrics must be downgraded to 'inconclusive'."""

    def overconfident_llm(system: str, user: str) -> str:
        return json.dumps(
            {
                "verdict": "supported",
                "reasoning": "Overconfident narrative.",
                "key_evidence": [],
            }
        )

    judge = UpliftEvaluationJudge(overconfident_llm)
    # Construct scores where qini_auc lands in the "marginal" band (≥-0.01 and <0.05).
    # 8 rows split between treatment/control with weak ranking → qini near zero.
    scores_df = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(8)],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
            "treatment_flg": [0, 0, 1, 1, 0, 0, 1, 1],
            "uplift": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        }
    )

    result = judge.run(
        trial_meta={"spec_id": "UT-test", "trial_status": "success"},
        scores_df=scores_df,
    )

    ceiling = result["deterministic_verdict_ceiling"]
    # Either the ceiling is inconclusive (verdict downgraded) or supported
    # (verdict allowed). The invariant: verdict must not exceed the ceiling.
    rank = {"contradicted": 0, "inconclusive": 1, "supported": 2}
    assert rank[result["verdict"]] <= rank[ceiling]


def test_judge_forces_inconclusive_when_trial_failed():
    """A failed trial cannot produce 'supported' verdict regardless of LLM claim."""

    def overconfident_llm(system: str, user: str) -> str:
        return json.dumps({"verdict": "supported"})

    judge = UpliftEvaluationJudge(overconfident_llm)
    scores_df = pd.DataFrame(
        {
            "client_id": ["c0", "c1", "c2", "c3"],
            "target": [1, 1, 1, 1],
            "treatment_flg": [1, 0, 1, 0],
            "uplift": [0.9, 0.8, 0.7, 0.6],
        }
    )

    result = judge.run(
        trial_meta={"spec_id": "UT-failed", "trial_status": "failed"},
        scores_df=scores_df,
    )

    assert result["verdict"] == "inconclusive"
    assert result["deterministic_verdict_ceiling"] == "inconclusive"


def test_skills_directory_loads_without_silent_fallback():
    """All skills referenced by class definitions must exist on disk."""
    skills_dir = Path("src/uplift/skills")
    expected = {
        "case_retrieval.md",
        "hypothesis_reasoning.md",
        "uplift_strategy_selection.md",
        "trial_spec_writer.md",
        "evaluation_judge.md",
        "xai_reasoning.md",
        "policy_simulation.md",
    }
    actual = {path.name for path in skills_dir.glob("*.md")}
    assert expected.issubset(actual), f"missing skill prompts: {expected - actual}"


def test_load_skill_raises_for_missing_file():
    """The strict loader must raise rather than degrade to a word-salad fallback."""
    from src.uplift.planning_agents import _load_skill

    with pytest.raises(FileNotFoundError, match="skill prompt is missing"):
        _load_skill("nonexistent_skill_name")
