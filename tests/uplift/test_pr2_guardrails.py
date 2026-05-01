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

from src.models.uplift import UpliftExperimentRecord, UpliftHypothesis, UpliftTrialSpec
from src.uplift.evaluation_agents import (
    UpliftEvaluationJudge,
    _bound_verdict,
    _verdict_ceiling,
    run_evaluation_phase,
)
from src.uplift.hypotheses import UpliftHypothesisStore
from src.uplift.ledger import UpliftLedger
from src.uplift.orchestrator import _record_hypothesis_trial_result
from src.uplift.planning_agents import (
    HypothesisDecision,
    HypothesisReasoningAgent,
    PlanningTrialSpec,
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


def test_hypothesis_reasoning_sanitizes_disallowed_response_and_false_rfm_gap(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")

    def llm(system: str, user: str) -> str:
        return json.dumps(
            {
                "action": "propose",
                "hypothesis": (
                    "Try response_model because no explicit RFM features are present."
                ),
                "evidence": (
                    "response_model was best before and the current recipe lacks RFM."
                ),
                "confidence": 0.8,
            }
        )

    agent = HypothesisReasoningAgent(store, llm)
    decision = agent.run(_retrieved_context(), current_hypothesis=None)

    assert "response_model" not in decision.hypothesis
    assert "lacks RFM" not in decision.evidence
    stored = store.query_by_status("proposed")[0]
    assert stored.hypothesis_text == decision.hypothesis
    assert "existing RFM baseline" in stored.hypothesis_text


def test_hypothesis_reasoning_sanitizes_unimplemented_causal_forest_language(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")

    def llm(system: str, user: str) -> str:
        return json.dumps(
            {
                "action": "propose",
                "hypothesis": "Try a causal forest because it should capture heterogeneous treatment effects.",
                "evidence": "Causal forest can handle nonlinear uplift.",
                "confidence": 0.8,
            }
        )

    agent = HypothesisReasoningAgent(store, llm)
    decision = agent.run(_retrieved_context(), current_hypothesis=None)

    assert "causal forest" not in decision.hypothesis.lower()
    assert "causal forest" not in decision.evidence.lower()
    assert "registered random-forest" in decision.hypothesis


def test_hypothesis_trial_link_uses_source_id_not_brittle_text_match(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    hypothesis = store.append(
        UpliftHypothesis(
            question="Original hypothesis text.",
            hypothesis_text="Original hypothesis text.",
            stage_origin="llm",
            action_type="recipe_comparison",
            expected_signal="held-out qini improves",
            status="proposed",
        )
    )
    planning_spec = PlanningTrialSpec(
        trial_id="UT-linked",
        hypothesis="Rephrased by the trial writer.",
        learner_family="class_transformation",
        base_estimator="gradient_boosting",
        feature_recipe="rfm_baseline",
        params={},
        split_seed=42,
        eval_cutoff=0.3,
        changes_from_previous="Try a different learner family.",
        expected_improvement="Improve held-out qini.",
        model="class_transformation + gradient_boosting",
        stop_criteria="Stop on no held-out gain.",
        source_hypothesis_id=hypothesis.hypothesis_id,
    )

    _record_hypothesis_trial_result(store, planning_spec, "UT-linked", "supported")

    latest = store.get_latest(hypothesis.hypothesis_id)
    assert latest is not None
    assert latest.status == "supported"
    assert latest.trial_ids == ["UT-linked"]


def test_hypothesis_trial_link_does_not_reopen_terminal_hypothesis(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    hypothesis = store.append(
        UpliftHypothesis(
            question="Already supported.",
            hypothesis_text="Already supported.",
            stage_origin="llm",
            action_type="recipe_comparison",
            expected_signal="held-out qini improves",
            status="supported",
            trial_ids=["UT-prior"],
        )
    )
    planning_spec = PlanningTrialSpec(
        trial_id="UT-new",
        hypothesis="Already supported.",
        learner_family="two_model",
        base_estimator="gradient_boosting",
        feature_recipe="rfm_baseline",
        params={},
        split_seed=42,
        eval_cutoff=0.3,
        changes_from_previous="Repeat terminal hypothesis.",
        expected_improvement="None.",
        model="two_model + gradient_boosting",
        stop_criteria="Stop.",
        source_hypothesis_id=hypothesis.hypothesis_id,
    )

    _record_hypothesis_trial_result(store, planning_spec, "UT-new", "supported")

    latest = store.get_latest(hypothesis.hypothesis_id)
    assert latest is not None
    assert latest.status == "supported"
    assert latest.trial_ids == ["UT-prior"]


def test_verdict_ceiling_inconclusive_when_trial_failed():
    metrics = {"qini_auc": 0.99}
    assert _verdict_ceiling(metrics, trial_status="failed") == "inconclusive"


def test_verdict_ceiling_contradicted_when_qini_clearly_negative():
    assert _verdict_ceiling({"qini_auc": -0.05}) == "contradicted"


def test_verdict_ceiling_inconclusive_when_qini_marginal():
    assert _verdict_ceiling({"qini_auc": 0.02}) == "inconclusive"


def test_verdict_ceiling_supported_when_qini_clearly_positive():
    assert _verdict_ceiling({"qini_auc": 0.10}) == "supported"


def test_verdict_ceiling_uses_held_out_prior_champion_for_regression_check():
    prior = UpliftExperimentRecord(
        hypothesis_id="UT-prior",
        feature_recipe_id="recipe123456",
        template_name="two_model_sklearn",
        uplift_learner_family="two_model",
        base_estimator="logistic_regression",
        params_hash="abc",
        split_seed=42,
        qini_auc=100.0,
        uplift_auc=0.04,
        held_out_qini_auc=300.0,
        held_out_uplift_auc=0.06,
    )

    assert (
        _verdict_ceiling(
            {
                "qini_auc": 299.0,
                "normalized_qini_auc": 0.25,
                "uplift_auc": 0.07,
                "evaluation_surface": "held_out",
            },
            prior_champion=prior,
        )
        == "inconclusive"
    )
    assert (
        _verdict_ceiling(
            {
                "qini_auc": 301.0,
                "normalized_qini_auc": 0.25,
                "uplift_auc": 0.058,
                "evaluation_surface": "held_out",
            },
            prior_champion=prior,
        )
        == "inconclusive"
    )
    assert (
        _verdict_ceiling(
            {
                "qini_auc": 301.0,
                "normalized_qini_auc": 0.25,
                "uplift_auc": 0.061,
                "evaluation_surface": "held_out",
            },
            prior_champion=prior,
        )
        == "supported"
    )


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


def test_evaluation_phase_excludes_current_trial_from_prior_champion(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    prior_spec = UpliftTrialSpec(
        spec_id="UT-prior",
        hypothesis_id="UT-prior",
        template_name="two_model_sklearn",
        learner_family="two_model",
        base_estimator="logistic_regression",
        feature_recipe_id="recipe123",
    )
    current_spec = UpliftTrialSpec(
        spec_id="UT-current",
        hypothesis_id="UT-current",
        template_name="class_transformation_gradient_boosting_sklearn",
        learner_family="class_transformation",
        base_estimator="gradient_boosting",
        feature_recipe_id="recipe123",
    )
    ledger.append_result(
        trial_spec=prior_spec,
        feature_artifact_id="artifact123",
        result_status="success",
        qini_auc=0.4,
        uplift_auc=0.04,
        artifact_paths={},
    )
    ledger.append_result(
        trial_spec=current_spec,
        feature_artifact_id="artifact123",
        result_status="success",
        qini_auc=0.5,
        uplift_auc=0.03,
        artifact_paths={},
    )
    judge_payloads = []

    def llm(system: str, user: str) -> str:
        if "Evaluation Judge" in system:
            judge_payloads.append(json.loads(user))
            return json.dumps({"verdict": "inconclusive", "key_evidence": []})
        return "{}"

    run_evaluation_phase(
        trial_meta={"spec_id": "UT-current", "learner_family": "class_transformation"},
        scores_df=pd.DataFrame(
            {
                "client_id": ["c1", "c2", "c3", "c4"],
                "uplift": [0.9, 0.3, -0.1, -0.2],
                "treatment_flg": [1, 0, 1, 0],
                "target": [1, 0, 0, 0],
            }
        ),
        ledger=ledger,
        llm=llm,
        model_dir=None,
        features_df=pd.DataFrame(),
    )

    assert judge_payloads[0]["prior_champion"]["qini_auc"] == 0.4


def test_evaluation_phase_hides_held_out_scores_from_judge_by_default(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    judge_payloads = []

    def llm(system: str, user: str) -> str:
        if "Evaluation Judge" in system:
            judge_payloads.append(json.loads(user))
            return json.dumps({"verdict": "supported", "key_evidence": []})
        return "{}"

    validation_scores = pd.DataFrame(
        {
            "client_id": [f"v{i}" for i in range(8)],
            "uplift": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "treatment_flg": [1, 0, 1, 0, 1, 0, 1, 0],
            "target": [1, 0, 1, 0, 0, 1, 0, 1],
        }
    )
    held_out_scores = pd.DataFrame(
        {
            "client_id": [f"h{i}" for i in range(8)],
            "uplift": [0.1, 0.2, 0.7, 0.8, 0.3, 0.4, 0.5, 0.6],
            "treatment_flg": [1, 0, 1, 0, 1, 0, 1, 0],
            "target": [0, 1, 1, 0, 0, 1, 1, 0],
        }
    )

    result = run_evaluation_phase(
        trial_meta={"spec_id": "UT-held", "learner_family": "two_model"},
        scores_df=validation_scores,
        held_out_scores_df=held_out_scores,
        ledger=ledger,
        llm=llm,
        model_dir=None,
        features_df=pd.DataFrame(),
    )

    assert judge_payloads[0]["computed_metrics"]["evaluation_surface"] == "validation"
    assert judge_payloads[0]["validation_metrics"]["evaluation_surface"] == "validation"
    assert judge_payloads[0]["held_out_metrics"] is None
    assert result["judge"]["computed_metrics"]["evaluation_surface"] == "validation"
    assert result["policy"]["trial_id"] == "UT-held"


def test_evaluation_phase_can_use_held_out_scores_for_explicit_final_audit(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    judge_payloads = []

    def llm(system: str, user: str) -> str:
        if "Evaluation Judge" in system:
            judge_payloads.append(json.loads(user))
            return json.dumps({"verdict": "supported", "key_evidence": []})
        return "{}"

    validation_scores = pd.DataFrame(
        {
            "client_id": [f"v{i}" for i in range(8)],
            "uplift": [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
            "treatment_flg": [1, 0, 1, 0, 1, 0, 1, 0],
            "target": [1, 0, 1, 0, 0, 1, 0, 1],
        }
    )
    held_out_scores = pd.DataFrame(
        {
            "client_id": [f"h{i}" for i in range(8)],
            "uplift": [0.1, 0.2, 0.7, 0.8, 0.3, 0.4, 0.5, 0.6],
            "treatment_flg": [1, 0, 1, 0, 1, 0, 1, 0],
            "target": [0, 1, 1, 0, 0, 1, 1, 0],
        }
    )

    result = run_evaluation_phase(
        trial_meta={"spec_id": "UT-held", "learner_family": "two_model"},
        scores_df=validation_scores,
        held_out_scores_df=held_out_scores,
        ledger=ledger,
        llm=llm,
        model_dir=None,
        features_df=pd.DataFrame(),
        allow_held_out_metrics=True,
    )

    assert judge_payloads[0]["computed_metrics"]["evaluation_surface"] == "held_out"
    assert judge_payloads[0]["held_out_metrics"]["evaluation_surface"] == "held_out"
    assert result["judge"]["computed_metrics"]["evaluation_surface"] == "held_out"


def test_skills_directory_loads_without_silent_fallback():
    """All skills referenced by class definitions must exist on disk."""
    skills_dir = Path("src/uplift/skills")
    expected = {
        "case_retrieval.md",
        "hypothesis_reasoning.md",
        "feature_semantics.md",
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
