import pytest

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftHypothesis,
    UpliftWaveResult,
)
from src.uplift.hypotheses import UpliftHypothesisStore
from src.uplift.supervisor import (
    apply_stop_decision_to_hypothesis,
    evaluate_uplift_stop_policy,
)


def _record(
    run_id: str,
    *,
    qini_auc: float | None,
    policy_gain: dict[str, float] | None = None,
    status: str = "success",
) -> UpliftExperimentRecord:
    return UpliftExperimentRecord(
        run_id=run_id,
        hypothesis_id="UH-stop",
        feature_recipe_id=f"recipe-{run_id}",
        feature_artifact_id=f"artifact-{run_id}",
        template_name="random_baseline",
        uplift_learner_family="random",
        base_estimator="none",
        params_hash=f"params-{run_id}",
        split_seed=7,
        status=status,
        qini_auc=qini_auc,
        uplift_auc=qini_auc,
        policy_gain=policy_gain or {},
    )


def _wave(
    *,
    status: str = "completed",
    champion_run_id: str | None = "RUN-a",
    blocked_reason: str | None = None,
    failed_trial_ids: list[str] | None = None,
) -> UpliftWaveResult:
    failed = failed_trial_ids or []
    trial_ids = ["RUN-a", "RUN-b"]
    return UpliftWaveResult(
        wave_id="UW-stop-001",
        hypothesis_id="UH-stop",
        action_type="recipe_comparison",
        status=status,
        trial_ids=trial_ids,
        failed_trial_ids=failed,
        blocked_reason=blocked_reason,
        champion_run_id=champion_run_id,
        artifact_paths={"ledger": "runs/uplift_ledger.jsonl"},
    )


def _positive_records() -> list[UpliftExperimentRecord]:
    return [
        _record("RUN-a", qini_auc=0.12, policy_gain={"top_50pct_zero_cost": 1.5}),
        _record("RUN-b", qini_auc=0.02, policy_gain={"top_50pct_zero_cost": 0.2}),
    ]


def test_validity_blocked_overrides_positive_business_signal():
    decision = evaluate_uplift_stop_policy(
        _wave(status="blocked", champion_run_id=None, blocked_reason="bad template"),
        records=_positive_records(),
        valid_next_actions=["window_sweep"],
    )

    assert decision.stop_reason == "validity_blocked"
    assert decision.hypothesis_status == "inconclusive"
    assert decision.should_stop is True
    assert decision.champion_run_id is None


def test_compute_exhausted_precedes_no_valid_next_action_and_business_signal():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=_positive_records(),
        valid_next_actions=[],
        compute_exhausted=True,
    )

    assert decision.stop_reason == "compute_exhausted"
    assert decision.hypothesis_status == "inconclusive"


def test_no_valid_next_action_is_explicit_when_no_higher_priority_reason_applies():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=_positive_records(),
        valid_next_actions=[],
    )

    assert decision.stop_reason == "no_valid_next_action"
    assert decision.should_stop is True


def test_low_information_gain_requests_followup_when_actions_remain():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=[
            _record("RUN-a", qini_auc=0.105, policy_gain={"top_50pct_zero_cost": 0.1}),
            _record("RUN-b", qini_auc=0.100, policy_gain={"top_50pct_zero_cost": 0.1}),
        ],
        valid_next_actions=["feature_ablation"],
        min_metric_delta=0.02,
    )

    assert decision.stop_reason == "low_information_gain"
    assert decision.hypothesis_status == "inconclusive"
    assert decision.should_stop is False
    assert decision.next_action == "feature_ablation"


def test_low_information_gain_can_contradict_negative_hypothesis_evidence():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=[
            _record("RUN-a", qini_auc=-0.04, policy_gain={"top_50pct_zero_cost": -1.0}),
            _record("RUN-b", qini_auc=-0.05, policy_gain={"top_50pct_zero_cost": -1.2}),
        ],
        valid_next_actions=["window_sweep"],
        contradiction_metric_threshold=-0.01,
    )

    assert decision.stop_reason == "low_information_gain"
    assert decision.hypothesis_status == "contradicted"
    assert decision.should_stop is True


def test_champion_stable_precedes_policy_threshold_and_business_support():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=_positive_records(),
        valid_next_actions=["window_sweep"],
        champion_stability_runs=2,
        required_champion_stability_runs=2,
        policy_threshold_stable=True,
    )

    assert decision.stop_reason == "champion_stable"
    assert decision.hypothesis_status == "supported"
    assert decision.should_stop is True


def test_policy_threshold_stable_precedes_business_support():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=_positive_records(),
        valid_next_actions=["feature_ablation"],
        policy_threshold_stable=True,
    )

    assert decision.stop_reason == "policy_threshold_stable"
    assert decision.hypothesis_status == "supported"


def test_business_decision_supportable_requires_no_higher_priority_reason():
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=_positive_records(),
        valid_next_actions=["feature_ablation"],
    )

    assert decision.stop_reason == "business_decision_supportable"
    assert decision.hypothesis_status == "supported"
    assert decision.evidence_summary["champion_metric"] == 0.12
    assert decision.evidence_summary["best_policy_gain"] == 1.5


def test_stop_decision_updates_hypothesis_store_without_llm(tmp_path):
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    hypothesis = UpliftHypothesis(
        hypothesis_id="UH-stop",
        question="Does this wave support targeting?",
        hypothesis_text="The champion should improve uplift and policy gain.",
        stage_origin="manual",
        action_type="recipe_comparison",
    )
    store.append(hypothesis)
    decision = evaluate_uplift_stop_policy(
        _wave(),
        records=_positive_records(),
        valid_next_actions=["feature_ablation"],
    )

    updated = apply_stop_decision_to_hypothesis(hypothesis, decision)
    store.append(updated)

    latest = store.get_latest("UH-stop")
    assert latest is not None
    assert latest.status == "supported"
    assert latest.wave_ids == ["UW-stop-001"]
    assert latest.trial_ids == ["RUN-a", "RUN-b"]
    assert latest.next_action == "stop"


@pytest.mark.parametrize(
    ("records", "expected_status"),
    [
        (
            [_record("RUN-a", qini_auc=0.12, policy_gain={"top_50pct_zero_cost": 1.5})],
            "supported",
        ),
        (
            [_record("RUN-a", qini_auc=-0.04, policy_gain={"top_50pct_zero_cost": -1})],
            "contradicted",
        ),
        (
            [_record("RUN-a", qini_auc=None, policy_gain={})],
            "inconclusive",
        ),
    ],
)
def test_stop_decision_maps_wave_evidence_to_supported_contradicted_or_inconclusive(
    records,
    expected_status,
):
    decision = evaluate_uplift_stop_policy(
        _wave(champion_run_id="RUN-a"),
        records=records,
        valid_next_actions=["window_sweep"],
    )

    assert decision.hypothesis_status == expected_status
