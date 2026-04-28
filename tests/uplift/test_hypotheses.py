import pytest

from src.models.uplift import UpliftExperimentRecord, UpliftHypothesis
from src.uplift.hypotheses import (
    InvalidHypothesisTransitionError,
    link_ledger_records,
    transition_hypothesis,
)


def _hypothesis(status="proposed"):
    return UpliftHypothesis(
        question="Does recency matter?",
        hypothesis_text="Recent purchase behavior should improve uplift.",
        stage_origin="diagnosis",
        action_type="window_sweep",
        status=status,
    )


def test_transition_hypothesis_moves_proposed_to_under_test_with_wave_id():
    updated = transition_hypothesis(
        _hypothesis(),
        "under_test",
        wave_id="WAVE-1",
        next_action="run wave",
    )

    assert updated.status == "under_test"
    assert updated.wave_ids == ["WAVE-1"]
    assert updated.next_action == "run wave"


def test_transition_hypothesis_rejects_invalid_transition():
    with pytest.raises(InvalidHypothesisTransitionError, match="supported -> under_test"):
        transition_hypothesis(_hypothesis("supported"), "under_test")


def test_inconclusive_can_return_to_under_test_for_followup():
    updated = transition_hypothesis(_hypothesis("inconclusive"), "under_test")

    assert updated.status == "under_test"


def test_terminal_hypothesis_cannot_be_mutated_with_same_status_transition():
    with pytest.raises(InvalidHypothesisTransitionError, match="retired -> retired"):
        transition_hypothesis(
            _hypothesis("retired"),
            "retired",
            trial_ids=["RUN-late"],
        )


def test_link_ledger_records_adds_run_ids_and_rejects_wrong_hypothesis():
    hypothesis = _hypothesis("under_test")
    good = UpliftExperimentRecord(
        run_id="RUN-good",
        hypothesis_id=hypothesis.hypothesis_id,
        feature_recipe_id="recipe123",
        uplift_learner_family="two_model",
        base_estimator="logistic_regression",
        params_hash="abc123",
        split_seed=42,
    )
    bad = good.model_copy(update={"run_id": "RUN-bad", "hypothesis_id": "other"})

    linked = link_ledger_records(hypothesis, [good])
    assert linked.trial_ids == ["RUN-good"]

    with pytest.raises(ValueError, match="does not match hypothesis"):
        link_ledger_records(hypothesis, [bad])
