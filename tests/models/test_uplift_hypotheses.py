import pytest
from pydantic import ValidationError

from src.models.uplift import UpliftHypothesis


def test_uplift_hypothesis_defaults_to_proposed_pointer_only_record():
    hypothesis = UpliftHypothesis(
        question="Does 90d recency dominate lifetime RFM?",
        hypothesis_text="Recent engagement features should improve uplift ranking.",
        stage_origin="diagnosis",
        action_type="window_sweep",
        expected_signal="90d recipe beats lifetime recipe on qini_auc",
    )

    assert hypothesis.hypothesis_id.startswith("UH-")
    assert hypothesis.status == "proposed"
    assert hypothesis.wave_ids == []
    assert hypothesis.trial_ids == []
    assert not hasattr(hypothesis, "metric_evidence")
    assert not hasattr(hypothesis, "policy_evidence")


def test_uplift_hypothesis_rejects_empty_question_or_text():
    with pytest.raises(ValueError, match="question"):
        UpliftHypothesis(
            question=" ",
            hypothesis_text="Recent features help.",
            stage_origin="diagnosis",
            action_type="window_sweep",
        )


def test_uplift_hypothesis_deduplicates_wave_and_trial_ids():
    hypothesis = UpliftHypothesis(
        question="Does policy threshold stay stable?",
        hypothesis_text="Champion threshold should stay stable under cost scenarios.",
        stage_origin="policy",
        action_type="cost_sensitivity",
        wave_ids=["W1", "W1"],
        trial_ids=["RUN-1", "RUN-1", "RUN-2"],
    )

    assert hypothesis.wave_ids == ["W1"]
    assert hypothesis.trial_ids == ["RUN-1", "RUN-2"]


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("action_type", "freeform_codegen"),
        ("stage_origin", "unknown_agent"),
        ("status", "waiting_for_magic"),
    ],
)
def test_uplift_hypothesis_rejects_unknown_literals(field_name, bad_value):
    payload = {
        "question": "Does recency matter?",
        "hypothesis_text": "Recent purchase behavior should improve uplift.",
        "stage_origin": "diagnosis",
        "action_type": "window_sweep",
        "status": "proposed",
        field_name: bad_value,
    }

    with pytest.raises(ValidationError):
        UpliftHypothesis(**payload)
