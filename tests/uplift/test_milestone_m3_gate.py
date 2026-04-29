from src.models.uplift import UpliftStopDecision
from src.uplift.supervisor import evaluate_uplift_stop_policy


def test_m3_gate_has_pointer_only_stop_decision_contract():
    assert {
        "wave_id",
        "hypothesis_id",
        "action_type",
        "stop_reason",
        "hypothesis_status",
        "should_stop",
        "trial_ids",
        "champion_run_id",
        "next_action",
        "evidence_summary",
        "artifact_paths",
    }.issubset(UpliftStopDecision.model_fields)
    assert callable(evaluate_uplift_stop_policy)
    assert "selected_metric_summary" not in UpliftStopDecision.model_fields
