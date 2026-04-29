from src.models.uplift import (
    UpliftAdvisoryReport,
    UpliftAdvisoryVerdict,
    UpliftDiagnosisResult,
)
from src.uplift.supervisor import (
    diagnosis_call,
    report_call,
    verdict_call,
    wave_planning_call,
)


def test_m4_gate_exposes_all_advisory_call_contracts():
    assert {
        "unresolved_questions",
        "risks",
        "candidate_hypotheses",
    }.issubset(UpliftDiagnosisResult.model_fields)
    assert {
        "stop_reason",
        "hypothesis_status",
        "verdict_summary",
        "rationale",
        "next_action",
        "cited_artifact_paths",
    }.issubset(UpliftAdvisoryVerdict.model_fields)
    assert {
        "title",
        "executive_summary",
        "validation_summary",
        "held_out_summary",
        "scoring_summary",
        "limitations",
        "cited_artifact_paths",
    }.issubset(UpliftAdvisoryReport.model_fields)
    assert callable(diagnosis_call)
    assert callable(wave_planning_call)
    assert callable(verdict_call)
    assert callable(report_call)
