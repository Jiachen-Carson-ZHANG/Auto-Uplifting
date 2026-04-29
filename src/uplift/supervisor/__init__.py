"""Deterministic supervisor shell for uplift experiment waves."""

from src.uplift.supervisor.advisory import (
    build_diagnosis_prompt,
    build_report_prompt,
    build_verdict_prompt,
    build_wave_planning_prompt,
    diagnosis_call,
    report_call,
    verdict_call,
    wave_planning_call,
)
from src.uplift.supervisor.robustness import (
    evaluate_policy_threshold_stability,
    evaluate_ranking_stability,
    evaluate_robustness,
    rank_correlation,
    top_k_overlap,
)
from src.uplift.supervisor.stop_policy import (
    apply_stop_decision_to_hypothesis,
    evaluate_uplift_stop_policy,
)
from src.uplift.supervisor.waves import UpliftResearchLoop, validate_wave_spec

__all__ = [
    "UpliftResearchLoop",
    "apply_stop_decision_to_hypothesis",
    "build_diagnosis_prompt",
    "build_report_prompt",
    "build_verdict_prompt",
    "build_wave_planning_prompt",
    "diagnosis_call",
    "evaluate_policy_threshold_stability",
    "evaluate_ranking_stability",
    "evaluate_robustness",
    "evaluate_uplift_stop_policy",
    "rank_correlation",
    "report_call",
    "top_k_overlap",
    "validate_wave_spec",
    "verdict_call",
    "wave_planning_call",
]
