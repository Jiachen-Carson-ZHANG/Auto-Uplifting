"""Deterministic stop policy and hypothesis verdict helpers."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftHypothesis,
    UpliftRobustnessReport,
    UpliftStopDecision,
    UpliftWaveResult,
)
from src.uplift.hypotheses import transition_hypothesis


def evaluate_uplift_stop_policy(
    wave_result: UpliftWaveResult,
    *,
    records: Sequence[UpliftExperimentRecord],
    valid_next_actions: Sequence[str] | None = None,
    compute_exhausted: bool = False,
    min_metric_delta: float = 0.01,
    business_metric_threshold: float = 0.05,
    contradiction_metric_threshold: float = -0.01,
    policy_gain_threshold: float = 0.0,
    champion_stability_runs: int = 0,
    required_champion_stability_runs: int = 2,
    policy_threshold_stable: bool = False,
    robustness_report: UpliftRobustnessReport | None = None,
) -> UpliftStopDecision:
    """Evaluate deterministic stop precedence for one wave.

    The policy dereferences ledger records by run ID and returns a pointer-only
    decision. It does not refit models or mutate metric values.
    """
    wave_records = _records_for_wave(wave_result, records)
    champion = _champion_record(wave_result, wave_records)
    evidence_summary = _evidence_summary(
        wave_result,
        wave_records,
        champion=champion,
        primary_metric="qini_auc",
    )
    if robustness_report is not None:
        evidence_summary["robustness"] = robustness_report.model_dump()
    actions = list(valid_next_actions or [])

    if wave_result.status in {"blocked", "failed"} or wave_result.blocked_reason:
        return _decision(
            wave_result,
            stop_reason="validity_blocked",
            hypothesis_status="inconclusive",
            should_stop=True,
            next_action="fix_validity_blocker",
            evidence_summary=evidence_summary,
        )

    if compute_exhausted:
        return _decision(
            wave_result,
            stop_reason="compute_exhausted",
            hypothesis_status="inconclusive",
            should_stop=True,
            next_action="stop",
            evidence_summary=evidence_summary,
        )

    if not actions:
        return _decision(
            wave_result,
            stop_reason="no_valid_next_action",
            hypothesis_status=_status_from_evidence(
                champion,
                business_metric_threshold=business_metric_threshold,
                contradiction_metric_threshold=contradiction_metric_threshold,
                policy_gain_threshold=policy_gain_threshold,
            ),
            should_stop=True,
            next_action="stop",
            evidence_summary=evidence_summary,
        )

    champion_metric = evidence_summary.get("champion_metric")
    metric_delta = evidence_summary.get("metric_delta")
    if not _is_number(champion_metric) or (
        _is_number(metric_delta) and float(metric_delta) < min_metric_delta
    ):
        hypothesis_status = _low_information_status(
            champion,
            contradiction_metric_threshold=contradiction_metric_threshold,
            policy_gain_threshold=policy_gain_threshold,
        )
        return _decision(
            wave_result,
            stop_reason="low_information_gain",
            hypothesis_status=hypothesis_status,
            should_stop=hypothesis_status == "contradicted",
            next_action="stop" if hypothesis_status == "contradicted" else actions[0],
            evidence_summary=evidence_summary,
        )

    if robustness_report is not None and not robustness_report.stable:
        next_action = (
            "ranking_stability_check"
            if "ranking_stability_check" in actions
            else actions[0]
        )
        return _decision(
            wave_result,
            stop_reason="low_information_gain",
            hypothesis_status="inconclusive",
            should_stop=False,
            next_action=next_action,
            evidence_summary=evidence_summary,
        )

    if champion_stability_runs >= required_champion_stability_runs:
        return _decision(
            wave_result,
            stop_reason="champion_stable",
            hypothesis_status="supported",
            should_stop=True,
            next_action="stop",
            evidence_summary=evidence_summary,
        )

    best_policy_gain = evidence_summary.get("best_policy_gain")
    if (
        policy_threshold_stable
        and _is_number(best_policy_gain)
        and float(best_policy_gain) >= policy_gain_threshold
    ):
        return _decision(
            wave_result,
            stop_reason="policy_threshold_stable",
            hypothesis_status="supported",
            should_stop=True,
            next_action="stop",
            evidence_summary=evidence_summary,
        )

    if _business_supportable(
        champion,
        business_metric_threshold=business_metric_threshold,
        policy_gain_threshold=policy_gain_threshold,
    ):
        return _decision(
            wave_result,
            stop_reason="business_decision_supportable",
            hypothesis_status="supported",
            should_stop=True,
            next_action="stop",
            evidence_summary=evidence_summary,
        )

    hypothesis_status = _low_information_status(
        champion,
        contradiction_metric_threshold=contradiction_metric_threshold,
        policy_gain_threshold=policy_gain_threshold,
    )
    return _decision(
        wave_result,
        stop_reason="low_information_gain",
        hypothesis_status=hypothesis_status,
        should_stop=hypothesis_status == "contradicted",
        next_action="stop" if hypothesis_status == "contradicted" else actions[0],
        evidence_summary=evidence_summary,
    )


def apply_stop_decision_to_hypothesis(
    hypothesis: UpliftHypothesis,
    decision: UpliftStopDecision,
) -> UpliftHypothesis:
    """Return a hypothesis snapshot updated from deterministic wave evidence."""
    working = hypothesis
    if working.status == "proposed":
        working = transition_hypothesis(
            working,
            "under_test",
            wave_id=decision.wave_id,
            trial_ids=decision.trial_ids,
            next_action=decision.next_action,
        )
    return transition_hypothesis(
        working,
        decision.hypothesis_status,
        wave_id=decision.wave_id,
        trial_ids=decision.trial_ids,
        next_action=decision.next_action,
    )


def _decision(
    wave_result: UpliftWaveResult,
    *,
    stop_reason: str,
    hypothesis_status: str,
    should_stop: bool,
    next_action: str | None,
    evidence_summary: dict[str, object],
) -> UpliftStopDecision:
    return UpliftStopDecision(
        wave_id=wave_result.wave_id,
        hypothesis_id=wave_result.hypothesis_id,
        action_type=wave_result.action_type,
        stop_reason=stop_reason,
        hypothesis_status=hypothesis_status,
        should_stop=should_stop,
        trial_ids=wave_result.trial_ids,
        champion_run_id=wave_result.champion_run_id,
        next_action=next_action,
        evidence_summary=evidence_summary,
        artifact_paths=wave_result.artifact_paths,
    )


def _records_for_wave(
    wave_result: UpliftWaveResult,
    records: Sequence[UpliftExperimentRecord],
) -> list[UpliftExperimentRecord]:
    requested = set(wave_result.trial_ids)
    if not requested:
        return list(records)
    return [record for record in records if record.run_id in requested]


def _champion_record(
    wave_result: UpliftWaveResult,
    records: Sequence[UpliftExperimentRecord],
) -> UpliftExperimentRecord | None:
    if wave_result.champion_run_id is None:
        return None
    for record in records:
        if record.run_id == wave_result.champion_run_id:
            return record
    return None


def _evidence_summary(
    wave_result: UpliftWaveResult,
    records: Sequence[UpliftExperimentRecord],
    *,
    champion: UpliftExperimentRecord | None,
    primary_metric: str,
) -> dict[str, object]:
    sorted_scores = sorted(
        [
            float(score)
            for score in (_metric_value(record, primary_metric) for record in records)
            if score is not None and math.isfinite(score)
        ],
        reverse=True,
    )
    champion_metric = _metric_value(champion, primary_metric) if champion else None
    runner_up_metric = sorted_scores[1] if len(sorted_scores) > 1 else None
    metric_delta = (
        float(champion_metric) - runner_up_metric
        if champion_metric is not None and runner_up_metric is not None
        else None
    )
    best_policy_key, best_policy_gain = _best_policy_gain(champion)
    return {
        "wave_status": wave_result.status,
        "failed_trial_ids": list(wave_result.failed_trial_ids),
        "blocked_reason": wave_result.blocked_reason,
        "champion_metric_name": primary_metric,
        "champion_metric": champion_metric,
        "runner_up_metric": runner_up_metric,
        "metric_delta": metric_delta,
        "best_policy_gain_key": best_policy_key,
        "best_policy_gain": best_policy_gain,
        "cost_sensitivity": dict(champion.policy_gain) if champion else {},
        "response_overlap": None,
    }


def _metric_value(
    record: UpliftExperimentRecord | None,
    metric_name: str,
) -> float | None:
    if record is None:
        return None
    value = getattr(record, metric_name, None)
    if _is_number(value):
        return float(value)
    return None


def _best_policy_gain(record: UpliftExperimentRecord | None) -> tuple[str | None, float | None]:
    if record is None or not record.policy_gain:
        return None, None
    numeric_items = [
        (key, float(value))
        for key, value in record.policy_gain.items()
        if _is_number(value) and math.isfinite(float(value))
    ]
    if not numeric_items:
        return None, None
    return max(numeric_items, key=lambda item: item[1])


def _business_supportable(
    champion: UpliftExperimentRecord | None,
    *,
    business_metric_threshold: float,
    policy_gain_threshold: float,
) -> bool:
    metric = _metric_value(champion, "qini_auc")
    _, policy_gain = _best_policy_gain(champion)
    return (
        metric is not None
        and metric >= business_metric_threshold
        and policy_gain is not None
        and policy_gain >= policy_gain_threshold
    )


def _status_from_evidence(
    champion: UpliftExperimentRecord | None,
    *,
    business_metric_threshold: float,
    contradiction_metric_threshold: float,
    policy_gain_threshold: float,
) -> str:
    metric = _metric_value(champion, "qini_auc")
    _, policy_gain = _best_policy_gain(champion)
    if metric is None:
        return "inconclusive"
    if (
        metric >= business_metric_threshold
        and policy_gain is not None
        and policy_gain >= policy_gain_threshold
    ):
        return "supported"
    if metric <= contradiction_metric_threshold and (
        policy_gain is None or policy_gain <= policy_gain_threshold
    ):
        return "contradicted"
    return "inconclusive"


def _low_information_status(
    champion: UpliftExperimentRecord | None,
    *,
    contradiction_metric_threshold: float,
    policy_gain_threshold: float,
) -> str:
    metric = _metric_value(champion, "qini_auc")
    _, policy_gain = _best_policy_gain(champion)
    if metric is not None and metric <= contradiction_metric_threshold and (
        policy_gain is None or policy_gain <= policy_gain_threshold
    ):
        return "contradicted"
    return "inconclusive"


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)
