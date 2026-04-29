"""M6 robustness diagnostics for uplift supervisor decisions."""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftPolicyStabilityReport,
    UpliftRankingStabilityReport,
    UpliftRobustnessReport,
)


_POLICY_GAIN_KEY = re.compile(r"^top_(?P<cutoff>\d+)pct_(?P<scenario>.+)$")


def rank_correlation(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    entity_key: str = "client_id",
    score_column: str = "uplift",
) -> float | None:
    """Return Spearman-style rank correlation over shared entities."""
    merged = _shared_prediction_frame(left, right, entity_key, score_column)
    if len(merged) < 2:
        return None
    left_rank = merged[f"{score_column}_left"].rank(method="average")
    right_rank = merged[f"{score_column}_right"].rank(method="average")
    correlation = left_rank.corr(right_rank)
    if correlation is None or not math.isfinite(float(correlation)):
        return None
    return round(float(correlation), 6)


def top_k_overlap(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    k: float | int = 0.2,
    entity_key: str = "client_id",
    score_column: str = "uplift",
) -> float | None:
    """Return shared top-k entity overlap after restricting to comparable rows."""
    merged = _shared_prediction_frame(left, right, entity_key, score_column)
    if merged.empty:
        return None
    top_n = _top_n(len(merged), k)
    left_top = set(
        merged.nlargest(top_n, f"{score_column}_left")[entity_key].astype(str)
    )
    right_top = set(
        merged.nlargest(top_n, f"{score_column}_right")[entity_key].astype(str)
    )
    return round(len(left_top & right_top) / top_n, 6)


def evaluate_ranking_stability(
    records: Sequence[UpliftExperimentRecord],
    *,
    top_k: float | int = 0.2,
    min_rank_correlation: float = 0.8,
    min_top_k_overlap: float = 0.7,
    entity_key: str = "client_id",
    score_column: str = "uplift",
) -> UpliftRankingStabilityReport:
    """Aggregate pairwise ranking stability from prediction artifacts."""
    run_ids = [record.run_id for record in records]
    limitations: list[str] = []
    prediction_frames: list[tuple[str, pd.DataFrame]] = []
    for record in records:
        path = record.artifact_paths.get("predictions")
        if not path:
            limitations.append(f"{record.run_id} missing predictions artifact")
            continue
        artifact_path = Path(path)
        if not artifact_path.exists():
            limitations.append(f"{record.run_id} predictions artifact not found")
            continue
        prediction_frames.append((record.run_id, pd.read_csv(artifact_path)))

    correlations: list[float] = []
    overlaps: list[float] = []
    for (left_id, left), (right_id, right) in itertools.combinations(
        prediction_frames,
        2,
    ):
        correlation = rank_correlation(
            left,
            right,
            entity_key=entity_key,
            score_column=score_column,
        )
        overlap = top_k_overlap(
            left,
            right,
            k=top_k,
            entity_key=entity_key,
            score_column=score_column,
        )
        if correlation is None:
            limitations.append(f"{left_id}/{right_id} rank correlation unavailable")
        else:
            correlations.append(correlation)
        if overlap is None:
            limitations.append(f"{left_id}/{right_id} top-k overlap unavailable")
        else:
            overlaps.append(overlap)

    pair_count = len(prediction_frames) * (len(prediction_frames) - 1) // 2
    if pair_count == 0:
        limitations.append("ranking stability requires at least two prediction artifacts")

    min_correlation = min(correlations) if correlations else None
    min_overlap = min(overlaps) if overlaps else None
    stable = (
        not limitations
        and min_correlation is not None
        and min_overlap is not None
        and min_correlation >= min_rank_correlation
        and min_overlap >= min_top_k_overlap
    )

    return UpliftRankingStabilityReport(
        run_ids=run_ids,
        pair_count=pair_count,
        min_rank_correlation=min_correlation,
        mean_rank_correlation=round(sum(correlations) / len(correlations), 6)
        if correlations
        else None,
        min_top_k_overlap=min_overlap,
        mean_top_k_overlap=round(sum(overlaps) / len(overlaps), 6)
        if overlaps
        else None,
        stable=stable,
        limitations=limitations,
    )


def evaluate_policy_threshold_stability(
    records: Sequence[UpliftExperimentRecord],
    *,
    min_policy_gain: float = 0.0,
) -> UpliftPolicyStabilityReport:
    """Check whether each cost scenario selects the same best targeting cutoff."""
    run_ids = [record.run_id for record in records]
    limitations: list[str] = []
    best_by_record = {
        record.run_id: _best_cutoff_by_scenario(record.policy_gain)
        for record in records
    }
    scenarios = sorted(
        {
            scenario
            for best_by_scenario in best_by_record.values()
            for scenario in best_by_scenario
        }
    )
    if not scenarios:
        limitations.append("policy stability requires policy_gain cost scenarios")

    stable_cutoff_by_scenario: dict[str, str | None] = {}
    unstable_scenarios: list[str] = []
    for scenario in scenarios:
        selected = [
            best_by_record[run_id].get(scenario)
            for run_id in best_by_record
            if scenario in best_by_record[run_id]
        ]
        if len(selected) != len(records):
            limitations.append(f"{scenario} missing from at least one run")
            stable_cutoff_by_scenario[scenario] = None
            unstable_scenarios.append(scenario)
            continue

        cutoff_values = {cutoff for cutoff, _gain in selected}
        gains = [gain for _cutoff, gain in selected]
        if len(cutoff_values) == 1 and min(gains) >= min_policy_gain:
            stable_cutoff_by_scenario[scenario] = next(iter(cutoff_values))
        else:
            stable_cutoff_by_scenario[scenario] = None
            unstable_scenarios.append(scenario)

    return UpliftPolicyStabilityReport(
        run_ids=run_ids,
        stable_cutoff_by_scenario=stable_cutoff_by_scenario,
        unstable_scenarios=sorted(dict.fromkeys(unstable_scenarios)),
        stable=not limitations and not unstable_scenarios,
        limitations=limitations,
    )


def evaluate_robustness(
    records: Sequence[UpliftExperimentRecord],
    *,
    top_k: float | int = 0.2,
    min_rank_correlation: float = 0.8,
    min_top_k_overlap: float = 0.7,
    min_policy_gain: float = 0.0,
) -> UpliftRobustnessReport:
    """Return combined ranking and policy robustness evidence."""
    ranking = evaluate_ranking_stability(
        records,
        top_k=top_k,
        min_rank_correlation=min_rank_correlation,
        min_top_k_overlap=min_top_k_overlap,
    )
    policy = evaluate_policy_threshold_stability(
        records,
        min_policy_gain=min_policy_gain,
    )
    limitations = list(dict.fromkeys([*ranking.limitations, *policy.limitations]))
    return UpliftRobustnessReport(
        run_ids=[record.run_id for record in records],
        ranking=ranking,
        policy=policy,
        stable=ranking.stable and policy.stable,
        limitations=limitations,
    )


def _shared_prediction_frame(
    left: pd.DataFrame,
    right: pd.DataFrame,
    entity_key: str,
    score_column: str,
) -> pd.DataFrame:
    required = {entity_key, score_column}
    if not required.issubset(left.columns) or not required.issubset(right.columns):
        return pd.DataFrame()
    merged = left[[entity_key, score_column]].merge(
        right[[entity_key, score_column]],
        on=entity_key,
        how="inner",
        suffixes=("_left", "_right"),
    )
    merged[f"{score_column}_left"] = pd.to_numeric(
        merged[f"{score_column}_left"],
        errors="coerce",
    )
    merged[f"{score_column}_right"] = pd.to_numeric(
        merged[f"{score_column}_right"],
        errors="coerce",
    )
    return merged.dropna(subset=[f"{score_column}_left", f"{score_column}_right"])


def _top_n(row_count: int, k: float | int) -> int:
    if isinstance(k, int):
        if k <= 0:
            raise ValueError("k must be positive")
        return min(k, row_count)
    if k <= 0 or k > 1:
        raise ValueError("k must be in (0, 1]")
    return max(1, int(math.ceil(row_count * k)))


def _best_cutoff_by_scenario(
    policy_gain: dict[str, float],
) -> dict[str, tuple[str, float]]:
    best: dict[str, tuple[str, float]] = {}
    for key, raw_gain in policy_gain.items():
        if not isinstance(raw_gain, (int, float)) or isinstance(raw_gain, bool):
            continue
        gain = float(raw_gain)
        if not math.isfinite(gain):
            continue
        match = _POLICY_GAIN_KEY.match(key)
        if match is None:
            continue
        cutoff = f"top_{match.group('cutoff')}pct"
        scenario = match.group("scenario")
        current = best.get(scenario)
        if current is None or gain > current[1]:
            best[scenario] = (cutoff, gain)
    return best
