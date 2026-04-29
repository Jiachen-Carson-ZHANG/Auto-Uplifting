"""Uplift metric primitives for campaign targeting evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence
import warnings

import numpy as np
import pandas as pd

from src.models.uplift import UpliftEvaluationPolicy


@dataclass(frozen=True)
class UpliftMetricResult:
    """Computed uplift metrics plus tabular artifacts."""

    qini_auc: float
    uplift_auc: float
    uplift_at_k: Dict[str, float]
    policy_gain: Dict[str, float]
    decile_table: pd.DataFrame
    qini_curve: pd.DataFrame
    uplift_curve: pd.DataFrame


def _as_1d_array(values: Sequence[float] | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    return arr


def _validate_uplift_inputs(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = _as_1d_array(y_true, "y_true").astype(int)
    t = _as_1d_array(treatment, "treatment").astype(int)
    u = _as_1d_array(uplift, "uplift").astype(float)
    if not (len(y) == len(t) == len(u)):
        raise ValueError("y_true, treatment, and uplift must have the same length")
    if len(y) == 0:
        raise ValueError("uplift metrics require at least one row")
    if not set(np.unique(y)).issubset({0, 1}):
        raise ValueError("y_true must be binary 0/1")
    if not set(np.unique(t)).issubset({0, 1}):
        raise ValueError("treatment must be binary 0/1")
    if len(np.unique(t)) < 2:
        raise ValueError("uplift metrics require both treatment and control rows")
    if np.isnan(u).any():
        raise ValueError("uplift contains NaN values")
    return y, t, u


def _trapz_compatible(y_values: pd.Series, x_values: pd.Series) -> float:
    """Integrate with numpy 1.26 compatibility and quiet numpy 2.x deprecation."""
    # Keep np.trapz for numpy 1.26 compatibility; np.trapezoid is unavailable there.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="`trapz` is deprecated.*",
            category=DeprecationWarning,
        )
        return float(np.trapz(y_values, x_values))


def _sorted_frame(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    y, t, u = _validate_uplift_inputs(y_true, treatment, uplift)
    return pd.DataFrame({"target": y, "treatment": t, "uplift": u}).sort_values(
        "uplift",
        ascending=False,
        kind="mergesort",
    )


def qini_curve_data(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return cumulative Qini curve points sorted by predicted uplift."""
    frame = _sorted_frame(y_true, treatment, uplift).reset_index(drop=True)
    treated = frame["treatment"] == 1
    control = ~treated

    cum_treated = treated.cumsum()
    cum_control = control.cumsum()
    cum_y_treated = (frame["target"] * treated.astype(int)).cumsum()
    cum_y_control = (frame["target"] * control.astype(int)).cumsum()

    control_scale = cum_treated / cum_control.replace(0, np.nan)
    qini = (cum_y_treated - control_scale.fillna(0.0) * cum_y_control).astype(float)

    return pd.DataFrame(
        {
            "fraction": (np.arange(len(frame)) + 1) / len(frame),
            "qini": qini,
        }
    )


def uplift_curve_data(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Return cumulative treatment-control response-rate differences."""
    frame = _sorted_frame(y_true, treatment, uplift).reset_index(drop=True)
    treated = frame["treatment"] == 1
    control = ~treated

    cum_treated = treated.cumsum()
    cum_control = control.cumsum()
    cum_y_treated = (frame["target"] * treated.astype(int)).cumsum()
    cum_y_control = (frame["target"] * control.astype(int)).cumsum()

    treated_rate = cum_y_treated / cum_treated.replace(0, np.nan)
    control_rate = cum_y_control / cum_control.replace(0, np.nan)
    uplift_rate = (treated_rate - control_rate).fillna(0.0).astype(float)

    return pd.DataFrame(
        {
            "fraction": (np.arange(len(frame)) + 1) / len(frame),
            "uplift": uplift_rate,
        }
    )


def qini_auc_score(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> float:
    """Area under the cumulative Qini curve."""
    curve = qini_curve_data(y_true, treatment, uplift)
    return round(_trapz_compatible(curve["qini"], curve["fraction"]), 6)


def uplift_auc_score(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
) -> float:
    """Area under the cumulative uplift-rate curve."""
    curve = uplift_curve_data(y_true, treatment, uplift)
    return round(_trapz_compatible(curve["uplift"], curve["fraction"]), 6)


def uplift_at_k(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    *,
    k: float,
) -> float:
    """Treatment-control response-rate difference among top-k ranked rows.

    Returns NaN when the top-k slice has zero treated or zero control rows: the
    uplift estimate is undefined, not zero. Callers should handle NaN explicitly
    rather than treating it as a value of 0.
    """
    if k <= 0 or k > 1:
        raise ValueError("k must be in (0, 1]")
    frame = _sorted_frame(y_true, treatment, uplift)
    top_n = max(1, int(np.ceil(len(frame) * k)))
    top = frame.head(top_n)
    treated = top[top["treatment"] == 1]
    control = top[top["treatment"] == 0]
    if treated.empty or control.empty:
        return float("nan")
    return round(float(treated["target"].mean() - control["target"].mean()), 6)


def decile_table(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Build a ranked-bin table with response rates and observed uplift."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1")
    frame = _sorted_frame(y_true, treatment, uplift).reset_index(drop=True)
    bins = np.array_split(frame.index.to_numpy(), min(n_bins, len(frame)))
    rows = []
    for i, idx in enumerate(bins, start=1):
        part = frame.loc[idx]
        treated = part[part["treatment"] == 1]
        control = part[part["treatment"] == 0]
        treated_rate = float(treated["target"].mean()) if not treated.empty else 0.0
        control_rate = float(control["target"].mean()) if not control.empty else 0.0
        rows.append(
            {
                "bin": i,
                "n": int(len(part)),
                "treated_n": int(len(treated)),
                "control_n": int(len(control)),
                "treated_response_rate": round(treated_rate, 6),
                "control_response_rate": round(control_rate, 6),
                "uplift": round(treated_rate - control_rate, 6),
                "avg_predicted_uplift": round(float(part["uplift"].mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def policy_gain_by_cutoff(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    policy: UpliftEvaluationPolicy,
) -> Dict[str, float]:
    """Estimate simple policy gain by cutoff and configured cost scenario.

    When the underlying uplift_at_k is undefined (NaN) for a cutoff, the gain
    for every cost scenario at that cutoff is also NaN: there is no honest
    way to claim a gain when the treatment-control comparison is undefined.
    """
    gains: Dict[str, float] = {}
    conversion_value = 1.0 if policy.conversion_value is None else policy.conversion_value
    n_rows = len(_as_1d_array(y_true, "y_true"))
    for cutoff in policy.cutoff_grid:
        observed_uplift = uplift_at_k(y_true, treatment, uplift, k=cutoff)
        n_contacted = int(np.ceil(n_rows * cutoff))
        cutoff_pct = int(round(cutoff * 100))
        for scenario, communication_cost in policy.cost_scenarios.items():
            key = f"top_{cutoff_pct}pct_{scenario}"
            if np.isnan(observed_uplift):
                gains[key] = float("nan")
                continue
            gain = observed_uplift * n_contacted * conversion_value
            gain -= n_contacted * communication_cost
            gains[key] = round(float(gain), 6)
    return gains


def evaluate_uplift_predictions(
    y_true: Sequence[int] | np.ndarray,
    treatment: Sequence[int] | np.ndarray,
    uplift: Sequence[float] | np.ndarray,
    policy: UpliftEvaluationPolicy,
) -> UpliftMetricResult:
    """Compute all first-pass uplift metrics and tabular artifacts."""
    qini_curve = qini_curve_data(y_true, treatment, uplift)
    uplift_curve = uplift_curve_data(y_true, treatment, uplift)
    at_k = {
        f"top_{int(round(cutoff * 100))}pct": uplift_at_k(
            y_true,
            treatment,
            uplift,
            k=cutoff,
        )
        for cutoff in policy.cutoff_grid
    }
    return UpliftMetricResult(
        qini_auc=qini_auc_score(y_true, treatment, uplift),
        uplift_auc=uplift_auc_score(y_true, treatment, uplift),
        uplift_at_k=at_k,
        policy_gain=policy_gain_by_cutoff(y_true, treatment, uplift, policy),
        decile_table=decile_table(y_true, treatment, uplift),
        qini_curve=qini_curve,
        uplift_curve=uplift_curve,
    )
