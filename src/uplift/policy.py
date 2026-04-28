"""Policy simulation tools — pure computation, no LLM.

Converts uplift scores into targeting decisions and business ROI estimates.
Uses the repo's uplift_at_k from src.uplift.metrics for consistency.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.uplift.metrics import uplift_at_k, decile_table


def simulate_targeting_policies(
    scores_df: pd.DataFrame,
    thresholds: list[float] | None = None,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
) -> list[dict]:
    """For each top-k% threshold simulate incremental lift and ROI.

    Args:
        scores_df: DataFrame with columns [client_id, uplift, treatment_flg, target]
        thresholds: top-k fractions to evaluate (default: [0.05, 0.10, 0.20, 0.30])
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30]

    y     = scores_df["target"].values
    treat = scores_df["treatment_flg"].values
    u     = scores_df["uplift"].values
    n     = len(scores_df)

    results = []
    for k in thresholds:
        lift_rate    = uplift_at_k(y, treat, u, k=k)
        n_targeted   = max(1, int(np.ceil(n * k)))
        incremental  = float(lift_rate * n_targeted) if not np.isnan(lift_rate) else float("nan")
        cost         = n_targeted * coupon_cost
        revenue_lift = incremental * revenue_per_conversion if not np.isnan(incremental) else float("nan")
        roi          = (revenue_lift - cost) / cost if (not np.isnan(revenue_lift) and cost > 0) else float("nan")

        results.append({
            "threshold_pct":           int(k * 100),
            "n_targeted":              n_targeted,
            "lift_rate":               round(float(lift_rate),    4) if not np.isnan(lift_rate)    else None,
            "incremental_conversions": round(float(incremental),  2) if not np.isnan(incremental)  else None,
            "total_cost":              round(float(cost),         2),
            "estimated_revenue_lift":  round(float(revenue_lift), 2) if not np.isnan(revenue_lift) else None,
            "roi":                     round(float(roi),          3) if not np.isnan(roi)           else None,
        })

    return results


def budget_constrained_targeting(
    scores_df: pd.DataFrame,
    budget: float,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
) -> dict:
    """Find the best targeting under a fixed coupon budget.

    Only targets customers with positive uplift, up to the budget limit.
    """
    max_coupons = int(budget / coupon_cost)
    df = scores_df.sort_values("uplift", ascending=False).reset_index(drop=True)

    positive     = df[df["uplift"] > 0]
    n_to_target  = min(max_coupons, len(positive))
    targeted     = df.iloc[:n_to_target]

    y     = targeted["target"].values
    treat = targeted["treatment_flg"].values
    u     = targeted["uplift"].values

    k_actual     = n_to_target / len(df) if len(df) > 0 else 0.0
    lift_rate    = uplift_at_k(y, treat, u, k=1.0) if n_to_target > 0 else float("nan")
    incremental  = float(lift_rate * n_to_target) if not np.isnan(lift_rate) else float("nan")
    cost         = n_to_target * coupon_cost
    revenue_lift = incremental * revenue_per_conversion if not np.isnan(incremental) else float("nan")
    roi          = (revenue_lift - cost) / cost if (not np.isnan(revenue_lift) and cost > 0) else float("nan")

    return {
        "budget":                  budget,
        "max_coupons_affordable":  max_coupons,
        "n_targeted":              n_to_target,
        "pct_of_base":             round(k_actual,         3),
        "incremental_conversions": round(float(incremental),  2) if not np.isnan(incremental)  else None,
        "total_cost":              round(float(cost),         2),
        "estimated_revenue_lift":  round(float(revenue_lift), 2) if not np.isnan(revenue_lift) else None,
        "roi":                     round(float(roi),          3) if not np.isnan(roi)           else None,
    }


def find_elbow_threshold(policy_results: list[dict]) -> int:
    """Return threshold_pct where marginal ROI starts dropping the most."""
    if len(policy_results) < 2:
        return policy_results[0]["threshold_pct"] if policy_results else 10

    rois = [r["roi"] if r["roi"] is not None else 0.0 for r in policy_results]
    drops = [rois[i] - rois[i + 1] for i in range(len(rois) - 1)]
    return policy_results[int(np.argmax(drops))]["threshold_pct"]


def customer_segment_summary(scores_df: pd.DataFrame, threshold: float = 0.10) -> dict:
    """Classify customers into uplift segments using score quantiles."""
    n   = len(scores_df)
    u   = scores_df["uplift"]

    high_cutoff = u.quantile(1 - threshold)
    low_cutoff  = u.quantile(threshold)

    persuadables  = int((u >= high_cutoff).sum())
    sleeping_dogs = int((u <= low_cutoff).sum())
    middle        = n - persuadables - sleeping_dogs

    return {
        "total_customers":   n,
        "persuadables":      persuadables,
        "sleeping_dogs":     sleeping_dogs,
        "middle_ground":     middle,
        "persuadable_pct":   round(persuadables  / n, 3),
        "sleeping_dog_pct":  round(sleeping_dogs / n, 3),
    }


def build_policy_summary(
    scores_df: pd.DataFrame,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
    budget: Optional[float] = None,
) -> dict:
    """Convenience wrapper that returns all policy artifacts in one call."""
    targeting  = simulate_targeting_policies(scores_df, coupon_cost=coupon_cost,
                                             revenue_per_conversion=revenue_per_conversion)
    budget_res = budget_constrained_targeting(scores_df, budget, coupon_cost,
                                              revenue_per_conversion) if budget else None
    elbow      = find_elbow_threshold(targeting)
    segments   = customer_segment_summary(scores_df)
    deciles    = decile_table(
        scores_df["target"].values,
        scores_df["treatment_flg"].values,
        scores_df["uplift"].values,
    )

    return {
        "targeting_results":  targeting,
        "budget_result":      budget_res,
        "elbow_threshold_pct":elbow,
        "segment_summary":    segments,
        "decile_table":       deciles.to_dict(orient="records"),
    }
