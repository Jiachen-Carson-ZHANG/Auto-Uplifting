"""Business-facing targeting policy simulation for uplift scores."""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.uplift.metrics import decile_table, uplift_at_k


def simulate_targeting_policies(
    scores_df: pd.DataFrame,
    thresholds: list[float] | None = None,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
) -> list[dict]:
    """Estimate incremental conversions and ROI for top-k targeting cutoffs."""
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30]

    y = scores_df["target"].values
    treatment = scores_df["treatment_flg"].values
    uplift = scores_df["uplift"].values
    n_rows = len(scores_df)

    results: list[dict] = []
    for cutoff in thresholds:
        lift_rate = uplift_at_k(y, treatment, uplift, k=cutoff)
        n_targeted = max(1, int(np.ceil(n_rows * cutoff)))
        incremental = (
            float(lift_rate * n_targeted)
            if not np.isnan(lift_rate)
            else float("nan")
        )
        cost = n_targeted * coupon_cost
        revenue_lift = (
            incremental * revenue_per_conversion
            if not np.isnan(incremental)
            else float("nan")
        )
        roi = (
            (revenue_lift - cost) / cost
            if not np.isnan(revenue_lift) and cost > 0
            else float("nan")
        )
        results.append(
            {
                "threshold_pct": int(round(cutoff * 100)),
                "n_targeted": n_targeted,
                "lift_rate": round(float(lift_rate), 4)
                if not np.isnan(lift_rate)
                else None,
                "incremental_conversions": round(float(incremental), 2)
                if not np.isnan(incremental)
                else None,
                "total_cost": round(float(cost), 2),
                "estimated_revenue_lift": round(float(revenue_lift), 2)
                if not np.isnan(revenue_lift)
                else None,
                "roi": round(float(roi), 3) if not np.isnan(roi) else None,
            }
        )
    return results


def budget_constrained_targeting(
    scores_df: pd.DataFrame,
    budget: float,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
) -> dict:
    """Estimate targeting results under a fixed coupon budget."""
    max_coupons = int(budget / coupon_cost) if coupon_cost > 0 else len(scores_df)
    ranked = scores_df.sort_values("uplift", ascending=False).reset_index(drop=True)
    positive = ranked[ranked["uplift"] > 0]
    n_to_target = min(max_coupons, len(positive))
    targeted = positive.head(n_to_target)

    if n_to_target == 0:
        return {
            "budget": budget,
            "max_coupons_affordable": max_coupons,
            "n_targeted": 0,
            "pct_of_base": 0.0,
            "incremental_conversions": None,
            "total_cost": 0.0,
            "estimated_revenue_lift": None,
            "roi": None,
        }

    lift_rate = uplift_at_k(
        targeted["target"].values,
        targeted["treatment_flg"].values,
        targeted["uplift"].values,
        k=1.0,
    )
    incremental = (
        float(lift_rate * n_to_target)
        if not np.isnan(lift_rate)
        else float("nan")
    )
    cost = n_to_target * coupon_cost
    revenue_lift = (
        incremental * revenue_per_conversion
        if not np.isnan(incremental)
        else float("nan")
    )
    roi = (
        (revenue_lift - cost) / cost
        if not np.isnan(revenue_lift) and cost > 0
        else float("nan")
    )

    return {
        "budget": budget,
        "max_coupons_affordable": max_coupons,
        "n_targeted": n_to_target,
        "pct_of_base": round(n_to_target / len(ranked), 3)
        if len(ranked)
        else 0.0,
        "incremental_conversions": round(float(incremental), 2)
        if not np.isnan(incremental)
        else None,
        "total_cost": round(float(cost), 2),
        "estimated_revenue_lift": round(float(revenue_lift), 2)
        if not np.isnan(revenue_lift)
        else None,
        "roi": round(float(roi), 3) if not np.isnan(roi) else None,
    }


def find_elbow_threshold(policy_results: list[dict]) -> int:
    """Return the threshold where ROI begins dropping most sharply."""
    if len(policy_results) < 2:
        return policy_results[0]["threshold_pct"] if policy_results else 10
    rois = [result["roi"] if result["roi"] is not None else 0.0 for result in policy_results]
    drops = [rois[index] - rois[index + 1] for index in range(len(rois) - 1)]
    return policy_results[int(np.argmax(drops))]["threshold_pct"]


def customer_segment_summary(scores_df: pd.DataFrame, threshold: float = 0.10) -> dict:
    """Summarize high-uplift, low-uplift, and middle customer groups."""
    n_rows = len(scores_df)
    uplift = scores_df["uplift"]
    if n_rows == 0:
        return {
            "total_customers": 0,
            "persuadables": 0,
            "sleeping_dogs": 0,
            "middle_ground": 0,
            "persuadable_pct": 0.0,
            "sleeping_dog_pct": 0.0,
        }

    high_cutoff = uplift.quantile(1 - threshold)
    low_cutoff = uplift.quantile(threshold)
    persuadables = int((uplift >= high_cutoff).sum())
    sleeping_dogs = int((uplift <= low_cutoff).sum())
    middle = n_rows - persuadables - sleeping_dogs

    return {
        "total_customers": n_rows,
        "persuadables": persuadables,
        "sleeping_dogs": sleeping_dogs,
        "middle_ground": middle,
        "persuadable_pct": round(persuadables / n_rows, 3),
        "sleeping_dog_pct": round(sleeping_dogs / n_rows, 3),
    }


def build_policy_summary(
    scores_df: pd.DataFrame,
    coupon_cost: float = 1.0,
    revenue_per_conversion: float = 10.0,
    budget: Optional[float] = None,
) -> dict:
    """Return all policy artifacts needed by the PR2 PolicyAdvisor."""
    targeting = simulate_targeting_policies(
        scores_df,
        coupon_cost=coupon_cost,
        revenue_per_conversion=revenue_per_conversion,
    )
    budget_result = (
        budget_constrained_targeting(
            scores_df,
            budget,
            coupon_cost=coupon_cost,
            revenue_per_conversion=revenue_per_conversion,
        )
        if budget is not None
        else None
    )
    deciles = decile_table(
        scores_df["target"].values,
        scores_df["treatment_flg"].values,
        scores_df["uplift"].values,
    )
    return {
        "targeting_results": targeting,
        "budget_result": budget_result,
        "elbow_threshold_pct": find_elbow_threshold(targeting),
        "segment_summary": customer_segment_summary(scores_df),
        "decile_table": deciles.to_dict(orient="records"),
    }
