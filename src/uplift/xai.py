"""SHAP-based explainability tools for uplift models.

Supports TwoModels (separate treated/control arms) and SoloModel
(single model with treatment as a counterfactual feature).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def run_shap_two_model(
    model_t_path: Path,
    model_c_path: Path,
    features_df: pd.DataFrame,
    max_samples: int = 2000,
) -> dict:
    """SHAP for TwoModels: treated arm vs control arm.

    The ``gap`` column (shap_treated - shap_control) highlights features that
    drive the *differential* response — these are most relevant to uplift.
    """
    import shap

    with open(model_t_path, "rb") as f:
        model_t = pickle.load(f)
    with open(model_c_path, "rb") as f:
        model_c = pickle.load(f)

    X = features_df.sample(min(max_samples, len(features_df)), random_state=42)

    shap_t = np.abs(shap.TreeExplainer(model_t).shap_values(X)).mean(axis=0)
    shap_c = np.abs(shap.TreeExplainer(model_c).shap_values(X)).mean(axis=0)

    feature_names = features_df.columns.tolist()
    gap = shap_t - shap_c

    results = [
        {
            "feature":       feat,
            "shap_treated":  round(float(shap_t[i]), 4),
            "shap_control":  round(float(shap_c[i]), 4),
            "gap":           round(float(gap[i]),    4),
        }
        for i, feat in enumerate(feature_names)
    ]
    results.sort(key=lambda x: abs(x["gap"]), reverse=True)

    return {
        "method":         "two_model",
        "top_features":   results[:10],
        "n_samples_used": len(X),
    }


def run_shap_solo_model(
    model_path: Path,
    features_df: pd.DataFrame,
    treatment_col: str = "treatment_flg",
    max_samples: int = 2000,
) -> dict:
    """SHAP for SoloModel: compare explanations when treatment=1 vs treatment=0.

    Features where importance changes between counterfactuals are uplift-relevant.
    """
    import shap

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X = features_df.sample(min(max_samples, len(features_df)), random_state=42).copy()

    X_treat = X.copy(); X_treat[treatment_col] = 1
    X_ctrl  = X.copy(); X_ctrl[treatment_col]  = 0

    explainer = shap.TreeExplainer(model)
    shap_t    = np.abs(explainer.shap_values(X_treat)).mean(axis=0)
    shap_c    = np.abs(explainer.shap_values(X_ctrl)).mean(axis=0)

    col_list = X.columns.tolist()
    gap      = shap_t - shap_c

    results = [
        {
            "feature":       feat,
            "shap_treated":  round(float(shap_t[i]), 4),
            "shap_control":  round(float(shap_c[i]), 4),
            "gap":           round(float(gap[i]),    4),
        }
        for i, feat in enumerate(col_list)
    ]
    results.sort(key=lambda x: abs(x["gap"]), reverse=True)

    return {
        "method":         "solo_model",
        "top_features":   results[:10],
        "n_samples_used": len(X),
    }


def check_leakage_signals(
    shap_result: dict,
    leakage_keywords: Optional[list[str]] = None,
) -> bool:
    """Flag potential leakage if post-treatment feature names dominate top-5."""
    if leakage_keywords is None:
        leakage_keywords = ["post", "after", "response", "outcome", "target"]
    top = shap_result.get("top_features", [])[:5]
    return any(
        any(kw in entry["feature"].lower() for kw in leakage_keywords)
        for entry in top
    )


def stability_summary(shap_results_multi_seed: list[dict]) -> dict:
    """Check if top-3 features are consistent across multiple seed runs."""
    if len(shap_results_multi_seed) < 2:
        return {"stable": True, "consistent_top3": []}

    rankings = [
        set(r["feature"] for r in result.get("top_features", [])[:3])
        for result in shap_results_multi_seed
    ]
    consistent = rankings[0].copy()
    for r in rankings[1:]:
        consistent &= r

    return {"stable": len(consistent) >= 2, "consistent_top3": list(consistent)}
