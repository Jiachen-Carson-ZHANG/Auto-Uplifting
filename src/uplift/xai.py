"""Optional SHAP explainability helpers for PR2-style evaluation."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def explain_score_feature_associations(
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    *,
    entity_key: str = "client_id",
    top_n: int = 10,
) -> dict:
    """Return model-agnostic global/local explanations from scores plus features.

    This is a fast fallback when fitted model files are unavailable. It does not
    claim causality; it shows which pre-treatment features move with predicted
    uplift and surfaces representative customers for report inspection.
    """
    if features_df.empty or scores_df.empty or "uplift" not in scores_df.columns:
        return {
            "method": "score_feature_association",
            "global_top_features": [],
            "representative_cases": {},
            "n_rows_used": 0,
        }

    if entity_key in features_df.columns and entity_key in scores_df.columns:
        merged = scores_df[[entity_key, "uplift"]].merge(
            features_df,
            on=entity_key,
            how="inner",
        )
    else:
        merged = pd.concat(
            [
                scores_df[["uplift"]].reset_index(drop=True),
                features_df.reset_index(drop=True),
            ],
            axis=1,
        )
    if merged.empty:
        return {
            "method": "score_feature_association",
            "global_top_features": [],
            "representative_cases": {},
            "n_rows_used": 0,
        }

    excluded = {entity_key, "uplift", "target", "treatment_flg"}
    rows: list[dict] = []
    for column in merged.columns:
        if column in excluded:
            continue
        values = pd.to_numeric(merged[column], errors="coerce")
        if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
            continue
        association = values.corr(merged["uplift"], method="spearman")
        if pd.isna(association):
            continue
        rows.append(
            {
                "feature": column,
                "spearman_with_uplift": round(float(association), 4),
                "direction": "higher_feature_higher_uplift"
                if association >= 0
                else "higher_feature_lower_uplift",
            }
        )
    rows.sort(key=lambda row: abs(row["spearman_with_uplift"]), reverse=True)
    ranked = merged.sort_values("uplift", ascending=False).reset_index(drop=True)
    case_columns = [column for column in [entity_key, "uplift"] if column in ranked.columns]
    return {
        "method": "score_feature_association",
        "global_top_features": rows[:top_n],
        "representative_cases": {
            "highest_uplift": ranked.head(3)[case_columns].to_dict(orient="records"),
            "lowest_uplift": ranked.tail(3)[case_columns].to_dict(orient="records"),
            "near_boundary": ranked.iloc[
                max(0, len(ranked) // 2 - 1) : min(len(ranked), len(ranked) // 2 + 2)
            ][case_columns].to_dict(orient="records"),
        },
        "n_rows_used": len(merged),
    }


def explain_cached_uplift_model(
    model_path: Path,
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame | None = None,
    *,
    entity_key: str = "client_id",
    top_n: int = 10,
    max_samples: int = 500,
) -> dict:
    """Explain a cached fitted uplift model by prediction sensitivity.

    This works across the current learner families, including logistic,
    gradient boosting, and optional boosters, because it uses the model's own
    `predict_uplift()` interface rather than a model-specific explainer.
    """
    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    if features_df.empty:
        return {
            "method": "cached_model_permutation",
            "global_top_features": [],
            "representative_cases": {},
            "n_rows_used": 0,
            "model_family": getattr(model, "learner_family", None),
            "base_estimator": getattr(model, "base_estimator", None),
        }

    frame = _align_features_to_scores(features_df, scores_df, entity_key=entity_key)
    if frame.empty:
        return {
            "method": "cached_model_permutation",
            "global_top_features": [],
            "representative_cases": {},
            "n_rows_used": 0,
            "model_family": getattr(model, "learner_family", None),
            "base_estimator": getattr(model, "base_estimator", None),
        }
    feature_columns = [
        column
        for column in getattr(model, "feature_columns", [])
        if column in frame.columns
    ]
    if not feature_columns:
        return {
            "method": "cached_model_permutation",
            "global_top_features": [],
            "representative_cases": {},
            "n_rows_used": len(frame),
            "model_family": getattr(model, "learner_family", None),
            "base_estimator": getattr(model, "base_estimator", None),
        }

    sample = frame.sample(
        min(max_samples, len(frame)),
        random_state=42,
    )
    base_uplift = np.asarray(model.predict_uplift(sample))
    rows: list[dict] = []
    for column in feature_columns:
        values = pd.to_numeric(sample[column], errors="coerce")
        if values.notna().sum() < 2 or values.nunique(dropna=True) < 2:
            continue
        permuted = sample.copy()
        permuted[column] = values.sample(frac=1.0, random_state=42).to_numpy()
        permuted_uplift = np.asarray(model.predict_uplift(permuted))
        importance = float(np.mean(np.abs(base_uplift - permuted_uplift)))
        association = values.reset_index(drop=True).corr(
            pd.Series(base_uplift),
            method="spearman",
        )
        rows.append(
            {
                "feature": column,
                "mean_abs_uplift_change": round(importance, 6),
                "spearman_with_uplift": None
                if pd.isna(association)
                else round(float(association), 4),
                "direction": _association_direction(association),
            }
        )

    rows.sort(key=lambda row: row["mean_abs_uplift_change"], reverse=True)
    explanation_frame = sample.copy()
    explanation_frame["uplift"] = base_uplift
    ranked = explanation_frame.sort_values("uplift", ascending=False).reset_index(drop=True)
    top_case_features = [row["feature"] for row in rows[:3]]
    case_columns = [
        column
        for column in [entity_key, "uplift", *top_case_features]
        if column in ranked.columns
    ]
    return {
        "method": "cached_model_permutation",
        "global_top_features": rows[:top_n],
        "representative_cases": {
            "highest_uplift": ranked.head(3)[case_columns].to_dict(orient="records"),
            "lowest_uplift": ranked.tail(3)[case_columns].to_dict(orient="records"),
            "near_boundary": ranked.iloc[
                max(0, len(ranked) // 2 - 1) : min(len(ranked), len(ranked) // 2 + 2)
            ][case_columns].to_dict(orient="records"),
        },
        "n_rows_used": len(sample),
        "model_family": getattr(model, "learner_family", None),
        "base_estimator": getattr(model, "base_estimator", None),
        "explanation_scope": "prediction_sensitivity_and_representative_cases",
        "limitations": (
            "Permutation sensitivity explains model behavior; it is not causal "
            "proof of treatment effect."
        ),
    }


def _align_features_to_scores(
    features_df: pd.DataFrame,
    scores_df: pd.DataFrame | None,
    *,
    entity_key: str,
) -> pd.DataFrame:
    if (
        scores_df is not None
        and not scores_df.empty
        and entity_key in features_df.columns
        and entity_key in scores_df.columns
    ):
        return scores_df[[entity_key]].merge(features_df, on=entity_key, how="inner")
    return features_df.copy()


def _association_direction(association: float) -> str:
    if pd.isna(association):
        return "unknown"
    if association >= 0:
        return "higher_feature_higher_uplift"
    return "higher_feature_lower_uplift"


def run_shap_two_model(
    model_t_path: Path,
    model_c_path: Path,
    features_df: pd.DataFrame,
    max_samples: int = 2000,
) -> dict:
    """Explain separate treatment/control tree models and rank uplift gaps."""
    import shap

    with model_t_path.open("rb") as handle:
        model_t = pickle.load(handle)
    with model_c_path.open("rb") as handle:
        model_c = pickle.load(handle)

    sample = features_df.sample(min(max_samples, len(features_df)), random_state=42)
    shap_t = np.abs(shap.TreeExplainer(model_t).shap_values(sample)).mean(axis=0)
    shap_c = np.abs(shap.TreeExplainer(model_c).shap_values(sample)).mean(axis=0)
    gaps = shap_t - shap_c
    rows = [
        {
            "feature": feature,
            "shap_treated": round(float(shap_t[index]), 4),
            "shap_control": round(float(shap_c[index]), 4),
            "gap": round(float(gaps[index]), 4),
        }
        for index, feature in enumerate(features_df.columns.tolist())
    ]
    rows.sort(key=lambda row: abs(row["gap"]), reverse=True)
    return {"method": "two_model", "top_features": rows[:10], "n_samples_used": len(sample)}


def run_shap_solo_model(
    model_path: Path,
    features_df: pd.DataFrame,
    treatment_col: str = "treatment_flg",
    max_samples: int = 2000,
) -> dict:
    """Explain a solo tree model under treated and control counterfactuals."""
    import shap

    with model_path.open("rb") as handle:
        model = pickle.load(handle)

    sample = features_df.sample(min(max_samples, len(features_df)), random_state=42)
    treated = sample.copy()
    control = sample.copy()
    treated[treatment_col] = 1
    control[treatment_col] = 0

    explainer = shap.TreeExplainer(model)
    shap_t = np.abs(explainer.shap_values(treated)).mean(axis=0)
    shap_c = np.abs(explainer.shap_values(control)).mean(axis=0)
    gaps = shap_t - shap_c
    rows = [
        {
            "feature": feature,
            "shap_treated": round(float(shap_t[index]), 4),
            "shap_control": round(float(shap_c[index]), 4),
            "gap": round(float(gaps[index]), 4),
        }
        for index, feature in enumerate(treated.columns.tolist())
    ]
    rows.sort(key=lambda row: abs(row["gap"]), reverse=True)
    return {"method": "solo_model", "top_features": rows[:10], "n_samples_used": len(sample)}


def check_leakage_signals(
    shap_result: dict,
    leakage_keywords: list[str] | None = None,
) -> bool:
    """Flag likely leakage if post-treatment feature names dominate top SHAP rows."""
    keywords = leakage_keywords or [
        "post",
        "after",
        "response",
        "outcome",
        "target",
        "redeem",
    ]
    top_features = shap_result.get("top_features", [])[:5]
    return any(
        any(keyword in row["feature"].lower() for keyword in keywords)
        for row in top_features
    )


def stability_summary(shap_results_multi_seed: list[dict]) -> dict:
    """Check whether top SHAP features are stable across multiple runs."""
    if len(shap_results_multi_seed) < 2:
        return {"stable": True, "consistent_top3": []}
    rankings = [
        {row["feature"] for row in result.get("top_features", [])[:3]}
        for result in shap_results_multi_seed
    ]
    common = rankings[0].copy()
    for ranking in rankings[1:]:
        common &= ranking
    return {"stable": len(common) >= 2, "consistent_top3": sorted(common)}
