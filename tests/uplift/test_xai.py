from __future__ import annotations

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.uplift.xai import diagnose_xai_feature_semantics, explain_cached_uplift_model


class WarningModel:
    learner_family = "two_model"
    base_estimator = "lightgbm"
    feature_columns = ["feature_a", "feature_b"]

    def predict_uplift(self, frame: pd.DataFrame) -> np.ndarray:
        warnings.warn(
            "X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            UserWarning,
            stacklevel=2,
        )
        return frame["feature_a"].to_numpy(dtype=float) * 0.1


def test_cached_model_xai_suppresses_lightgbm_feature_name_warning(tmp_path: Path):
    model_path = tmp_path / "model.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(WarningModel(), handle)

    features = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(6)],
            "feature_a": [1, 2, 3, 4, 5, 6],
            "feature_b": [6, 5, 4, 3, 2, 1],
        }
    )
    scores = pd.DataFrame({"client_id": [f"c{i}" for i in range(6)]})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = explain_cached_uplift_model(model_path, features, scores)

    assert result["method"] == "cached_model_permutation"
    assert result["global_top_features"]
    assert not any("valid feature names" in str(warning.message) for warning in caught)


def test_xai_flags_age_dominance_without_behavioral_top_features():
    top_features = [
        {"feature": "age_clean", "mean_abs_uplift_change": 0.01},
        {"feature": "issue_year", "mean_abs_uplift_change": 0.001},
        {"feature": "gender_F", "mean_abs_uplift_change": 0.0005},
    ]

    result = diagnose_xai_feature_semantics(top_features)

    assert result["age_dominance_warning"] is True
    assert result["behavioral_top5_present"] is False


def test_xai_accepts_behavioral_driver_in_top_features():
    top_features = [
        {"feature": "purchase_txn_count_60d", "mean_abs_uplift_change": 0.01},
        {"feature": "age_clean", "mean_abs_uplift_change": 0.001},
    ]

    result = diagnose_xai_feature_semantics(top_features)

    assert result["age_dominance_warning"] is False
    assert result["behavioral_top5_present"] is True


def test_xai_still_flags_age_when_behavioral_feature_is_only_secondary():
    top_features = [
        {"feature": "age_clean", "mean_abs_uplift_change": 0.01},
        {"feature": "purchase_txn_count_60d", "mean_abs_uplift_change": 0.001},
    ]

    result = diagnose_xai_feature_semantics(top_features)

    assert result["age_dominance_warning"] is True
    assert result["top_feature_is_age"] is True
    assert result["behavioral_top5_present"] is True
