"""Tests for src/features/validator.py — FeatureValidator."""
import pandas as pd
import numpy as np
import pytest

from src.features.validator import FeatureValidator


@pytest.fixture
def validator():
    return FeatureValidator()


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0, 1, 0, 1, 0],
    })


class TestValidateResult:
    def test_valid_result_no_issues(self, validator, base_df):
        result_df = base_df.copy()
        result_df["new_feature"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        issues = validator.validate_result(base_df, result_df, "target")
        assert issues == []

    def test_row_count_changed(self, validator, base_df):
        result_df = base_df.iloc[:2].copy()
        issues = validator.validate_result(base_df, result_df, "target")
        assert any("Row count" in i for i in issues)

    def test_target_missing(self, validator, base_df):
        result_df = base_df.drop(columns=["target"])
        issues = validator.validate_result(base_df, result_df, "target")
        assert any("missing" in i.lower() for i in issues)

    def test_target_modified(self, validator, base_df):
        result_df = base_df.copy()
        result_df["target"] = [1, 1, 1, 1, 1]
        issues = validator.validate_result(base_df, result_df, "target")
        assert any("modified" in i.lower() for i in issues)

    def test_all_null_new_column(self, validator, base_df):
        result_df = base_df.copy()
        result_df["bad_col"] = np.nan
        issues = validator.validate_result(base_df, result_df, "target")
        assert any("all null" in i.lower() for i in issues)

    def test_constant_new_column(self, validator, base_df):
        result_df = base_df.copy()
        result_df["const_col"] = 42
        issues = validator.validate_result(base_df, result_df, "target")
        assert any("constant" in i.lower() for i in issues)

    def test_nan_explosion(self, validator, base_df):
        result_df = base_df.copy()
        result_df["feature_a"] = np.nan  # all rows NaN — 100% of rows
        issues = validator.validate_result(base_df, result_df, "target")
        assert any("NaN" in i for i in issues)
