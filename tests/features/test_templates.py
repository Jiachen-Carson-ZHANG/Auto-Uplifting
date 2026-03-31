"""Tests for src/features/templates/ — all template functions."""
import pandas as pd
import numpy as np
import pytest

from src.features.templates.customer import rfm_recency, rfm_frequency, rfm_monetary
from src.features.templates.order import avg_order_value, basket_size, category_diversity
from src.features.templates.temporal import (
    days_since, count_in_window, sum_in_window, mean_in_window, nunique_in_window,
)
from src.features.templates.transforms import (
    log1p_transform, clip_transform, bucketize_transform, is_missing_transform,
)
from src.features.templates.composites import (
    safe_divide, subtract_cols, add_cols, multiply_cols, ratio_to_baseline,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ecommerce_df():
    """Simple ecommerce-like DataFrame for customer/order/temporal tests."""
    return pd.DataFrame({
        "customer_id": ["A", "A", "B", "B", "B"],
        "order_id": ["o1", "o2", "o3", "o3", "o4"],
        "date": pd.to_datetime(["2025-01-01", "2025-06-01", "2025-03-01", "2025-03-01", "2025-12-01"]),
        "amount": [100.0, 200.0, 50.0, 75.0, 300.0],
        "item": ["X", "Y", "X", "Z", "Y"],
        "category": ["elec", "elec", "book", "cloth", "elec"],
    })


@pytest.fixture
def numeric_df():
    """Simple numeric DataFrame for transforms/composites."""
    return pd.DataFrame({
        "a": [10.0, 20.0, 0.0, 50.0],
        "b": [5.0, 0.0, 10.0, 25.0],
        "c": [1.0, np.nan, 3.0, 4.0],
    })


# ── Customer templates ────────────────────────────────────────────────

class TestRFMRecency:
    def test_output_col_exists(self, ecommerce_df):
        result = rfm_recency(ecommerce_df, entity_key="customer_id", time_col="date")
        assert "recency_days" in result.columns

    def test_does_not_mutate_input(self, ecommerce_df):
        original_cols = list(ecommerce_df.columns)
        rfm_recency(ecommerce_df, entity_key="customer_id", time_col="date")
        assert list(ecommerce_df.columns) == original_cols

    def test_row_count_preserved(self, ecommerce_df):
        result = rfm_recency(ecommerce_df, entity_key="customer_id", time_col="date")
        assert len(result) == len(ecommerce_df)


class TestRFMFrequency:
    def test_output_col_exists(self, ecommerce_df):
        result = rfm_frequency(ecommerce_df, entity_key="customer_id", time_col="date")
        assert "frequency" in result.columns

    def test_counts_within_window(self, ecommerce_df):
        result = rfm_frequency(
            ecommerce_df, entity_key="customer_id", time_col="date", window_days=365
        )
        assert result["frequency"].sum() > 0


class TestRFMMonetary:
    def test_output_col_exists(self, ecommerce_df):
        result = rfm_monetary(
            ecommerce_df, entity_key="customer_id", time_col="date", amount_col="amount"
        )
        assert "monetary" in result.columns

    def test_sums_within_window(self, ecommerce_df):
        result = rfm_monetary(
            ecommerce_df, entity_key="customer_id", time_col="date",
            amount_col="amount", window_days=365,
        )
        assert result["monetary"].sum() > 0


# ── Order templates ───────────────────────────────────────────────────

class TestAvgOrderValue:
    def test_output_col_exists(self, ecommerce_df):
        result = avg_order_value(
            ecommerce_df, entity_key="customer_id",
            amount_col="amount", order_id_col="order_id",
        )
        assert "avg_order_value" in result.columns

    def test_values_positive(self, ecommerce_df):
        result = avg_order_value(
            ecommerce_df, entity_key="customer_id",
            amount_col="amount", order_id_col="order_id",
        )
        assert (result["avg_order_value"] >= 0).all()


class TestBasketSize:
    def test_output_col_exists(self, ecommerce_df):
        result = basket_size(
            ecommerce_df, entity_key="customer_id",
            order_id_col="order_id", item_col="item",
        )
        assert "avg_basket_size" in result.columns


class TestCategoryDiversity:
    def test_output_col_exists(self, ecommerce_df):
        result = category_diversity(
            ecommerce_df, entity_key="customer_id", category_col="category"
        )
        assert "category_nunique" in result.columns

    def test_nunique_correct(self, ecommerce_df):
        result = category_diversity(
            ecommerce_df, entity_key="customer_id", category_col="category"
        )
        # Customer A: elec (2 rows, 1 unique). B: book, cloth, elec (3 unique)
        b_rows = result[result["customer_id"] == "B"]
        assert b_rows["category_nunique"].iloc[0] == 3


# ── Temporal templates ────────────────────────────────────────────────

class TestDaysSince:
    def test_output_col_exists(self, ecommerce_df):
        result = days_since(ecommerce_df, entity_key="customer_id", time_col="date")
        assert "days_since" in result.columns

    def test_most_recent_is_zero(self, ecommerce_df):
        result = days_since(ecommerce_df, entity_key="customer_id", time_col="date")
        assert result["days_since"].min() == 0


class TestCountInWindow:
    def test_output_col_exists(self, ecommerce_df):
        result = count_in_window(
            ecommerce_df, entity_key="customer_id", time_col="date", window_days=180
        )
        assert "count_in_window" in result.columns


class TestSumInWindow:
    def test_output_col_exists(self, ecommerce_df):
        result = sum_in_window(
            ecommerce_df, entity_key="customer_id", time_col="date",
            value_col="amount", window_days=365,
        )
        assert "sum_in_window" in result.columns


class TestMeanInWindow:
    def test_output_col_exists(self, ecommerce_df):
        result = mean_in_window(
            ecommerce_df, entity_key="customer_id", time_col="date",
            value_col="amount", window_days=365,
        )
        assert "mean_in_window" in result.columns


class TestNuniqueInWindow:
    def test_output_col_exists(self, ecommerce_df):
        result = nunique_in_window(
            ecommerce_df, entity_key="customer_id", time_col="date",
            value_col="category", window_days=365,
        )
        assert "nunique_in_window" in result.columns


# ── Transform templates ───────────────────────────────────────────────

class TestLog1pTransform:
    def test_output_col_exists(self, numeric_df):
        result = log1p_transform(numeric_df, input_col="a", output_col="log_a")
        assert "log_a" in result.columns

    def test_values_correct(self, numeric_df):
        result = log1p_transform(numeric_df, input_col="a", output_col="log_a")
        expected = np.log1p(numeric_df["a"].fillna(0))
        pd.testing.assert_series_equal(result["log_a"], expected, check_names=False)


class TestClipTransform:
    def test_clips_values(self, numeric_df):
        result = clip_transform(numeric_df, input_col="a", lower=5, upper=30, output_col="clipped")
        assert result["clipped"].min() >= 5
        assert result["clipped"].max() <= 30


class TestBucketizeTransform:
    def test_output_col_exists(self, numeric_df):
        result = bucketize_transform(
            numeric_df, input_col="a", bins=[0, 15, 30, 60], output_col="bucket"
        )
        assert "bucket" in result.columns


class TestIsMissingTransform:
    def test_detects_nulls(self, numeric_df):
        result = is_missing_transform(numeric_df, input_col="c", output_col="c_missing")
        assert result["c_missing"].sum() == 1  # one NaN in column c


# ── Composite templates ───────────────────────────────────────────────

class TestSafeDivide:
    def test_handles_zero_denominator(self, numeric_df):
        result = safe_divide(
            numeric_df, numerator_col="a", denominator_col="b", output_col="ratio"
        )
        # b has a 0 at index 1 — should be filled with 0.0
        assert result["ratio"].iloc[1] == 0.0

    def test_normal_division(self, numeric_df):
        result = safe_divide(
            numeric_df, numerator_col="a", denominator_col="b", output_col="ratio"
        )
        assert result["ratio"].iloc[0] == pytest.approx(2.0)


class TestSubtractCols:
    def test_subtraction(self, numeric_df):
        result = subtract_cols(numeric_df, col_a="a", col_b="b", output_col="diff")
        assert result["diff"].iloc[0] == 5.0


class TestAddCols:
    def test_addition(self, numeric_df):
        result = add_cols(numeric_df, col_a="a", col_b="b", output_col="total")
        assert result["total"].iloc[0] == 15.0


class TestMultiplyCols:
    def test_multiplication(self, numeric_df):
        result = multiply_cols(numeric_df, col_a="a", col_b="b", output_col="product")
        assert result["product"].iloc[0] == 50.0


class TestRatioToBaseline:
    def test_ratio(self, numeric_df):
        result = ratio_to_baseline(
            numeric_df, col="a", baseline_col="b", output_col="ratio"
        )
        assert result["ratio"].iloc[0] == pytest.approx(2.0)

    def test_zero_baseline_filled(self, numeric_df):
        result = ratio_to_baseline(
            numeric_df, col="a", baseline_col="b", output_col="ratio"
        )
        assert result["ratio"].iloc[1] == 0.0
