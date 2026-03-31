"""Tests for src/features/executor.py — BoundedExecutor."""
import pandas as pd
import numpy as np
import pytest

from src.features.executor import BoundedExecutor
from src.features.registry import build_default_registry
from src.models.feature_engineering import (
    CompositeFeatureSpec,
    FeatureDecision,
    TemplateFeatureSpec,
    TransformFeatureSpec,
)


@pytest.fixture
def executor():
    return BoundedExecutor(build_default_registry())


@pytest.fixture
def df():
    return pd.DataFrame({
        "customer_id": ["A", "A", "B", "B"],
        "date": pd.to_datetime(["2025-01-01", "2025-06-01", "2025-03-01", "2025-12-01"]),
        "amount": [100.0, 200.0, 50.0, 300.0],
        "price": [10.0, 20.0, 5.0, 30.0],
    })


class TestExecuteTemplate:
    def test_success(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="add", reasoning="test",
            feature_spec=TemplateFeatureSpec(
                template_name="rfm_recency",
                params={"entity_key": "customer_id", "time_col": "date"},
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "success"
        assert "recency_days" in result.produced_columns

    def test_unknown_template_fails(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="add", reasoning="test",
            feature_spec=TemplateFeatureSpec(template_name="nonexistent"),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "failed"
        assert result.failure_reason is not None


class TestExecuteTransform:
    def test_log1p(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="transform", reasoning="test",
            feature_spec=TransformFeatureSpec(
                input_col="amount", op="log1p", output_col="log_amount",
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "success"
        assert "log_amount" in result.produced_columns

    def test_clip(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="transform", reasoning="test",
            feature_spec=TransformFeatureSpec(
                input_col="amount", op="clip", output_col="clipped",
                params={"lower": 50, "upper": 200},
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "success"

    def test_invalid_op_fails(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="transform", reasoning="test",
            feature_spec=TransformFeatureSpec(
                input_col="amount", op="foobar", output_col="out",
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "failed"


class TestExecuteComposite:
    def test_safe_divide(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="composite", reasoning="test",
            feature_spec=CompositeFeatureSpec(
                name="amount_per_price", op="safe_divide",
                inputs=[{"ref": "amount"}, {"ref": "price"}],
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "success"
        assert "amount_per_price" in result.produced_columns

    def test_missing_inputs_fails(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="composite", reasoning="test",
            feature_spec=CompositeFeatureSpec(
                name="bad", op="safe_divide", inputs=[],
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "failed"


class TestExecuteDrop:
    def test_drop_returns_success(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="drop", reasoning="test",
            feature_spec=TransformFeatureSpec(
                input_col="price", op="log1p", output_col="unused",
            ),
        )
        _, result = executor.execute(df, decision)
        assert result.status == "success"
        assert "price" in result.dropped_columns


class TestExecuteNeverRaises:
    def test_no_spec_returns_failed(self, executor, df):
        decision = FeatureDecision(
            status="proposed", action="add", reasoning="test",
            feature_spec=None,
        )
        _, result = executor.execute(df, decision)
        assert result.status == "failed"
        assert "No feature_spec" in result.failure_reason
