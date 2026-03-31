"""Tests for src/features/dsl.py — DSL validation."""
import pytest

from src.features.dsl import (
    VALID_OPS,
    VALID_POST_OPS,
    parse_post_ops,
    validate_dsl_config,
)
from src.models.feature_engineering import CompositeFeatureSpec, TransformFeatureSpec


class TestValidateDslConfig:
    def test_valid_op_accepted(self):
        spec = TransformFeatureSpec(
            input_col="x", op="safe_divide", output_col="y"
        )
        assert validate_dsl_config(spec) == []

    def test_invalid_op_rejected(self):
        spec = TransformFeatureSpec(
            input_col="x", op="foobar", output_col="y"
        )
        errors = validate_dsl_config(spec)
        assert len(errors) == 1
        assert "foobar" in errors[0]

    def test_time_op_requires_entity_key(self):
        spec = TransformFeatureSpec(
            input_col="x",
            op="count_in_window",
            output_col="y",
            params={"time_col": "date", "window_days": 30},
        )
        errors = validate_dsl_config(spec)
        assert any("entity_key" in e for e in errors)

    def test_time_op_requires_time_col(self):
        spec = TransformFeatureSpec(
            input_col="x",
            op="days_since",
            output_col="y",
            params={"entity_key": "cid"},
        )
        errors = validate_dsl_config(spec)
        assert any("time_col" in e for e in errors)

    def test_time_op_requires_window(self):
        spec = TransformFeatureSpec(
            input_col="x",
            op="sum_in_window",
            output_col="y",
            params={"entity_key": "cid", "time_col": "date"},
        )
        errors = validate_dsl_config(spec)
        assert any("window_days" in e for e in errors)

    def test_days_since_no_window_needed(self):
        spec = TransformFeatureSpec(
            input_col="x",
            op="days_since",
            output_col="y",
            params={"entity_key": "cid", "time_col": "date"},
        )
        errors = validate_dsl_config(spec)
        assert errors == []

    def test_composite_inputs_required(self):
        spec = CompositeFeatureSpec(
            name="ratio", op="safe_divide", inputs=[]
        )
        errors = validate_dsl_config(spec)
        assert any("input" in e.lower() for e in errors)

    def test_unknown_post_op_rejected(self):
        spec = CompositeFeatureSpec(
            name="ratio",
            op="safe_divide",
            inputs=[{"ref": "a"}, {"ref": "b"}],
            post=["unknown_op"],
        )
        errors = validate_dsl_config(spec)
        assert any("unknown_op" in e for e in errors)


class TestParsePostOps:
    def test_valid_post_ops(self):
        fns = parse_post_ops(["clip_0_1", "log1p"])
        assert len(fns) == 2
        assert callable(fns[0])
        assert callable(fns[1])

    def test_unknown_post_op_raises(self):
        with pytest.raises(ValueError, match="unknown_op"):
            parse_post_ops(["unknown_op"])

    def test_empty_list(self):
        assert parse_post_ops([]) == []
