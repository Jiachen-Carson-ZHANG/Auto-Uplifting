"""Tests for src/features/registry.py — TemplateRegistry."""
import pytest
import pandas as pd

from src.features.registry import TemplateRegistry, build_default_registry


class TestTemplateRegistry:
    def test_register_and_get(self):
        reg = TemplateRegistry()
        fn = lambda df, **kw: df
        reg.register("my_template", fn)
        assert reg.get("my_template") is fn

    def test_get_missing_returns_none(self):
        reg = TemplateRegistry()
        assert reg.get("nonexistent") is None

    def test_list_templates_sorted(self):
        reg = TemplateRegistry()
        reg.register("z_template", lambda df: df)
        reg.register("a_template", lambda df: df)
        assert reg.list_templates() == ["a_template", "z_template"]

    def test_execute_calls_function(self):
        reg = TemplateRegistry()
        reg.register("add_col", lambda df, *, val=1: df.assign(new=val))
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = reg.execute("add_col", df, {"val": 42})
        assert "new" in result.columns
        assert result["new"].iloc[0] == 42

    def test_execute_missing_raises_key_error(self):
        reg = TemplateRegistry()
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(KeyError, match="not_here"):
            reg.execute("not_here", df, {})


class TestBuildDefaultRegistry:
    def test_all_20_templates_registered(self):
        reg = build_default_registry()
        templates = reg.list_templates()
        assert len(templates) == 20

    def test_expected_names_present(self):
        reg = build_default_registry()
        templates = reg.list_templates()
        expected = [
            "rfm_recency", "rfm_frequency", "rfm_monetary",
            "avg_order_value", "basket_size", "category_diversity",
            "days_since", "count_in_window", "sum_in_window", "mean_in_window", "nunique_in_window",
            "log1p", "clip", "bucketize", "is_missing",
            "safe_divide", "subtract", "add", "multiply", "ratio_to_baseline",
        ]
        for name in expected:
            assert name in templates, f"Missing template: {name}"

    def test_each_template_is_callable(self):
        reg = build_default_registry()
        for name in reg.list_templates():
            assert callable(reg.get(name))
