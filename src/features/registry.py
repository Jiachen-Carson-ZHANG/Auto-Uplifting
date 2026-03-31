"""
Template registry for bounded feature engineering.

Maps template names to implementation functions.
Template functions follow: (df: pd.DataFrame, **params) -> pd.DataFrame
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional
import pandas as pd


class TemplateRegistry:
    """Registry of named feature template functions."""

    def __init__(self) -> None:
        self._templates: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None:
        """Register a template function under the given name."""
        self._templates[name] = fn

    def get(self, name: str) -> Optional[Callable]:
        """Return template function or None if not registered."""
        return self._templates.get(name)

    def list_templates(self) -> List[str]:
        """Return sorted list of all registered template names."""
        return sorted(self._templates.keys())

    def execute(
        self, name: str, df: pd.DataFrame, params: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Look up and call a template function.

        Raises KeyError if the template is not registered.
        """
        fn = self._templates.get(name)
        if fn is None:
            raise KeyError(f"Template '{name}' not registered. Available: {self.list_templates()}")
        return fn(df, **params)


def build_default_registry() -> TemplateRegistry:
    """
    Build a registry pre-loaded with all shipped template functions.

    Called by FeatureEngineeringAgent if no registry is provided.
    """
    from src.features.templates.customer import (
        rfm_recency,
        rfm_frequency,
        rfm_monetary,
    )
    from src.features.templates.order import (
        avg_order_value,
        basket_size,
        category_diversity,
    )
    from src.features.templates.temporal import (
        days_since,
        count_in_window,
        sum_in_window,
        mean_in_window,
        nunique_in_window,
    )
    from src.features.templates.transforms import (
        log1p_transform,
        clip_transform,
        bucketize_transform,
        is_missing_transform,
    )
    from src.features.templates.composites import (
        safe_divide,
        subtract_cols,
        add_cols,
        multiply_cols,
        ratio_to_baseline,
    )

    reg = TemplateRegistry()

    # Customer / RFM
    reg.register("rfm_recency", rfm_recency)
    reg.register("rfm_frequency", rfm_frequency)
    reg.register("rfm_monetary", rfm_monetary)

    # Order / basket
    reg.register("avg_order_value", avg_order_value)
    reg.register("basket_size", basket_size)
    reg.register("category_diversity", category_diversity)

    # Temporal / windowed
    reg.register("days_since", days_since)
    reg.register("count_in_window", count_in_window)
    reg.register("sum_in_window", sum_in_window)
    reg.register("mean_in_window", mean_in_window)
    reg.register("nunique_in_window", nunique_in_window)

    # Transforms
    reg.register("log1p", log1p_transform)
    reg.register("clip", clip_transform)
    reg.register("bucketize", bucketize_transform)
    reg.register("is_missing", is_missing_transform)

    # Composites
    reg.register("safe_divide", safe_divide)
    reg.register("subtract", subtract_cols)
    reg.register("add", add_cols)
    reg.register("multiply", multiply_cols)
    reg.register("ratio_to_baseline", ratio_to_baseline)

    return reg
