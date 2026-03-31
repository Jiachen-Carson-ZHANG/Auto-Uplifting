"""
BoundedExecutor — dispatches FeatureDecision to template functions via registry.

Runs in-process (no subprocess) — only pre-approved templates with validated params.
Never raises — returns FeatureExecutionResult(status="failed") on error.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.dsl import TIME_REQUIRING_OPS, VALID_OPS, parse_post_ops, validate_dsl_config
from src.features.registry import TemplateRegistry
from src.models.feature_engineering import (
    CompositeFeatureSpec,
    FeatureDecision,
    FeatureExecutionResult,
    TemplateFeatureSpec,
    TransformFeatureSpec,
)

logger = logging.getLogger(__name__)

# ── Op dispatch table for TransformFeatureSpec ──────────────────────

_TRANSFORM_OPS: Dict[str, Any] = {
    "log1p": lambda s, **kw: np.log1p(s.fillna(0)),
    "clip": lambda s, *, lower=0, upper=1, **kw: s.clip(lower=lower, upper=upper),
    "bucketize": lambda s, *, bins, labels=None, **kw: pd.cut(s, bins=bins, labels=labels, include_lowest=True),
    "is_missing": lambda s, **kw: s.isnull().astype(int),
}

_COMPOSITE_OPS: Dict[str, Any] = {
    "safe_divide": lambda a, b, **kw: (a / b.replace(0, float("nan"))).fillna(kw.get("fill_value", 0.0)),
    "subtract": lambda a, b, **kw: a - b,
    "add": lambda a, b, **kw: a + b,
    "multiply": lambda a, b, **kw: a * b,
    "ratio_to_baseline": lambda a, b, **kw: (a / b.replace(0, float("nan"))).fillna(0.0),
}


class BoundedExecutor:
    """Dispatches FeatureDecision to template functions or DSL ops."""

    def __init__(self, registry: TemplateRegistry) -> None:
        self._registry = registry

    def execute(
        self, df: pd.DataFrame, decision: FeatureDecision
    ) -> Tuple[Optional[pd.DataFrame], FeatureExecutionResult]:
        """
        Execute a feature decision against a DataFrame.

        Returns (result_df, FeatureExecutionResult). result_df is None on failure.
        Never raises.
        """
        try:
            original_cols = set(df.columns)
            spec = decision.feature_spec

            if decision.action == "drop" and spec is not None:
                drop_result = self._execute_drop(df, spec)
                return df, drop_result

            if spec is None:
                return None, FeatureExecutionResult(
                    status="failed",
                    failure_reason="No feature_spec provided.",
                )

            if isinstance(spec, TemplateFeatureSpec):
                result_df = self._execute_template(df, spec)
            elif isinstance(spec, TransformFeatureSpec):
                result_df = self._execute_transform(df, spec)
            elif isinstance(spec, CompositeFeatureSpec):
                result_df = self._execute_composite(df, spec)
            else:
                return None, FeatureExecutionResult(
                    status="failed",
                    failure_reason=f"Unsupported spec type: {type(spec).__name__}",
                )

            new_cols = sorted(set(result_df.columns) - original_cols)
            return result_df, FeatureExecutionResult(
                status="success",
                produced_columns=new_cols,
            )

        except Exception as exc:
            logger.warning("BoundedExecutor failed: %s", exc)
            return None, FeatureExecutionResult(
                status="failed",
                failure_reason=str(exc),
            )

    # ── Private dispatch methods ────────────────────────────────────

    def _execute_template(
        self, df: pd.DataFrame, spec: TemplateFeatureSpec
    ) -> pd.DataFrame:
        return self._registry.execute(spec.template_name, df, spec.params)

    def _execute_transform(
        self, df: pd.DataFrame, spec: TransformFeatureSpec
    ) -> pd.DataFrame:
        errors = validate_dsl_config(spec)
        if errors:
            raise ValueError(f"DSL validation failed: {errors}")

        # Temporal ops need the full DataFrame — route through template registry
        if spec.op in TIME_REQUIRING_OPS:
            template_name = spec.op  # temporal template names match op names
            params = {**spec.params, "output_col": spec.output_col}
            if spec.input_col:
                # input_col maps to the primary column (time_col or value_col)
                if spec.op == "days_since":
                    params.setdefault("time_col", spec.input_col)
                else:
                    params.setdefault("value_col", spec.input_col)
            return self._registry.execute(template_name, df, params)

        op_fn = _TRANSFORM_OPS.get(spec.op)
        if op_fn is None:
            raise ValueError(f"No in-process handler for transform op '{spec.op}'.")

        df = df.copy()
        df[spec.output_col] = op_fn(df[spec.input_col], **spec.params)
        return df

    def _execute_composite(
        self, df: pd.DataFrame, spec: CompositeFeatureSpec
    ) -> pd.DataFrame:
        errors = validate_dsl_config(spec)
        if errors:
            raise ValueError(f"DSL validation failed: {errors}")

        op_fn = _COMPOSITE_OPS.get(spec.op)
        if op_fn is None:
            raise ValueError(f"No in-process handler for composite op '{spec.op}'.")

        # Resolve inputs (first two as positional a, b)
        resolved: List[pd.Series] = []
        for inp in spec.inputs:
            if "ref" in inp:
                resolved.append(df[inp["ref"]])
            elif "literal" in inp:
                resolved.append(pd.Series(inp["literal"], index=df.index))
            else:
                raise ValueError(f"Unrecognized input format: {inp}")

        if len(resolved) < 2:
            raise ValueError("Composite ops require at least 2 inputs.")

        df = df.copy()
        extra = getattr(spec, "params", {}) or {}
        result = op_fn(resolved[0], resolved[1], **extra)

        # Apply post-ops
        for post_fn in parse_post_ops(spec.post):
            result = post_fn(result)

        df[spec.name] = result
        return df

    def _execute_drop(
        self, df: pd.DataFrame, spec: Any
    ) -> FeatureExecutionResult:
        """Drop columns specified in the spec."""
        cols_to_drop: List[str] = []
        if hasattr(spec, "params") and "columns" in spec.params:
            cols_to_drop = spec.params["columns"]
        elif hasattr(spec, "input_col"):
            cols_to_drop = [spec.input_col]

        dropped = [c for c in cols_to_drop if c in df.columns]
        return FeatureExecutionResult(
            status="success",
            dropped_columns=dropped,
        )
