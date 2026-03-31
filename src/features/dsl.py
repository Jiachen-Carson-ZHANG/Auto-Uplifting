"""
DSL operator surface and validation for bounded feature engineering.

The 14 operators define the expressible surface for Phase 1.
Time-requiring ops enforce entity_key, time_col, and window params
as a primary leakage defense.
"""
from __future__ import annotations
from typing import Any, Callable, Dict, List, Set, Union
import numpy as np

from src.models.feature_engineering import (
    CompositeFeatureSpec,
    TransformFeatureSpec,
)

# ── Operator surface ────────────────────────────────────────────────

VALID_OPS: Set[str] = {
    "safe_divide",
    "subtract",
    "add",
    "multiply",
    "ratio_to_baseline",
    "log1p",
    "clip",
    "bucketize",
    "is_missing",
    "days_since",
    "count_in_window",
    "sum_in_window",
    "mean_in_window",
    "nunique_in_window",
}

TIME_REQUIRING_OPS: Set[str] = {
    "days_since",
    "count_in_window",
    "sum_in_window",
    "mean_in_window",
    "nunique_in_window",
}

# ── Post-op registry ───────────────────────────────────────────────

_POST_OPS: Dict[str, Callable] = {
    "clip_0_1": lambda s: s.clip(0, 1),
    "log1p": lambda s: np.log1p(s),
    "abs": lambda s: s.abs(),
    "fillna_0": lambda s: s.fillna(0),
}

VALID_POST_OPS: Set[str] = set(_POST_OPS.keys())


# ── Validation ─────────────────────────────────────────────────────

def validate_dsl_config(
    spec: Union[TransformFeatureSpec, CompositeFeatureSpec],
) -> List[str]:
    """
    Validate a DSL config spec. Returns list of error strings (empty = valid).
    """
    errors: List[str] = []

    # 1. op must be in VALID_OPS
    if spec.op not in VALID_OPS:
        errors.append(f"Unknown operator '{spec.op}'. Valid: {sorted(VALID_OPS)}")

    # 2. Time-requiring ops need entity_key, time_col, window
    if spec.op in TIME_REQUIRING_OPS:
        params = spec.params
        for required in ("entity_key", "time_col"):
            if required not in params:
                errors.append(
                    f"Operator '{spec.op}' requires '{required}' in params."
                )
        # days_since doesn't need a window, but the windowed aggregations do
        if spec.op != "days_since" and "window_days" not in params:
            errors.append(
                f"Operator '{spec.op}' requires 'window_days' in params."
            )

    # 3. Composite-specific: inputs must be non-empty
    if isinstance(spec, CompositeFeatureSpec):
        if not spec.inputs:
            errors.append("Composite spec requires at least one input.")
        # 4. Post-ops must be recognized
        for p in spec.post:
            if p not in VALID_POST_OPS:
                errors.append(
                    f"Unknown post-op '{p}'. Valid: {sorted(VALID_POST_OPS)}"
                )

    return errors


def parse_post_ops(post: List[str]) -> List[Callable]:
    """
    Resolve post-op names to callables.

    Raises ValueError for unrecognized post-ops.
    """
    callables: List[Callable] = []
    for name in post:
        fn = _POST_OPS.get(name)
        if fn is None:
            raise ValueError(
                f"Unknown post-op '{name}'. Valid: {sorted(VALID_POST_OPS)}"
            )
        callables.append(fn)
    return callables
