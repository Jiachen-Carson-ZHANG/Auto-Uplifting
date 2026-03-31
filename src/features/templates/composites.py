"""
Multi-input composite feature templates.
"""
from __future__ import annotations
import pandas as pd


def safe_divide(
    df: pd.DataFrame,
    *,
    numerator_col: str,
    denominator_col: str,
    output_col: str,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Divide numerator by denominator, filling div-by-zero with fill_value."""
    df = df.copy()
    denom = df[denominator_col].replace(0, float("nan"))
    df[output_col] = (df[numerator_col] / denom).fillna(fill_value)
    return df


def subtract_cols(
    df: pd.DataFrame,
    *,
    col_a: str,
    col_b: str,
    output_col: str,
) -> pd.DataFrame:
    """Subtract col_b from col_a."""
    df = df.copy()
    df[output_col] = df[col_a] - df[col_b]
    return df


def add_cols(
    df: pd.DataFrame,
    *,
    col_a: str,
    col_b: str,
    output_col: str,
) -> pd.DataFrame:
    """Add col_a and col_b."""
    df = df.copy()
    df[output_col] = df[col_a] + df[col_b]
    return df


def multiply_cols(
    df: pd.DataFrame,
    *,
    col_a: str,
    col_b: str,
    output_col: str,
) -> pd.DataFrame:
    """Multiply col_a by col_b."""
    df = df.copy()
    df[output_col] = df[col_a] * df[col_b]
    return df


def ratio_to_baseline(
    df: pd.DataFrame,
    *,
    col: str,
    baseline_col: str,
    output_col: str,
) -> pd.DataFrame:
    """Ratio of col to baseline_col (safe division)."""
    df = df.copy()
    baseline = df[baseline_col].replace(0, float("nan"))
    df[output_col] = (df[col] / baseline).fillna(0.0)
    return df
