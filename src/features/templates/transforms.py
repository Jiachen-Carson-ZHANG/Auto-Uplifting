"""
Simple single-column transform templates.
"""
from __future__ import annotations
from typing import List, Optional
import numpy as np
import pandas as pd


def log1p_transform(
    df: pd.DataFrame,
    *,
    input_col: str,
    output_col: str,
) -> pd.DataFrame:
    """Apply np.log1p to a column."""
    df = df.copy()
    df[output_col] = np.log1p(df[input_col].fillna(0))
    return df


def clip_transform(
    df: pd.DataFrame,
    *,
    input_col: str,
    lower: float,
    upper: float,
    output_col: str,
) -> pd.DataFrame:
    """Clip a column to [lower, upper]."""
    df = df.copy()
    df[output_col] = df[input_col].clip(lower=lower, upper=upper)
    return df


def bucketize_transform(
    df: pd.DataFrame,
    *,
    input_col: str,
    bins: List[float],
    labels: Optional[List[str]] = None,
    output_col: Optional[str] = None,
) -> pd.DataFrame:
    """Bucketize a column using pd.cut."""
    df = df.copy()
    out = output_col or f"{input_col}_bucket"
    df[out] = pd.cut(df[input_col], bins=bins, labels=labels, include_lowest=True)
    return df


def is_missing_transform(
    df: pd.DataFrame,
    *,
    input_col: str,
    output_col: str,
) -> pd.DataFrame:
    """Create binary indicator for missing values."""
    df = df.copy()
    df[output_col] = df[input_col].isnull().astype(int)
    return df
