"""
Time-based feature templates.

All windowed templates require entity_key + time_col — leakage defense by construction.
"""
from __future__ import annotations
import pandas as pd


def days_since(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    reference_date: str | None = None,
    output_col: str = "days_since",
) -> pd.DataFrame:
    """Days between each row's timestamp and a reference date (default: max date)."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    ref = pd.to_datetime(reference_date) if reference_date else df[time_col].max()
    df[output_col] = (ref - df[time_col]).dt.days
    return df


def count_in_window(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    window_days: int,
    output_col: str = "count_in_window",
) -> pd.DataFrame:
    """Count of rows per entity within window_days of the max date."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    cutoff = df[time_col].max()
    start = cutoff - pd.Timedelta(days=window_days)
    mask = df[time_col] >= start
    counts = df.loc[mask].groupby(entity_key)[time_col].transform("count")
    df[output_col] = 0
    df.loc[mask, output_col] = counts
    return df


def sum_in_window(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    value_col: str,
    window_days: int,
    output_col: str = "sum_in_window",
) -> pd.DataFrame:
    """Sum of value_col per entity within window_days of the max date."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    cutoff = df[time_col].max()
    start = cutoff - pd.Timedelta(days=window_days)
    mask = df[time_col] >= start
    sums = df.loc[mask].groupby(entity_key)[value_col].transform("sum")
    df[output_col] = 0.0
    df.loc[mask, output_col] = sums
    return df


def mean_in_window(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    value_col: str,
    window_days: int,
    output_col: str = "mean_in_window",
) -> pd.DataFrame:
    """Mean of value_col per entity within window_days of the max date."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    cutoff = df[time_col].max()
    start = cutoff - pd.Timedelta(days=window_days)
    mask = df[time_col] >= start
    means = df.loc[mask].groupby(entity_key)[value_col].transform("mean")
    df[output_col] = 0.0
    df.loc[mask, output_col] = means
    return df


def nunique_in_window(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    value_col: str,
    window_days: int,
    output_col: str = "nunique_in_window",
) -> pd.DataFrame:
    """Count of unique values of value_col per entity within window."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    cutoff = df[time_col].max()
    start = cutoff - pd.Timedelta(days=window_days)
    mask = df[time_col] >= start

    # nunique doesn't have a transform shortcut — use map
    nuniq = df.loc[mask].groupby(entity_key)[value_col].nunique()
    df[output_col] = df[entity_key].map(nuniq).fillna(0).astype(int)
    # Zero out rows outside the window
    df.loc[~mask, output_col] = 0
    return df
