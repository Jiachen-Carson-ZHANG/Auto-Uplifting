"""
RFM (Recency, Frequency, Monetary) customer features.

All windowed templates require entity_key + time_col — leakage defense by construction.
"""
from __future__ import annotations
import pandas as pd


def rfm_recency(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    cutoff_col: str | None = None,
    output_col: str = "recency_days",
) -> pd.DataFrame:
    """Days since each entity's most recent transaction."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    if cutoff_col and cutoff_col in df.columns:
        cutoff = pd.to_datetime(df[cutoff_col])
    else:
        cutoff = df[time_col].max()

    last_txn = df.groupby(entity_key)[time_col].transform("max")
    df[output_col] = (cutoff - last_txn).dt.days
    return df


def rfm_frequency(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    window_days: int = 365,
    output_col: str = "frequency",
) -> pd.DataFrame:
    """Transaction count per entity within the window."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    cutoff = df[time_col].max()
    start = cutoff - pd.Timedelta(days=window_days)
    mask = df[time_col] >= start

    counts = (
        df.loc[mask]
        .groupby(entity_key)[time_col]
        .transform("count")
    )
    df[output_col] = 0
    df.loc[mask, output_col] = counts
    return df


def rfm_monetary(
    df: pd.DataFrame,
    *,
    entity_key: str,
    time_col: str,
    amount_col: str,
    window_days: int = 365,
    output_col: str = "monetary",
) -> pd.DataFrame:
    """Sum of amounts per entity within the window."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    cutoff = df[time_col].max()
    start = cutoff - pd.Timedelta(days=window_days)
    mask = df[time_col] >= start

    sums = (
        df.loc[mask]
        .groupby(entity_key)[amount_col]
        .transform("sum")
    )
    df[output_col] = 0.0
    df.loc[mask, output_col] = sums
    return df
