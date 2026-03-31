"""
Order and basket features for ecommerce datasets.
"""
from __future__ import annotations
import pandas as pd


def avg_order_value(
    df: pd.DataFrame,
    *,
    entity_key: str,
    amount_col: str,
    order_id_col: str,
    output_col: str = "avg_order_value",
) -> pd.DataFrame:
    """Average order value per entity."""
    df = df.copy()
    order_totals = df.groupby([entity_key, order_id_col])[amount_col].sum().reset_index()
    aov = order_totals.groupby(entity_key)[amount_col].mean()
    df[output_col] = df[entity_key].map(aov).fillna(0.0)
    return df


def basket_size(
    df: pd.DataFrame,
    *,
    entity_key: str,
    order_id_col: str,
    item_col: str,
    output_col: str = "avg_basket_size",
) -> pd.DataFrame:
    """Average number of items per order per entity."""
    df = df.copy()
    items_per_order = df.groupby([entity_key, order_id_col])[item_col].count().reset_index()
    items_per_order.columns = [entity_key, order_id_col, "_items"]
    avg_items = items_per_order.groupby(entity_key)["_items"].mean()
    df[output_col] = df[entity_key].map(avg_items).fillna(0.0)
    return df


def category_diversity(
    df: pd.DataFrame,
    *,
    entity_key: str,
    category_col: str,
    output_col: str = "category_nunique",
) -> pd.DataFrame:
    """Number of unique categories per entity."""
    df = df.copy()
    nunique = df.groupby(entity_key)[category_col].nunique()
    df[output_col] = df[entity_key].map(nunique).fillna(0).astype(int)
    return df
