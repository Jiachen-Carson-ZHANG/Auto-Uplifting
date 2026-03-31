"""
Feature validation — checks that feature execution output is safe.

Returns list of warning/error strings (empty = valid).
"""
from __future__ import annotations
from typing import List
import pandas as pd


class FeatureValidator:
    """Validates feature execution output against safety invariants."""

    def validate_result(
        self,
        original_df: pd.DataFrame,
        result_df: pd.DataFrame,
        target_col: str,
    ) -> List[str]:
        """
        Check that the result DataFrame is safe for downstream use.

        Returns list of error/warning strings. Empty list = valid.
        """
        issues: List[str] = []

        # 1. Row count preserved (within 1% tolerance)
        orig_rows = len(original_df)
        result_rows = len(result_df)
        if orig_rows > 0 and abs(result_rows - orig_rows) / orig_rows > 0.01:
            issues.append(
                f"Row count changed: {orig_rows} → {result_rows} "
                f"({abs(result_rows - orig_rows) / orig_rows:.1%} change)"
            )

        # 2. Target column preserved
        if target_col not in result_df.columns:
            issues.append(f"Target column '{target_col}' missing from result.")
        elif not original_df[target_col].equals(result_df[target_col]):
            issues.append(f"Target column '{target_col}' values were modified.")

        # 3. No all-null new columns
        new_cols = set(result_df.columns) - set(original_df.columns)
        for col in sorted(new_cols):
            if result_df[col].isnull().all():
                issues.append(f"New column '{col}' is all null.")

        # 4. No constant-value new columns
        for col in sorted(new_cols):
            if col in result_df.columns and result_df[col].nunique(dropna=True) <= 1:
                if not result_df[col].isnull().all():  # already caught above
                    issues.append(f"New column '{col}' has constant value.")

        # 5. No NaN explosion in existing columns
        shared_cols = set(original_df.columns) & set(result_df.columns)
        for col in sorted(shared_cols):
            if col == target_col:
                continue
            orig_nulls = original_df[col].isnull().sum()
            result_nulls = result_df[col].isnull().sum()
            new_nulls = result_nulls - orig_nulls
            if new_nulls > 0 and new_nulls > 0.1 * len(result_df):
                issues.append(
                    f"Column '{col}' gained {new_nulls} NaNs "
                    f"({new_nulls / len(result_df):.1%} of rows)."
                )

        return issues
