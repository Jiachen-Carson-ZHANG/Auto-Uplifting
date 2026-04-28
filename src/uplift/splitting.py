"""Customer-level splitting helpers for labeled uplift rows."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.models.uplift import UpliftProjectContract
from src.uplift.validation import determine_stratification


@dataclass(frozen=True)
class UpliftSplitFrames:
    """Concrete train/validation/test frames plus split diagnostics."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    strategy: str
    warnings: List[str] = field(default_factory=list)


def _safe_stratify(values: pd.Series | None, n_rows: int) -> pd.Series | None:
    if values is None or n_rows == 0:
        return None
    counts = values.value_counts()
    return values if not counts.empty and int(counts.min()) >= 2 else None


def split_labeled_uplift_frame(
    labeled_df: pd.DataFrame,
    contract: UpliftProjectContract,
) -> UpliftSplitFrames:
    """Split only labeled uplift_train rows; never reads scoring rows."""
    split_contract = contract.split_contract
    decision = determine_stratification(
        labeled_df,
        treatment_col=contract.treatment_column,
        target_col=contract.target_column,
        split_contract=split_contract,
    )

    indices = np.arange(len(labeled_df))
    val_test_fraction = split_contract.val_fraction + split_contract.test_fraction
    stratify = _safe_stratify(decision.key, len(labeled_df))

    if val_test_fraction <= 0:
        return UpliftSplitFrames(
            train=labeled_df.reset_index(drop=True),
            validation=labeled_df.iloc[[]].copy(),
            test=labeled_df.iloc[[]].copy(),
            strategy=decision.strategy,
            warnings=decision.warnings,
        )

    train_idx, rest_idx = train_test_split(
        indices,
        test_size=val_test_fraction,
        random_state=split_contract.random_seed,
        stratify=stratify,
    )

    if split_contract.test_fraction <= 0:
        val_idx = rest_idx
        test_idx = np.array([], dtype=int)
    elif split_contract.val_fraction <= 0:
        val_idx = np.array([], dtype=int)
        test_idx = rest_idx
    else:
        relative_test = split_contract.test_fraction / val_test_fraction
        rest_stratify = (
            _safe_stratify(decision.key.iloc[rest_idx], len(rest_idx))
            if decision.key is not None
            else None
        )
        val_idx, test_idx = train_test_split(
            rest_idx,
            test_size=relative_test,
            random_state=split_contract.random_seed,
            stratify=rest_stratify,
        )

    return UpliftSplitFrames(
        train=labeled_df.iloc[train_idx].reset_index(drop=True),
        validation=labeled_df.iloc[val_idx].reset_index(drop=True),
        test=labeled_df.iloc[test_idx].reset_index(drop=True),
        strategy=decision.strategy,
        warnings=decision.warnings,
    )
