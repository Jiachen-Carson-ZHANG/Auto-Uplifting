import ast
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.uplift import UpliftEvaluationPolicy
from src.uplift.metrics import (
    decile_table,
    evaluate_uplift_predictions,
    policy_gain_by_cutoff,
    qini_auc_score,
    uplift_at_k,
    uplift_auc_score,
)


def _toy_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    treatment = np.array([1, 1, 0, 0, 1, 0, 1, 0])
    uplift_good = np.array([0.9, 0.8, 0.2, 0.1, -0.1, -0.2, 0.7, -0.3])
    return y_true, treatment, uplift_good


def test_auc_integration_uses_numpy_126_compatible_trapz():
    source = Path("src/uplift/metrics.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    numpy_integration_attrs = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"trapz", "trapezoid"}
    ]

    assert "trapz" in numpy_integration_attrs
    assert "trapezoid" not in numpy_integration_attrs


def test_auc_integration_suppresses_numpy_trapz_deprecation_warning():
    y_true, treatment, uplift_good = _toy_arrays()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        qini_auc_score(y_true, treatment, uplift_good)
        uplift_auc_score(y_true, treatment, uplift_good)

    assert not any(
        "trapz" in str(warning.message) and "deprecated" in str(warning.message)
        for warning in caught
    )


def test_uplift_metrics_reward_better_ranking_than_reversed_ranking():
    y_true, treatment, uplift_good = _toy_arrays()
    uplift_bad = -uplift_good

    assert qini_auc_score(y_true, treatment, uplift_good) > qini_auc_score(
        y_true, treatment, uplift_bad
    )
    assert uplift_auc_score(y_true, treatment, uplift_good) > uplift_auc_score(
        y_true, treatment, uplift_bad
    )


def test_uplift_at_k_uses_top_ranked_treatment_control_difference():
    y_true, treatment, uplift_good = _toy_arrays()

    assert uplift_at_k(y_true, treatment, uplift_good, k=0.75) > 0


def test_uplift_metrics_reject_shape_and_treatment_errors():
    y_true, treatment, uplift_good = _toy_arrays()

    with pytest.raises(ValueError, match="same length"):
        qini_auc_score(y_true[:-1], treatment, uplift_good)

    with pytest.raises(ValueError, match="binary"):
        qini_auc_score(y_true, np.array([0, 1, 2, 0, 1, 0, 1, 0]), uplift_good)


def test_decile_table_and_policy_gain_are_grounded_in_cutoffs():
    y_true, treatment, uplift_good = _toy_arrays()
    policy = UpliftEvaluationPolicy(
        cutoff_grid=[0.5, 0.75],
        conversion_value=10.0,
        cost_scenarios={"zero": 0.0, "paid": 1.0},
    )

    table = decile_table(y_true, treatment, uplift_good, n_bins=4)
    gains = policy_gain_by_cutoff(y_true, treatment, uplift_good, policy)

    assert isinstance(table, pd.DataFrame)
    assert set(["bin", "n", "uplift", "avg_predicted_uplift"]).issubset(table.columns)
    assert set(gains) == {"top_50pct_zero", "top_50pct_paid", "top_75pct_zero", "top_75pct_paid"}
    assert gains["top_50pct_zero"] >= gains["top_50pct_paid"]


def test_uplift_at_k_returns_nan_when_top_slice_has_no_control_rows():
    y_true, treatment, uplift_good = _toy_arrays()

    # Top 25% of 8 sorted rows is the two highest predicted-uplift rows,
    # and in this fixture both happen to be treatment=1, leaving zero
    # control rows in the slice. The estimator is then undefined.
    value = uplift_at_k(y_true, treatment, uplift_good, k=0.25)
    assert np.isnan(value)


def test_policy_gain_propagates_nan_when_uplift_is_undefined():
    y_true, treatment, uplift_good = _toy_arrays()
    policy = UpliftEvaluationPolicy(
        cutoff_grid=[0.25],
        conversion_value=10.0,
        cost_scenarios={"zero": 0.0, "paid": 1.0},
    )

    gains = policy_gain_by_cutoff(y_true, treatment, uplift_good, policy)

    assert np.isnan(gains["top_25pct_zero"])
    assert np.isnan(gains["top_25pct_paid"])


def test_evaluate_uplift_predictions_returns_metric_and_artifact_frames():
    y_true, treatment, uplift_good = _toy_arrays()
    policy = UpliftEvaluationPolicy(cutoff_grid=[0.5])

    result = evaluate_uplift_predictions(y_true, treatment, uplift_good, policy)

    assert result.qini_auc is not None
    assert result.uplift_auc is not None
    assert "top_50pct" in result.uplift_at_k
    assert not result.decile_table.empty
    assert not result.qini_curve.empty
