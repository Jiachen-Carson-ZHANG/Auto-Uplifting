from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.models.uplift import UpliftTrialSpec
from src.uplift.tuning import (
    build_pre_run_tuning_specs,
    select_stable_tuning_record,
)


def _write_scores(path: Path, uplift: list[float]) -> str:
    pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(8)],
            "uplift": uplift,
            "treatment_flg": [1, 1, 0, 0, 1, 0, 1, 0],
            "target": [1, 1, 1, 0, 0, 0, 1, 0],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_build_pre_run_tuning_specs_expands_model_into_seeded_candidates():
    base = UpliftTrialSpec(
        spec_id="UT-plan",
        hypothesis_id="UT-plan",
        template_name="two_model_gradient_boosting_sklearn",
        learner_family="two_model",
        base_estimator="gradient_boosting",
        feature_recipe_id="recipe123",
        params={"n_estimators": 50},
        split_seed=42,
    )

    specs = build_pre_run_tuning_specs(
        base,
        split_seeds=(42, 123),
        max_param_sets=1,
    )

    assert [spec.split_seed for spec in specs] == [42, 123]
    assert all(spec.spec_id.startswith("UT-plan__tune_") for spec in specs)
    assert specs[0].params == {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "subsample": 0.7,
    }


def test_select_stable_tuning_record_penalizes_validation_only_spikes(tmp_path):
    good = [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3]
    unstable = SimpleNamespace(
        status="success",
        artifact_paths={
            "uplift_scores": _write_scores(tmp_path / "unstable_val.csv", good),
            "held_out_predictions": _write_scores(
                tmp_path / "unstable_held.csv",
                [-value for value in good],
            ),
        },
    )
    stable = SimpleNamespace(
        status="success",
        artifact_paths={
            "uplift_scores": _write_scores(tmp_path / "stable_val.csv", good),
            "held_out_predictions": _write_scores(tmp_path / "stable_held.csv", good),
        },
    )

    selected = select_stable_tuning_record([unstable, stable])

    assert selected is stable
