from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.models.uplift import UpliftTrialSpec
from src.models.uplift import UpliftExperimentRecord
from src.uplift.tuning import (
    build_agentic_tuning_plan,
    build_pre_run_tuning_specs,
    select_top_tuning_candidates,
    select_stable_tuning_record,
    validate_tuning_search_space,
    write_agentic_tuning_plan,
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


def test_select_top_tuning_candidates_uses_internal_stability_not_external_baseline():
    records = [
        _record(
            "RUN-manual",
            "manual_baseline",
            "two_model",
            "logistic_regression",
            qini_auc=500.0,
            held_out_qini_auc=500.0,
            verdict="baseline",
        ),
        _record(
            "RUN-overfit",
            "UT-overfit",
            "two_model",
            "lightgbm",
            qini_auc=420.0,
            held_out_qini_auc=250.0,
        ),
        _record(
            "RUN-xgb",
            "UT-xgb",
            "class_transformation",
            "xgboost",
            qini_auc=341.0,
            held_out_qini_auc=326.0,
        ),
        _record(
            "RUN-lgbm",
            "UT-lgbm",
            "class_transformation",
            "lightgbm",
            qini_auc=333.0,
            held_out_qini_auc=331.0,
        ),
        _record(
            "RUN-failed",
            "UT-failed",
            "class_transformation",
            "random_forest",
            status="failed",
            qini_auc=999.0,
            held_out_qini_auc=999.0,
        ),
    ]

    selected = select_top_tuning_candidates(records, top_k=2)

    assert [(c.learner_family, c.base_estimator) for c in selected] == [
        ("class_transformation", "lightgbm"),
        ("class_transformation", "xgboost"),
    ]
    assert all("manual" not in c.source_hypothesis_id for c in selected)


def test_validate_tuning_search_space_rejects_unknown_and_out_of_range_params():
    search_space, warnings = validate_tuning_search_space(
        "lightgbm",
        {
            "n_estimators": [50, 400, 900],
            "learning_rate": [0.03, 0.5],
            "subsample": [0.7, 1.2],
            "drop_table": [1],
        },
    )

    assert search_space == {
        "learning_rate": [0.03],
        "n_estimators": [400],
        "subsample": [0.7],
    }
    assert any("drop_table" in warning for warning in warnings)
    assert any("n_estimators" in warning and "50" in warning for warning in warnings)
    assert any("learning_rate" in warning and "0.5" in warning for warning in warnings)


def test_agentic_tuning_plan_calls_llm_and_samples_deterministically_without_human_baseline():
    captured = {}

    def fake_llm(system: str, user: str) -> str:
        captured["system"] = system
        captured["user"] = user
        return """
        {
          "rationale": "Tune only the strongest internal AutoLift candidates.",
          "search_spaces": [
            {
              "template_name": "class_transformation_lightgbm",
              "rationale": "Refine leaf shape and regularization.",
              "search_space": {
                "n_estimators": [300, 400],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "num_leaves": [7, 15]
              }
            },
            {
              "template_name": "class_transformation_xgboost",
              "rationale": "Refine depth and child-weight regularization.",
              "search_space": {
                "n_estimators": [300, 400],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "min_child_weight": [10, 20]
              }
            }
          ]
        }
        """

    records = [
        _record(
            "RUN-xgb",
            "UT-xgb",
            "class_transformation",
            "xgboost",
            qini_auc=341.0,
            held_out_qini_auc=326.0,
        ),
        _record(
            "RUN-lgbm",
            "UT-lgbm",
            "class_transformation",
            "lightgbm",
            qini_auc=333.0,
            held_out_qini_auc=331.0,
        ),
    ]

    plan = build_agentic_tuning_plan(
        records,
        llm=fake_llm,
        tuning_seed=20260501,
        top_k=2,
        max_trials_per_candidate=16,
    )
    repeat = build_agentic_tuning_plan(
        records,
        llm=fake_llm,
        tuning_seed=20260501,
        top_k=2,
        max_trials_per_candidate=16,
    )

    prompt = captured["system"] + captured["user"]
    assert "human_baseline" not in prompt
    assert "328.3899" not in prompt
    assert "Human Notebook" not in prompt
    assert len(plan.candidates) == 2
    assert len(plan.trial_specs) == 32
    assert [spec.params for spec in plan.trial_specs] == [
        spec.params for spec in repeat.trial_specs
    ]
    assert all(spec.hypothesis_id.startswith("agentic_tune__") for spec in plan.trial_specs)
    assert {
        spec.template_name for spec in plan.trial_specs
    } == {"class_transformation_lightgbm", "class_transformation_xgboost"}


def test_write_agentic_tuning_plan_saves_audit_json(tmp_path):
    records = [
        _record(
            "RUN-lgbm",
            "UT-lgbm",
            "class_transformation",
            "lightgbm",
            qini_auc=333.0,
            held_out_qini_auc=331.0,
        )
    ]

    plan = build_agentic_tuning_plan(
        records,
        llm=lambda _system, _user: "{}",
        tuning_seed=20260501,
        top_k=1,
        max_trials_per_candidate=4,
    )
    output = write_agentic_tuning_plan(tmp_path / "tuning_plan.json", plan)

    text = Path(output).read_text()
    assert '"tuning_seed": 20260501' in text
    assert '"trial_specs"' in text
    assert "human_baseline" not in text


def _record(
    run_id: str,
    hypothesis_id: str,
    learner_family: str,
    base_estimator: str,
    *,
    status: str = "success",
    qini_auc: float = 0.0,
    held_out_qini_auc: float = 0.0,
    verdict: str = "supported",
) -> UpliftExperimentRecord:
    return UpliftExperimentRecord(
        run_id=run_id,
        hypothesis_id=hypothesis_id,
        feature_recipe_id="recipe-hybrid",
        template_name=f"{learner_family}_{base_estimator}",
        uplift_learner_family=learner_family,
        base_estimator=base_estimator,
        params_hash=f"{run_id}-params",
        split_seed=42,
        status=status,  # type: ignore[arg-type]
        qini_auc=qini_auc,
        uplift_auc=0.06,
        held_out_qini_auc=held_out_qini_auc,
        held_out_uplift_auc=0.061,
        verdict=verdict,  # type: ignore[arg-type]
    )
