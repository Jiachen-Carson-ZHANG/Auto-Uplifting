from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.models.uplift import (
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
)
from src.uplift.agentic_tuning_execution import (
    execute_agentic_tuning_plan,
    trial_specs_from_plan,
)
from src.uplift.features import build_feature_table


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _contract() -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table=str(FIXTURE_DIR / "clients.csv"),
            purchases_table=str(FIXTURE_DIR / "purchases.csv"),
            train_table=str(FIXTURE_DIR / "uplift_train.csv"),
            scoring_table=str(FIXTURE_DIR / "uplift_test.csv"),
            products_table=str(FIXTURE_DIR / "products.csv"),
        ),
        split_contract=UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            min_rows_per_partition=1,
            random_seed=7,
        ),
    )


def _feature_artifact(tmp_path):
    return build_feature_table(
        _contract(),
        recipe=UpliftFeatureRecipeSpec(
            source_tables=["clients", "purchases"],
            feature_groups=["demographic", "rfm", "basket", "points"],
            windows_days=[30],
            builder_version="v1",
        ),
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )


def _plan_payload(feature_recipe_id: str) -> dict:
    return {
        "tuning_seed": 20260501,
        "budget_rule": "min(16, 4 * tunable_parameter_count) per candidate",
        "trial_specs": [
            {
                "spec_id": "AT-01-01-response",
                "hypothesis_id": "agentic_tune__fixture__p01",
                "template_name": "response_model_sklearn",
                "learner_family": "response_model",
                "base_estimator": "logistic_regression",
                "feature_recipe_id": feature_recipe_id,
                "params": {"C": 0.3, "max_iter": 1000},
                "split_seed": 42,
            },
            {
                "spec_id": "AT-01-02-random",
                "hypothesis_id": "agentic_tune__fixture__p02",
                "template_name": "random_baseline",
                "learner_family": "random",
                "base_estimator": "logistic_regression",
                "feature_recipe_id": feature_recipe_id,
                "params": {},
                "split_seed": 42,
            },
        ],
    }


def test_trial_specs_from_plan_preserves_order_and_params():
    specs = trial_specs_from_plan(_plan_payload("recipe-123"))

    assert [spec.spec_id for spec in specs] == [
        "AT-01-01-response",
        "AT-01-02-random",
    ]
    assert specs[0].hypothesis_id == "agentic_tune__fixture__p01"
    assert specs[0].params == {"C": 0.3, "max_iter": 1000}
    assert specs[0].feature_recipe_id == "recipe-123"


def test_execute_agentic_tuning_plan_runs_specs_and_writes_combined_audit(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    plan_path = tmp_path / "agentic_tuning_plan.json"
    plan_path.write_text(
        json.dumps(_plan_payload(feature_artifact.feature_recipe_id)),
        encoding="utf-8",
    )

    result = execute_agentic_tuning_plan(
        contract,
        plan_path=plan_path,
        feature_artifacts_by_recipe_id={
            feature_artifact.feature_recipe_id: feature_artifact
        },
        output_dir=tmp_path / "agentic_tuning_run",
    )

    assert len(result.records) == 2
    assert Path(result.ledger_path).exists()
    assert Path(result.summary_path).exists()
    assert result.champion_run_id is not None
    assert result.group_outputs == {
        feature_artifact.feature_recipe_id: str(
            tmp_path
            / "agentic_tuning_run"
            / f"feature_recipe_{feature_artifact.feature_recipe_id}"
        )
    }

    combined_records = Path(result.ledger_path).read_text().strip().splitlines()
    assert len(combined_records) == 2
    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["n_trial_specs"] == 2
    assert summary["n_records"] == 2
    assert summary["plan_path"] == str(plan_path)
    assert summary["champion"]["run_id"] == result.champion_run_id
    summary_text = Path(result.summary_path).read_text()
    assert "held_out_qini_auc" not in summary_text
    assert "held_out_normalized_qini" not in summary_text

    for record in result.records:
        assert record.status == "success"
        assert Path(record.artifact_paths["predictions"]).exists()
        assert Path(record.artifact_paths["model"]).exists()
        assert "held_out_predictions" not in record.artifact_paths
        predictions = pd.read_csv(record.artifact_paths["predictions"])
        assert "uplift" in predictions.columns

    repeat = execute_agentic_tuning_plan(
        contract,
        plan_path=plan_path,
        feature_artifacts_by_recipe_id={
            feature_artifact.feature_recipe_id: feature_artifact
        },
        output_dir=tmp_path / "agentic_tuning_run",
    )
    group_ledger = (
        Path(repeat.group_outputs[feature_artifact.feature_recipe_id])
        / "uplift_ledger.jsonl"
    )
    assert len(group_ledger.read_text().strip().splitlines()) == 2


def test_execute_agentic_tuning_plan_rejects_missing_feature_artifact(tmp_path):
    plan_path = tmp_path / "agentic_tuning_plan.json"
    plan_path.write_text(json.dumps(_plan_payload("missing-recipe")), encoding="utf-8")

    with pytest.raises(ValueError, match="missing feature artifact"):
        execute_agentic_tuning_plan(
            _contract(),
            plan_path=plan_path,
            feature_artifacts_by_recipe_id={},
            output_dir=tmp_path / "agentic_tuning_run",
        )
