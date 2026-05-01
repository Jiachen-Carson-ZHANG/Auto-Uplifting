import json
import subprocess
import sys
from pathlib import Path

from src.models.uplift import (
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
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


def test_agentic_tuning_execute_demo_runs_fixture_plan(tmp_path):
    artifact = build_feature_table(
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
    plan_path = tmp_path / "agentic_tuning_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "trial_specs": [
                    {
                        "spec_id": "AT-fixture-01",
                        "hypothesis_id": "agentic_tune__fixture__p01",
                        "template_name": "response_model_sklearn",
                        "learner_family": "response_model",
                        "base_estimator": "logistic_regression",
                        "feature_recipe_id": artifact.feature_recipe_id,
                        "params": {"C": 0.3, "max_iter": 1000},
                        "split_seed": 42,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "demos/uplift_execute_tuning_plan.py",
            "--data-dir",
            str(FIXTURE_DIR),
            "--plan",
            str(plan_path),
            "--feature-metadata",
            artifact.metadata_path,
            "--output-dir",
            str(tmp_path / "agentic_tuning_run"),
            "--small-fixture-mode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SUMMARY_JSON=")
    )
    summary = json.loads(summary_line.removeprefix("SUMMARY_JSON="))

    assert summary["n_records"] == 1
    assert summary["n_successful_records"] == 1
    assert summary["champion_run_id"]
    assert Path(summary["ledger_path"]).exists()
    assert Path(summary["summary_path"]).exists()
