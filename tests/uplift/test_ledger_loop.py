from pathlib import Path

import pandas as pd

from src.models.uplift import (
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
)
from src.uplift.features import build_feature_table
from src.uplift.ledger import UpliftLedger, params_hash
from src.uplift.loop import run_uplift_trials


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
            val_fraction=0.5,
            test_fraction=0.0,
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


def test_uplift_ledger_appends_and_reloads_records(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    record = ledger.append_result(
        trial_spec=UpliftTrialSpec(
            hypothesis_id="baseline",
            template_name="random_baseline",
            learner_family="random",
            feature_recipe_id="recipe123456",
            params={"seed": 1},
        ),
        feature_artifact_id="artifact1234",
        result_status="success",
        qini_auc=0.1,
        uplift_auc=0.2,
        uplift_at_k={"top_50pct": 0.3},
        policy_gain={"top_50pct_zero_cost": 1.0},
        artifact_paths={"predictions": "predictions.csv"},
    )

    loaded = ledger.load()

    assert len(loaded) == 1
    assert loaded[0].run_id == record.run_id
    assert loaded[0].status == "success"
    assert loaded[0].params_hash == params_hash({"seed": 1})


def test_run_uplift_trials_writes_ledger_and_metric_artifacts(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    trials = [
        UpliftTrialSpec(
            hypothesis_id="baseline-random",
            template_name="random_baseline",
            learner_family="random",
            feature_recipe_id=feature_artifact.feature_recipe_id,
        ),
        UpliftTrialSpec(
            hypothesis_id="baseline-response",
            template_name="response_model_sklearn",
            learner_family="response_model",
            feature_recipe_id=feature_artifact.feature_recipe_id,
        ),
    ]

    result = run_uplift_trials(
        contract,
        feature_artifact=feature_artifact,
        trial_specs=trials,
        output_dir=tmp_path / "runs",
    )

    assert len(result.records) == 2
    assert all(record.status == "success" for record in result.records)
    assert Path(result.ledger_path).exists()
    for record in result.records:
        assert Path(record.artifact_paths["predictions"]).exists()
        assert Path(record.artifact_paths["decile_table"]).exists()
        predictions = pd.read_csv(record.artifact_paths["predictions"])
        assert not set(predictions["client_id"]).intersection({"s001", "s002", "s003", "s004"})


def _contract_with_test_split() -> UpliftProjectContract:
    contract = _contract()
    return contract.model_copy(
        update={
            "split_contract": UpliftSplitContract(
                train_fraction=0.5,
                val_fraction=0.25,
                test_fraction=0.25,
                min_rows_per_partition=1,
                random_seed=7,
            ),
        }
    )


def test_run_uplift_trials_emits_held_out_metrics_when_test_partition_present(tmp_path):
    contract = _contract_with_test_split()
    feature_artifact = _feature_artifact(tmp_path)
    trials = [
        UpliftTrialSpec(
            hypothesis_id="baseline-response",
            template_name="response_model_sklearn",
            learner_family="response_model",
            feature_recipe_id=feature_artifact.feature_recipe_id,
        ),
    ]

    result = run_uplift_trials(
        contract,
        feature_artifact=feature_artifact,
        trial_specs=trials,
        output_dir=tmp_path / "runs",
    )

    record = result.records[0]
    assert record.status == "success"
    assert record.qini_auc is not None
    # Held-out metrics come from the test partition; the values themselves can be
    # anything on a tiny fixture, but the held-out artifact paths must exist.
    assert "held_out_predictions" in record.artifact_paths
    assert "held_out_decile_table" in record.artifact_paths
    assert Path(record.artifact_paths["held_out_predictions"]).exists()


def test_run_uplift_trials_omits_held_out_metrics_when_no_test_partition(tmp_path):
    contract = _contract()  # default fixture: train=0.5, val=0.5, test=0.0
    feature_artifact = _feature_artifact(tmp_path)
    trials = [
        UpliftTrialSpec(
            hypothesis_id="baseline-response",
            template_name="response_model_sklearn",
            learner_family="response_model",
            feature_recipe_id=feature_artifact.feature_recipe_id,
        ),
    ]

    result = run_uplift_trials(
        contract,
        feature_artifact=feature_artifact,
        trial_specs=trials,
        output_dir=tmp_path / "runs",
    )

    record = result.records[0]
    assert record.status == "success"
    assert record.held_out_qini_auc is None
    assert record.held_out_uplift_at_k == {}
    assert "held_out_predictions" not in record.artifact_paths


def test_run_uplift_trials_records_failures_for_invalid_specs(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    trial = UpliftTrialSpec(
        hypothesis_id="bad",
        template_name="missing_template",
        learner_family="random",
        feature_recipe_id=feature_artifact.feature_recipe_id,
    )

    result = run_uplift_trials(
        contract,
        feature_artifact=feature_artifact,
        trial_specs=[trial],
        output_dir=tmp_path / "runs",
    )

    assert len(result.records) == 1
    assert result.records[0].status == "failed"
    assert "unknown uplift template" in result.records[0].error
