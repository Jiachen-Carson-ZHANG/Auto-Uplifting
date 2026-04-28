from pathlib import Path

import pandas as pd
import pytest

from src.models.uplift import (
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftTableSchema,
)
from src.uplift.features import (
    build_feature_table,
    compute_dataset_fingerprint,
    validate_feature_table,
)


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
    )


def _recipe() -> UpliftFeatureRecipeSpec:
    return UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=[30],
        builder_version="v1",
    )


def test_dataset_fingerprint_is_stable_for_same_sources():
    contract = _contract()

    fingerprint_a = compute_dataset_fingerprint(contract)
    fingerprint_b = compute_dataset_fingerprint(contract)

    assert fingerprint_a == fingerprint_b
    assert len(fingerprint_a) == 12


def test_build_feature_table_creates_one_labeled_customer_row_without_leakage_columns(tmp_path):
    artifact = build_feature_table(
        _contract(),
        recipe=_recipe(),
        output_dir=tmp_path,
        cohort="train",
        chunksize=2,
    )

    feature_df = pd.read_csv(artifact.artifact_path)

    assert artifact.row_count == 8
    assert feature_df.shape[0] == 8
    assert feature_df["client_id"].is_unique
    assert "target" not in feature_df.columns
    assert "treatment_flg" not in feature_df.columns
    assert {
        "age_clean",
        "redeem_missing_flag",
        "purchase_txn_count_lifetime",
        "purchase_sum_lifetime",
        "avg_transaction_value_lifetime",
        "points_received_to_purchase_ratio_lifetime",
        "purchase_txn_count_30d",
    }.issubset(feature_df.columns)
    assert set(artifact.generated_columns).issubset(set(feature_df.columns))


def test_build_feature_table_aggregates_purchase_rows_by_transaction(tmp_path):
    artifact = build_feature_table(
        _contract(),
        recipe=_recipe(),
        output_dir=tmp_path,
        cohort="train",
        chunksize=2,
    )
    feature_df = pd.read_csv(artifact.artifact_path).set_index("client_id")

    assert feature_df.loc["c001", "purchase_txn_count_lifetime"] == 1
    assert feature_df.loc["c001", "purchase_sum_lifetime"] == 100.0
    assert feature_df.loc["c002", "avg_transaction_value_lifetime"] == 200.0
    assert feature_df.loc["c004", "recency_days_lifetime"] == 0.0
    assert feature_df.loc["c005", "purchase_txn_count_lifetime"] == 0
    assert feature_df.loc["c005", "recency_days_lifetime"] == -1.0


def test_build_feature_table_records_computed_reference_date(tmp_path):
    artifact = build_feature_table(
        _contract(),
        recipe=_recipe(),
        output_dir=tmp_path,
        cohort="train",
        chunksize=2,
    )

    assert artifact.reference_date == "2019-01-04T12:00:00"


def test_build_feature_table_uses_pinned_reference_date_and_excludes_later_purchases(tmp_path):
    recipe = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=[1],
        builder_version="v1",
        reference_date="2019-01-03 12:00:00",
    )

    artifact = build_feature_table(
        _contract(),
        recipe=recipe,
        output_dir=tmp_path,
        cohort="train",
        chunksize=2,
    )
    feature_df = pd.read_csv(artifact.artifact_path).set_index("client_id")

    assert artifact.reference_date == "2019-01-03T12:00:00"
    assert feature_df.loc["c003", "purchase_txn_count_lifetime"] == 1
    assert feature_df.loc["c004", "purchase_txn_count_lifetime"] == 0
    assert feature_df.loc["c002", "purchase_txn_count_1d"] == 1
    assert feature_df.loc["c001", "purchase_txn_count_1d"] == 0


def test_build_feature_table_reuses_cached_artifact_without_rewriting(tmp_path):
    artifact_a = build_feature_table(
        _contract(),
        recipe=_recipe(),
        output_dir=tmp_path,
        cohort="train",
        chunksize=2,
    )
    artifact_path = Path(artifact_a.artifact_path)
    original_mtime = artifact_path.stat().st_mtime_ns

    artifact_b = build_feature_table(
        _contract(),
        recipe=_recipe(),
        output_dir=tmp_path,
        cohort="train",
        chunksize=2,
    )

    assert artifact_a.feature_artifact_id == artifact_b.feature_artifact_id
    assert Path(artifact_b.artifact_path).stat().st_mtime_ns == original_mtime


def test_validate_feature_table_rejects_duplicate_or_leaky_columns():
    bad_df = pd.DataFrame(
        {
            "client_id": ["c001", "c001"],
            "target": [0, 1],
            "feature": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate client_id"):
        validate_feature_table(
            bad_df,
            entity_key="client_id",
            forbidden_columns=["target", "treatment_flg"],
            expected_ids=["c001", "c002"],
        )

    bad_df = pd.DataFrame(
        {
            "client_id": ["c001", "c002"],
            "target": [0, 1],
            "feature": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError, match="forbidden feature columns"):
        validate_feature_table(
            bad_df,
            entity_key="client_id",
            forbidden_columns=["target", "treatment_flg"],
            expected_ids=["c001", "c002"],
        )
