from pathlib import Path

import pandas as pd

from src.models.uplift import UpliftProjectContract, UpliftSplitContract, UpliftTableSchema
from src.uplift.validation import (
    compute_treatment_control_balance,
    determine_stratification,
    validate_uplift_dataset,
)


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _contract(data_dir: Path = FIXTURE_DIR) -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
            products_table=str(data_dir / "products.csv"),
        ),
        entity_key="client_id",
        treatment_column="treatment_flg",
        target_column="target",
    )


def test_validate_uplift_dataset_accepts_tiny_fixture():
    report = validate_uplift_dataset(_contract())

    assert report.valid is True
    assert report.errors == []
    assert report.table_rows["train"] == 8
    assert report.table_rows["scoring"] == 4
    assert report.scoring_is_unlabeled is True
    assert report.treatment_counts == {0: 4, 1: 4}
    assert report.target_counts == {0: 4, 1: 4}


def test_validate_uplift_dataset_rejects_target_in_scoring_table(tmp_path):
    for name in ["clients.csv", "purchases.csv", "products.csv", "uplift_train.csv"]:
        (tmp_path / name).write_text((FIXTURE_DIR / name).read_text(), encoding="utf-8")
    (tmp_path / "uplift_test.csv").write_text(
        "client_id,target\ns001,1\n",
        encoding="utf-8",
    )

    report = validate_uplift_dataset(_contract(tmp_path))

    assert report.valid is False
    assert any("scoring table must not contain target" in err for err in report.errors)


def test_validate_uplift_dataset_rejects_train_scoring_overlap(tmp_path):
    for name in ["clients.csv", "purchases.csv", "products.csv", "uplift_train.csv"]:
        (tmp_path / name).write_text((FIXTURE_DIR / name).read_text(), encoding="utf-8")
    (tmp_path / "uplift_test.csv").write_text("client_id\nc001\n", encoding="utf-8")

    report = validate_uplift_dataset(_contract(tmp_path))

    assert report.valid is False
    assert any("train/scoring overlap" in err for err in report.errors)


def test_validate_uplift_dataset_rejects_invalid_treatment_values(tmp_path):
    for name in ["clients.csv", "purchases.csv", "products.csv", "uplift_test.csv"]:
        (tmp_path / name).write_text((FIXTURE_DIR / name).read_text(), encoding="utf-8")
    (tmp_path / "uplift_train.csv").write_text(
        "client_id,treatment_flg,target\nc001,2,1\n",
        encoding="utf-8",
    )

    report = validate_uplift_dataset(_contract(tmp_path))

    assert report.valid is False
    assert any("treatment_flg must be binary" in err for err in report.errors)


def test_balance_diagnostics_reports_counts_rates_and_smd_warnings():
    train = pd.DataFrame({
        "client_id": ["c1", "c2", "c3", "c4"],
        "treatment_flg": [0, 0, 1, 1],
        "target": [0, 1, 0, 1],
    })
    features = pd.DataFrame({
        "client_id": ["c1", "c2", "c3", "c4"],
        "age": [20, 22, 70, 72],
    })

    diag = compute_treatment_control_balance(
        train,
        entity_key="client_id",
        treatment_col="treatment_flg",
        target_col="target",
        feature_df=features,
        numeric_columns=["age"],
        smd_warning_threshold=0.1,
    )

    assert diag.treatment_counts == {0: 2, 1: 2}
    assert diag.target_rates_by_treatment == {0: 0.5, 1: 0.5}
    assert diag.joint_counts["0:0"] == 1
    assert diag.standardized_mean_differences["age"] > 0.1
    assert any("age" in warning for warning in diag.warnings)


def test_determine_stratification_uses_joint_when_feasible():
    df = pd.DataFrame({
        "treatment_flg": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
        "target": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
    })

    decision = determine_stratification(
        df,
        treatment_col="treatment_flg",
        target_col="target",
        split_contract=UpliftSplitContract(min_rows_per_partition=1),
    )

    assert decision.strategy == "joint_treatment_outcome"
    assert decision.key.tolist() == ["0:0", "0:0", "0:0", "0:1", "0:1", "0:1", "1:0", "1:0", "1:0", "1:1", "1:1", "1:1"]
    assert decision.warnings == []


def test_determine_stratification_falls_back_to_treatment_only_when_joint_is_sparse():
    df = pd.DataFrame({
        "treatment_flg": [0, 0, 0, 1, 1, 1],
        "target": [0, 0, 1, 0, 0, 1],
    })

    decision = determine_stratification(
        df,
        treatment_col="treatment_flg",
        target_col="target",
        split_contract=UpliftSplitContract(min_rows_per_partition=1),
    )

    assert decision.strategy == "treatment_only"
    assert decision.key.tolist() == [0, 0, 0, 1, 1, 1]
    assert any("joint treatment/outcome stratification infeasible" in warning for warning in decision.warnings)


def test_determine_stratification_falls_back_to_random_with_hard_warning():
    df = pd.DataFrame({
        "treatment_flg": [0, 1],
        "target": [0, 1],
    })

    decision = determine_stratification(
        df,
        treatment_col="treatment_flg",
        target_col="target",
        split_contract=UpliftSplitContract(min_rows_per_partition=2),
    )

    assert decision.strategy == "random"
    assert decision.key is None
    assert any("falling back to random" in warning for warning in decision.warnings)
