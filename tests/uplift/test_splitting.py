from pathlib import Path

import pandas as pd

from src.models.uplift import UpliftProjectContract, UpliftSplitContract, UpliftTableSchema
from src.uplift.splitting import split_labeled_uplift_frame


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _contract() -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table=str(FIXTURE_DIR / "clients.csv"),
            purchases_table=str(FIXTURE_DIR / "purchases.csv"),
            train_table=str(FIXTURE_DIR / "uplift_train.csv"),
            scoring_table=str(FIXTURE_DIR / "uplift_test.csv"),
        ),
        split_contract=UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.5,
            test_fraction=0.0,
            min_rows_per_partition=1,
            random_seed=7,
        ),
    )


def test_split_labeled_uplift_frame_preserves_all_customer_ids_once():
    contract = _contract()
    labeled = pd.read_csv(FIXTURE_DIR / "uplift_train.csv")

    split = split_labeled_uplift_frame(labeled, contract)
    all_ids = pd.concat(
        [
            split.train["client_id"],
            split.validation["client_id"],
            split.test["client_id"],
        ],
        ignore_index=True,
    )

    assert split.strategy == "joint_treatment_outcome"
    assert len(all_ids) == len(labeled)
    assert all_ids.is_unique
    assert set(all_ids) == set(labeled["client_id"])


def test_split_labeled_uplift_frame_never_reads_scoring_table():
    contract = _contract()
    labeled = pd.read_csv(FIXTURE_DIR / "uplift_train.csv")

    split = split_labeled_uplift_frame(labeled, contract)

    assert not set(split.train["client_id"]).intersection({"s001", "s002", "s003", "s004"})
    assert not set(split.validation["client_id"]).intersection({"s001", "s002", "s003", "s004"})
    assert not set(split.test["client_id"]).intersection({"s001", "s002", "s003", "s004"})
