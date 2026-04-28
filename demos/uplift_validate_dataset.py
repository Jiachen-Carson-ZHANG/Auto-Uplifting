"""Validate a RetailHero-style uplift dataset without training models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.uplift import UpliftProjectContract, UpliftTableSchema
from src.uplift.validation import (
    compute_treatment_control_balance,
    validate_uplift_dataset,
)


def _build_contract(data_dir: Path) -> UpliftProjectContract:
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


def _balance_summary(contract: UpliftProjectContract) -> Dict[str, Any]:
    train = pd.read_csv(contract.table_schema.train_table)
    clients = pd.read_csv(contract.table_schema.clients_table)
    numeric_columns = [
        col
        for col in ["age"]
        if col in clients.columns and pd.api.types.is_numeric_dtype(clients[col])
    ]
    diagnostics = compute_treatment_control_balance(
        train,
        entity_key=contract.entity_key,
        treatment_col=contract.treatment_column,
        target_col=contract.target_column,
        feature_df=clients,
        numeric_columns=numeric_columns,
    )
    return diagnostics.model_dump()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("retailhero-uplift/data"),
        help="Directory containing clients/purchases/products/uplift_train/uplift_test CSVs.",
    )
    args = parser.parse_args()

    contract = _build_contract(args.data_dir)
    report = validate_uplift_dataset(contract)
    balance = _balance_summary(contract) if report.valid else {}

    summary: Dict[str, Any] = report.model_dump()
    summary["balance"] = balance
    summary["contract"] = {
        "entity_key": contract.entity_key,
        "treatment_column": contract.treatment_column,
        "target_column": contract.target_column,
        "scoring_table": contract.table_schema.scoring_table,
        "submission_policy": contract.submission_policy,
        "primary_metric": contract.evaluation_policy.primary_metric,
    }

    print("Uplift dataset validation")
    print(f"valid: {summary['valid']}")
    print(f"table_rows: {summary['table_rows']}")
    print(f"scoring_is_unlabeled: {summary['scoring_is_unlabeled']}")
    if report.errors:
        print(f"errors: {report.errors}")
    if report.warnings:
        print(f"warnings: {report.warnings}")
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
