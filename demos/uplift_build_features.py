#!/usr/bin/env python3
"""Build a cached RetailHero uplift feature table."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.uplift import (  # noqa: E402
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftTableSchema,
)
from src.uplift.features import build_feature_table  # noqa: E402


def _parse_windows(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(value.strip()) for value in raw.split(",") if value.strip()]


def _build_contract(data_dir: Path) -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        description="X5 RetailHero campaign uplift modeling.",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
            products_table=str(data_dir / "products.csv"),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="retailhero-uplift/data")
    parser.add_argument("--output-dir", default="artifacts/uplift/features")
    parser.add_argument("--cohort", choices=["train", "scoring", "all"], default="train")
    parser.add_argument("--windows", default="30,60,90")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    contract = _build_contract(Path(args.data_dir))
    recipe = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=_parse_windows(args.windows),
        builder_version="v1",
    )
    artifact = build_feature_table(
        contract,
        recipe=recipe,
        output_dir=args.output_dir,
        cohort=args.cohort,
        chunksize=args.chunksize,
        force=args.force,
    )

    summary = {
        "artifact_path": artifact.artifact_path,
        "metadata_path": artifact.metadata_path,
        "cohort": artifact.cohort,
        "row_count": artifact.row_count,
        "feature_recipe_id": artifact.feature_recipe_id,
        "feature_artifact_id": artifact.feature_artifact_id,
        "reference_date": artifact.reference_date,
        "n_generated_columns": len(artifact.generated_columns),
        "windows_days": artifact.windows_days,
    }

    print("Uplift feature table build")
    print(f"artifact_path: {artifact.artifact_path}")
    print(f"metadata_path: {artifact.metadata_path}")
    print(f"row_count: {artifact.row_count}")
    print(f"feature_artifact_id: {artifact.feature_artifact_id}")
    print(f"reference_date: {artifact.reference_date}")
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
