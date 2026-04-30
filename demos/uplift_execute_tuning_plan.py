#!/usr/bin/env python3
"""Execute an agentic tuning dry-run plan and write tuning audit artifacts."""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.uplift import (  # noqa: E402
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
)
from src.uplift.agentic_tuning_execution import (  # noqa: E402
    execute_agentic_tuning_plan,
    feature_artifacts_from_metadata,
)

DEFAULT_RUN_ARTIFACT_DIR = ROOT / "artifacts" / "uplift" / "run_20260430_221602"
DEFAULT_PLAN = ROOT / "results" / "run_20260430_best" / "agentic_tuning_plan.json"


def _build_contract(data_dir: Path, *, small_fixture_mode: bool) -> UpliftProjectContract:
    split_contract = (
        UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.25,
            test_fraction=0.25,
            min_rows_per_partition=1,
            random_seed=7,
        )
        if small_fixture_mode
        else UpliftSplitContract()
    )
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        description="Agentic tuning execution.",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            products_table=str(data_dir / "products.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
        ),
        split_contract=split_contract,
        feature_sources=["clients", "purchases", "products"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--data-dir", default="retailhero-uplift/data")
    parser.add_argument(
        "--feature-metadata",
        action="append",
        default=[],
        help=(
            "Feature artifact metadata JSON path or glob. May be repeated. "
            "Defaults to the 2026-04-30 best-run train feature metadata."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_RUN_ARTIFACT_DIR / "agentic_tuning"),
    )
    parser.add_argument("--small-fixture-mode", action="store_true")
    return parser.parse_args()


def _resolve_paths(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else ROOT / path


def _metadata_paths(patterns: list[str]) -> list[Path]:
    if not patterns:
        patterns = [
            str(DEFAULT_RUN_ARTIFACT_DIR / "features" / "uplift_features_train_*.metadata.json")
        ]
    paths: list[Path] = []
    for pattern in patterns:
        resolved_pattern = str(_resolve_paths(pattern))
        matches = sorted(glob.glob(resolved_pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(resolved_pattern))
    return list(dict.fromkeys(paths))


def main() -> int:
    args = _parse_args()
    plan_path = _resolve_paths(args.plan)
    data_dir = _resolve_paths(args.data_dir)
    output_dir = _resolve_paths(args.output_dir)
    metadata_paths = _metadata_paths(args.feature_metadata)

    contract = _build_contract(data_dir, small_fixture_mode=args.small_fixture_mode)
    feature_artifacts = feature_artifacts_from_metadata(metadata_paths)
    result = execute_agentic_tuning_plan(
        contract,
        plan_path=plan_path,
        feature_artifacts_by_recipe_id=feature_artifacts,
        output_dir=output_dir,
    )
    summary = {
        "plan_path": str(plan_path),
        "output_dir": result.output_dir,
        "ledger_path": result.ledger_path,
        "summary_path": result.summary_path,
        "n_records": len(result.records),
        "n_successful_records": sum(record.status == "success" for record in result.records),
        "champion_run_id": result.champion_run_id,
        "champion_hypothesis_id": result.champion_hypothesis_id,
        "champion_template_name": result.champion_template_name,
        "champion_qini_auc": result.champion_qini_auc,
        "champion_held_out_qini_auc": result.champion_held_out_qini_auc,
        "group_outputs": result.group_outputs,
    }
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
