#!/usr/bin/env python3
"""Run the deterministic Uplift V1 baseline ladder."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.uplift import (  # noqa: E402
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
)
from src.uplift.features import build_feature_table  # noqa: E402
from src.uplift.loop import run_uplift_trials  # noqa: E402
from src.uplift.reporting import (  # noqa: E402
    generate_submission_artifact,
    generate_uplift_report,
    validate_submission_artifact,
)
from src.uplift.templates import fit_uplift_model  # noqa: E402


def _build_contract(data_dir: Path, *, small_fixture_mode: bool) -> UpliftProjectContract:
    split_contract = (
        UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.5,
            test_fraction=0.0,
            min_rows_per_partition=1,
            random_seed=7,
        )
        if small_fixture_mode
        else UpliftSplitContract()
    )
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
        split_contract=split_contract,
    )


def _baseline_trials(feature_recipe_id: str, split_seed: int) -> list[UpliftTrialSpec]:
    return [
        UpliftTrialSpec(
            hypothesis_id="baseline-random",
            template_name="random_baseline",
            learner_family="random",
            feature_recipe_id=feature_recipe_id,
            split_seed=split_seed,
        ),
        UpliftTrialSpec(
            hypothesis_id="baseline-response",
            template_name="response_model_sklearn",
            learner_family="response_model",
            feature_recipe_id=feature_recipe_id,
            split_seed=split_seed,
        ),
        UpliftTrialSpec(
            hypothesis_id="baseline-two-model",
            template_name="two_model_sklearn",
            learner_family="two_model",
            feature_recipe_id=feature_recipe_id,
            split_seed=split_seed,
        ),
        UpliftTrialSpec(
            hypothesis_id="baseline-solo-model",
            template_name="solo_model_sklearn",
            learner_family="solo_model",
            feature_recipe_id=feature_recipe_id,
            split_seed=split_seed,
        ),
    ]


def _champion_trial(records, trials: list[UpliftTrialSpec]) -> UpliftTrialSpec:
    successful = [record for record in records if record.status == "success"]
    if not successful:
        return trials[0]
    champion_record = max(
        successful,
        key=lambda record: record.qini_auc if record.qini_auc is not None else float("-inf"),
    )
    return next(
        trial for trial in trials if trial.template_name == champion_record.template_name
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="retailhero-uplift/data")
    parser.add_argument("--output-dir", default="artifacts/uplift/baseline_runs")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--small-fixture-mode", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    run_dir = output_dir / "runs"
    contract = _build_contract(Path(args.data_dir), small_fixture_mode=args.small_fixture_mode)
    recipe = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=[30, 60, 90],
        builder_version="v1",
    )

    train_artifact = build_feature_table(
        contract,
        recipe=recipe,
        output_dir=feature_dir,
        cohort="train",
        chunksize=args.chunksize,
    )
    scoring_artifact = build_feature_table(
        contract,
        recipe=recipe,
        output_dir=feature_dir,
        cohort="scoring",
        chunksize=args.chunksize,
    )
    trials = _baseline_trials(
        train_artifact.feature_recipe_id,
        contract.split_contract.random_seed,
    )
    loop_result = run_uplift_trials(
        contract,
        feature_artifact=train_artifact,
        trial_specs=trials,
        output_dir=run_dir,
    )
    report_path = generate_uplift_report(
        contract,
        records=loop_result.records,
        output_path=output_dir / "uplift_report.md",
    )

    champion = _champion_trial(loop_result.records, trials)
    train_features = pd.read_csv(train_artifact.artifact_path)
    labels = pd.read_csv(contract.table_schema.train_table)
    train_frame = train_features.merge(labels, on=contract.entity_key, how="inner")
    model = fit_uplift_model(
        train_frame,
        learner_family=champion.learner_family,
        entity_key=contract.entity_key,
        treatment_col=contract.treatment_column,
        target_col=contract.target_column,
        random_seed=champion.split_seed,
    )
    submission = generate_submission_artifact(
        contract,
        model=model,
        scoring_feature_artifact=scoring_artifact,
        champion_trial=champion,
        output_path=output_dir / "uplift_submission.csv",
    )
    validate_submission_artifact(contract, submission)

    summary = {
        "n_records": len(loop_result.records),
        "ledger_path": loop_result.ledger_path,
        "report_path": report_path,
        "submission_path": submission.artifact_path,
        "champion_template": champion.template_name,
        "train_feature_artifact_id": train_artifact.feature_artifact_id,
        "scoring_feature_artifact_id": scoring_artifact.feature_artifact_id,
    }
    print("Uplift baseline ladder complete")
    print(f"ledger_path: {loop_result.ledger_path}")
    print(f"report_path: {report_path}")
    print(f"submission_path: {submission.artifact_path}")
    print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
