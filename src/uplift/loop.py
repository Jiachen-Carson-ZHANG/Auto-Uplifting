"""Deterministic uplift trial execution loop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftTrialSpec,
)
from src.uplift.ledger import UpliftLedger
from src.uplift.splitting import split_labeled_uplift_frame
from src.uplift.templates import run_uplift_template


@dataclass(frozen=True)
class UpliftLoopResult:
    """Summary of one deterministic uplift trial batch."""

    records: List[UpliftExperimentRecord]
    ledger_path: str
    output_dir: str


def _labeled_feature_frame(
    contract: UpliftProjectContract,
    feature_artifact: UpliftFeatureArtifact,
) -> pd.DataFrame:
    features = pd.read_csv(feature_artifact.artifact_path)
    labels = pd.read_csv(
        contract.table_schema.train_table,
        usecols=[
            contract.entity_key,
            contract.treatment_column,
            contract.target_column,
        ],
    )
    return features.merge(labels, on=contract.entity_key, how="inner")


def _write_trial_artifacts(
    trial_dir: Path,
    *,
    predictions: pd.DataFrame,
    decile_table: pd.DataFrame,
    qini_curve: pd.DataFrame,
    uplift_curve: pd.DataFrame,
    result_json: str,
    held_out_predictions: pd.DataFrame | None = None,
    held_out_decile_table: pd.DataFrame | None = None,
    held_out_qini_curve: pd.DataFrame | None = None,
    held_out_uplift_curve: pd.DataFrame | None = None,
) -> dict[str, str]:
    trial_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "predictions": str(trial_dir / "predictions.csv"),
        "decile_table": str(trial_dir / "decile_table.csv"),
        "qini_curve": str(trial_dir / "qini_curve.csv"),
        "uplift_curve": str(trial_dir / "uplift_curve.csv"),
        "result_card": str(trial_dir / "result_card.json"),
    }
    predictions.to_csv(paths["predictions"], index=False)
    decile_table.to_csv(paths["decile_table"], index=False)
    qini_curve.to_csv(paths["qini_curve"], index=False)
    uplift_curve.to_csv(paths["uplift_curve"], index=False)
    Path(paths["result_card"]).write_text(result_json, encoding="utf-8")

    if held_out_predictions is not None:
        paths["held_out_predictions"] = str(trial_dir / "held_out_predictions.csv")
        held_out_predictions.to_csv(paths["held_out_predictions"], index=False)
    if held_out_decile_table is not None:
        paths["held_out_decile_table"] = str(trial_dir / "held_out_decile_table.csv")
        held_out_decile_table.to_csv(paths["held_out_decile_table"], index=False)
    if held_out_qini_curve is not None:
        paths["held_out_qini_curve"] = str(trial_dir / "held_out_qini_curve.csv")
        held_out_qini_curve.to_csv(paths["held_out_qini_curve"], index=False)
    if held_out_uplift_curve is not None:
        paths["held_out_uplift_curve"] = str(trial_dir / "held_out_uplift_curve.csv")
        held_out_uplift_curve.to_csv(paths["held_out_uplift_curve"], index=False)
    return paths


def run_uplift_trials(
    contract: UpliftProjectContract,
    *,
    feature_artifact: UpliftFeatureArtifact,
    trial_specs: List[UpliftTrialSpec],
    output_dir: str | Path,
) -> UpliftLoopResult:
    """Execute trial specs, write artifacts, and append ledger records."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ledger = UpliftLedger(output / "uplift_ledger.jsonl")
    labeled = _labeled_feature_frame(contract, feature_artifact)
    split = split_labeled_uplift_frame(labeled, contract)

    # Validation drives champion selection. The test partition (when present) is held out
    # for an honest generalization estimate scored from the same fitted model. When no
    # test partition is configured, validation is the only evaluation surface.
    if not split.validation.empty:
        eval_df = split.validation
        held_out_df: pd.DataFrame | None = split.test if not split.test.empty else None
    else:
        eval_df = split.test
        held_out_df = None

    records: List[UpliftExperimentRecord] = []

    for spec in trial_specs:
        trial_dir = output / spec.spec_id
        try:
            template_output = run_uplift_template(
                spec,
                train_df=split.train,
                eval_df=eval_df,
                entity_key=contract.entity_key,
                treatment_col=contract.treatment_column,
                target_col=contract.target_column,
                cutoff_grid=contract.evaluation_policy.cutoff_grid,
                held_out_df=held_out_df,
            )
            artifact_paths = _write_trial_artifacts(
                trial_dir,
                predictions=template_output.predictions,
                decile_table=template_output.decile_table,
                qini_curve=template_output.qini_curve,
                uplift_curve=template_output.uplift_curve,
                result_json=template_output.result_card.model_dump_json(indent=2),
                held_out_predictions=template_output.held_out_predictions,
                held_out_decile_table=template_output.held_out_decile_table,
                held_out_qini_curve=template_output.held_out_qini_curve,
                held_out_uplift_curve=template_output.held_out_uplift_curve,
            )
            record = ledger.append_result(
                trial_spec=spec,
                feature_artifact_id=feature_artifact.feature_artifact_id,
                result_status=template_output.result_card.status,
                qini_auc=template_output.result_card.qini_auc,
                uplift_auc=template_output.result_card.uplift_auc,
                uplift_at_k=template_output.result_card.uplift_at_k,
                policy_gain=template_output.result_card.policy_gain,
                held_out_qini_auc=template_output.result_card.held_out_qini_auc,
                held_out_uplift_auc=template_output.result_card.held_out_uplift_auc,
                held_out_uplift_at_k=template_output.result_card.held_out_uplift_at_k,
                held_out_policy_gain=template_output.result_card.held_out_policy_gain,
                artifact_paths=artifact_paths,
            )
        except Exception as exc:
            record = ledger.append_result(
                trial_spec=spec,
                feature_artifact_id=feature_artifact.feature_artifact_id,
                result_status="failed",
                error=str(exc),
                artifact_paths={},
            )
        records.append(record)

    return UpliftLoopResult(
        records=records,
        ledger_path=str(ledger.path),
        output_dir=str(output),
    )
