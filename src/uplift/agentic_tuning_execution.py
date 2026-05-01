"""Execute deterministic agentic tuning plans with existing uplift runners."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftTrialSpec,
)
from src.uplift.ledger import UpliftLedger
from src.uplift.loop import run_uplift_trials
from src.uplift.tuning import select_stable_tuning_record, tuning_summary


@dataclass(frozen=True)
class AgenticTuningExecutionResult:
    """Summary of a completed agentic tuning plan execution."""

    records: list[UpliftExperimentRecord]
    ledger_path: str
    summary_path: str
    output_dir: str
    group_outputs: dict[str, str]
    champion_run_id: str | None
    champion_hypothesis_id: str | None
    champion_template_name: str | None
    champion_qini_auc: float | None


def load_agentic_tuning_plan(path: str | Path) -> dict[str, Any]:
    """Load a tuning plan JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def trial_specs_from_plan(plan: Mapping[str, Any]) -> list[UpliftTrialSpec]:
    """Return executable trial specs from a dry-run tuning plan."""
    raw_specs = plan.get("trial_specs", [])
    if not isinstance(raw_specs, list):
        raise ValueError("agentic tuning plan must contain a trial_specs list")
    return [UpliftTrialSpec.model_validate(spec) for spec in raw_specs]


def feature_artifacts_from_metadata(
    metadata_paths: Iterable[str | Path],
) -> dict[str, UpliftFeatureArtifact]:
    """Load feature artifacts keyed by feature_recipe_id from metadata JSON files."""
    artifacts: dict[str, UpliftFeatureArtifact] = {}
    for path in metadata_paths:
        artifact = UpliftFeatureArtifact.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
        artifacts[artifact.feature_recipe_id] = artifact
    return artifacts


def execute_agentic_tuning_plan(
    contract: UpliftProjectContract,
    *,
    plan_path: str | Path,
    feature_artifacts_by_recipe_id: Mapping[str, UpliftFeatureArtifact],
    output_dir: str | Path,
) -> AgenticTuningExecutionResult:
    """Execute all specs in a tuning plan and write combined audit artifacts."""
    plan_path = Path(plan_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    plan = load_agentic_tuning_plan(plan_path)
    specs = trial_specs_from_plan(plan)
    grouped_specs = _group_specs_by_recipe(specs)
    missing = sorted(
        recipe_id
        for recipe_id in grouped_specs
        if recipe_id not in feature_artifacts_by_recipe_id
    )
    if missing:
        raise ValueError(
            "missing feature artifact for tuning plan feature_recipe_id(s): "
            + ", ".join(missing)
        )

    records: list[UpliftExperimentRecord] = []
    group_outputs: dict[str, str] = {}
    for recipe_id, recipe_specs in grouped_specs.items():
        group_output = output / f"feature_recipe_{recipe_id}"
        group_outputs[recipe_id] = str(group_output)
        group_ledger = group_output / "uplift_ledger.jsonl"
        if group_ledger.exists():
            group_ledger.unlink()
        result = run_uplift_trials(
            contract,
            feature_artifact=feature_artifacts_by_recipe_id[recipe_id],
            trial_specs=recipe_specs,
            output_dir=group_output,
        )
        records.extend(result.records)

    combined_ledger = UpliftLedger(output / "uplift_ledger.jsonl")
    if combined_ledger.path.exists():
        combined_ledger.path.unlink()
    for record in records:
        combined_ledger.append(record)

    champion = select_stable_tuning_record(records)
    summary_path = output / "tuning_execution_summary.json"
    summary_path.write_text(
        json.dumps(
            _execution_summary(
                plan_path=plan_path,
                output_dir=output,
                specs=specs,
                records=records,
                group_outputs=group_outputs,
                ledger_path=combined_ledger.path,
                champion=champion,
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return AgenticTuningExecutionResult(
        records=records,
        ledger_path=str(combined_ledger.path),
        summary_path=str(summary_path),
        output_dir=str(output),
        group_outputs=group_outputs,
        champion_run_id=champion.run_id if champion else None,
        champion_hypothesis_id=champion.hypothesis_id if champion else None,
        champion_template_name=champion.template_name if champion else None,
        champion_qini_auc=champion.qini_auc if champion else None,
    )


def _group_specs_by_recipe(
    specs: list[UpliftTrialSpec],
) -> dict[str, list[UpliftTrialSpec]]:
    grouped: dict[str, list[UpliftTrialSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.feature_recipe_id].append(spec)
    return dict(grouped)


def _execution_summary(
    *,
    plan_path: Path,
    output_dir: Path,
    specs: list[UpliftTrialSpec],
    records: list[UpliftExperimentRecord],
    group_outputs: dict[str, str],
    ledger_path: Path,
    champion: UpliftExperimentRecord | None,
) -> dict[str, Any]:
    return {
        "plan_path": str(plan_path),
        "output_dir": str(output_dir),
        "ledger_path": str(ledger_path),
        "n_trial_specs": len(specs),
        "n_records": len(records),
        "group_outputs": group_outputs,
        "champion": _champion_summary(champion),
        "records": tuning_summary(records),
    }


def _champion_summary(record: UpliftExperimentRecord | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "run_id": record.run_id,
        "hypothesis_id": record.hypothesis_id,
        "template_name": record.template_name,
        "learner_family": record.uplift_learner_family,
        "base_estimator": record.base_estimator,
        "feature_recipe_id": record.feature_recipe_id,
        "params_hash": record.params_hash,
        "qini_auc": record.qini_auc,
        "uplift_auc": record.uplift_auc,
        "selection_score_source": "validation_only",
    }
