"""Manual uplift wave validation and execution."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftExperimentWaveSpec,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftWaveResult,
)
from src.uplift.hypotheses import UpliftHypothesisStore, transition_hypothesis
from src.uplift.loop import run_uplift_trials
from src.uplift.templates import REGISTERED_UPLIFT_TEMPLATES


APPROVED_WINDOW_SWEEP_DAYS = {1, 7, 14, 30, 60, 90, 180}
KNOWN_ABLATION_FEATURE_GROUPS = {
    "demographic",
    "rfm",
    "basket",
    "points",
    "product_category",
    "diversity",
}
APPROVED_EXPANSION_FEATURE_GROUPS = {"product_category", "diversity"}


def validate_wave_spec(
    wave_spec: UpliftExperimentWaveSpec,
    *,
    feature_artifacts: Mapping[str, UpliftFeatureArtifact],
    hypothesis_store: UpliftHypothesisStore | None = None,
) -> None:
    """Validate runtime wave dependencies before executing trial kernels."""
    missing_recipes = [
        feature_recipe_id
        for feature_recipe_id in wave_spec.required_feature_recipe_ids
        if feature_recipe_id not in feature_artifacts
    ]
    if missing_recipes:
        raise ValueError(
            "unknown feature recipe ids for wave "
            f"{wave_spec.wave_id}: {', '.join(missing_recipes)}"
        )

    wave_artifacts = [
        feature_artifacts[feature_recipe_id]
        for feature_recipe_id in wave_spec.required_feature_recipe_ids
    ]
    for feature_recipe_id, artifact in zip(
        wave_spec.required_feature_recipe_ids, wave_artifacts
    ):
        if artifact.feature_recipe_id != feature_recipe_id:
            raise ValueError(
                "feature artifact key does not match feature_recipe_id: "
                f"{feature_recipe_id}"
            )

    for trial_spec in wave_spec.trial_specs:
        expected_family = REGISTERED_UPLIFT_TEMPLATES.get(trial_spec.template_name)
        if expected_family is None:
            raise ValueError(f"unknown uplift template: {trial_spec.template_name}")
        if expected_family != trial_spec.learner_family:
            raise ValueError(
                f"template {trial_spec.template_name} expects "
                f"learner_family={expected_family}"
            )

    if hypothesis_store is not None:
        hypothesis = hypothesis_store.get_latest(wave_spec.hypothesis_id)
        if hypothesis is None:
            raise ValueError(f"unknown hypothesis_id: {wave_spec.hypothesis_id}")

    if wave_spec.action_type == "window_sweep":
        _validate_window_sweep_artifacts(wave_artifacts)
    elif wave_spec.action_type == "feature_ablation":
        _validate_feature_ablation_artifacts(wave_artifacts)
    elif wave_spec.action_type == "feature_group_expansion":
        _validate_feature_group_expansion_artifacts(wave_artifacts)
    elif wave_spec.action_type == "ranking_stability_check":
        _validate_ranking_stability_wave(wave_spec, wave_artifacts)


class UpliftResearchLoop:
    """Deterministic shell for manually specified uplift experiment waves."""

    def __init__(
        self,
        *,
        contract: UpliftProjectContract,
        feature_artifacts: Mapping[str, UpliftFeatureArtifact],
        output_dir: str | Path,
        hypothesis_store: UpliftHypothesisStore | None = None,
    ) -> None:
        self.contract = contract
        self.feature_artifacts = dict(feature_artifacts)
        self.output_dir = Path(output_dir)
        self.hypothesis_store = hypothesis_store

    def run_wave(self, wave_spec: UpliftExperimentWaveSpec) -> UpliftWaveResult:
        """Validate, execute, and summarize one manual uplift wave."""
        validate_wave_spec(
            wave_spec,
            feature_artifacts=self.feature_artifacts,
            hypothesis_store=self.hypothesis_store,
        )

        wave_output_dir = self.output_dir / wave_spec.wave_id
        records: list[UpliftExperimentRecord] = []
        failed_trial_ids: list[str] = []
        artifact_paths: dict[str, str] = {}
        blocked_reason: str | None = None

        for trial_spec in wave_spec.trial_specs:
            feature_artifact = self.feature_artifacts[trial_spec.feature_recipe_id]
            loop_result = run_uplift_trials(
                self.contract,
                feature_artifact=feature_artifact,
                trial_specs=[trial_spec],
                output_dir=wave_output_dir,
            )
            artifact_paths["ledger"] = loop_result.ledger_path

            if not loop_result.records:
                blocked_reason = f"trial {trial_spec.spec_id} produced no ledger record"
                if wave_spec.abort_on_first_failure:
                    break
                continue

            for record in loop_result.records:
                records.append(record)
                artifact_paths.update(_namespaced_artifacts(record))
                if record.status != "success":
                    failed_trial_ids.append(record.run_id)
                    if wave_spec.abort_on_first_failure:
                        blocked_reason = (
                            record.error
                            or f"trial {record.run_id} failed with status {record.status}"
                        )
                        break
            if blocked_reason and wave_spec.abort_on_first_failure:
                break

        trial_ids = [record.run_id for record in records]
        successful_records = [record for record in records if record.status == "success"]
        champion_run_id = _select_champion_run_id(
            successful_records,
            primary_metric=wave_spec.trial_specs[0].primary_metric,
            higher_is_better=self.contract.evaluation_policy.higher_is_better,
        )
        status = _wave_status(
            records=records,
            failed_trial_ids=failed_trial_ids,
            blocked_reason=blocked_reason,
        )

        result = UpliftWaveResult(
            wave_id=wave_spec.wave_id,
            hypothesis_id=wave_spec.hypothesis_id,
            action_type=wave_spec.action_type,
            status=status,
            trial_ids=trial_ids,
            failed_trial_ids=failed_trial_ids,
            blocked_reason=blocked_reason,
            champion_run_id=champion_run_id,
            artifact_paths=artifact_paths,
        )
        self._link_hypothesis(wave_spec, result)
        return result

    def _link_hypothesis(
        self,
        wave_spec: UpliftExperimentWaveSpec,
        result: UpliftWaveResult,
    ) -> None:
        if self.hypothesis_store is None:
            return
        hypothesis = self.hypothesis_store.get_latest(wave_spec.hypothesis_id)
        if hypothesis is None:
            return
        linked = transition_hypothesis(
            hypothesis,
            "under_test",
            wave_id=wave_spec.wave_id,
            trial_ids=result.trial_ids,
            next_action="review wave result",
        )
        self.hypothesis_store.append(linked)


def _namespaced_artifacts(record: UpliftExperimentRecord) -> dict[str, str]:
    return {
        f"{record.run_id}:{artifact_name}": artifact_path
        for artifact_name, artifact_path in record.artifact_paths.items()
    }


def _validate_window_sweep_artifacts(
    artifacts: list[UpliftFeatureArtifact],
) -> None:
    windows: list[int] = []
    for artifact in artifacts:
        if not artifact.feature_groups:
            raise ValueError("window_sweep requires feature_groups metadata")
        if len(artifact.windows_days) != 1:
            raise ValueError("window_sweep requires exactly one window per recipe")
        window = artifact.windows_days[0]
        if window not in APPROVED_WINDOW_SWEEP_DAYS:
            raise ValueError(f"unapproved window for window_sweep: {window}")
        windows.append(window)

    if len(set(windows)) != len(windows):
        raise ValueError("window_sweep requires distinct windows")
    _require_same_artifact_signature(
        artifacts,
        signature_fn=_window_sweep_signature,
        message="window_sweep recipes must differ only by approved window",
    )


def _validate_feature_ablation_artifacts(
    artifacts: list[UpliftFeatureArtifact],
) -> None:
    if len(artifacts) != 2:
        raise ValueError("feature_ablation requires exactly two feature recipes")

    group_sets: list[set[str]] = []
    for artifact in artifacts:
        if not artifact.feature_groups:
            raise ValueError("feature_ablation requires feature_groups metadata")
        groups = set(artifact.feature_groups)
        unknown_groups = sorted(groups - KNOWN_ABLATION_FEATURE_GROUPS)
        if unknown_groups:
            raise ValueError(
                "unknown feature group for feature_ablation: "
                f"{', '.join(unknown_groups)}"
            )
        group_sets.append(groups)

    full_groups, ablated_groups = sorted(group_sets, key=len, reverse=True)
    removed_groups = full_groups - ablated_groups
    added_groups = ablated_groups - full_groups
    if added_groups or len(removed_groups) != 1:
        raise ValueError(
            "feature_ablation requires exactly one known feature group removed"
        )

    _require_same_artifact_signature(
        artifacts,
        signature_fn=_feature_ablation_signature,
        message="feature_ablation recipes must differ only by feature group",
    )


def _validate_feature_group_expansion_artifacts(
    artifacts: list[UpliftFeatureArtifact],
) -> None:
    if len(artifacts) != 2:
        raise ValueError("feature_group_expansion requires exactly two feature recipes")

    for artifact in artifacts:
        if not artifact.feature_groups:
            raise ValueError("feature_group_expansion requires feature_groups metadata")

    left_groups = set(artifacts[0].feature_groups)
    right_groups = set(artifacts[1].feature_groups)
    if left_groups < right_groups:
        base_artifact, expanded_artifact = artifacts[0], artifacts[1]
        added_groups = right_groups - left_groups
    elif right_groups < left_groups:
        base_artifact, expanded_artifact = artifacts[1], artifacts[0]
        added_groups = left_groups - right_groups
    else:
        raise ValueError(
            "feature_group_expansion requires one recipe to be a strict "
            "feature-group superset"
        )

    if not added_groups.intersection(APPROVED_EXPANSION_FEATURE_GROUPS):
        raise ValueError(
            "feature_group_expansion requires at least one approved expansion "
            "feature group"
        )

    if not set(base_artifact.source_tables).issubset(expanded_artifact.source_tables):
        raise ValueError(
            "feature_group_expansion source tables must expand from the base recipe"
        )

    _require_same_artifact_signature(
        artifacts,
        signature_fn=_feature_group_expansion_signature,
        message=(
            "feature_group_expansion recipes must share dataset, builder, cohort, "
            "entity key, and row count"
        ),
    )


def _validate_ranking_stability_wave(
    wave_spec: UpliftExperimentWaveSpec,
    artifacts: list[UpliftFeatureArtifact],
) -> None:
    if len(artifacts) != 1:
        raise ValueError("ranking_stability_check requires exactly one feature recipe")
    split_seeds = [spec.split_seed for spec in wave_spec.trial_specs]
    if len(set(split_seeds)) != len(split_seeds):
        raise ValueError("ranking_stability_check requires distinct split_seed values")


def _require_same_artifact_signature(
    artifacts: list[UpliftFeatureArtifact],
    *,
    signature_fn,
    message: str,
) -> None:
    signatures = [signature_fn(artifact) for artifact in artifacts]
    if len(set(signatures)) != 1:
        raise ValueError(message)


def _window_sweep_signature(artifact: UpliftFeatureArtifact) -> tuple[object, ...]:
    return (
        tuple(artifact.source_tables),
        tuple(artifact.feature_groups),
        artifact.dataset_fingerprint,
        artifact.builder_version,
        artifact.reference_date,
        artifact.cohort,
        artifact.entity_key,
    )


def _feature_ablation_signature(artifact: UpliftFeatureArtifact) -> tuple[object, ...]:
    return (
        tuple(artifact.source_tables),
        tuple(artifact.windows_days),
        artifact.dataset_fingerprint,
        artifact.builder_version,
        artifact.reference_date,
        artifact.cohort,
        artifact.entity_key,
    )


def _feature_group_expansion_signature(
    artifact: UpliftFeatureArtifact,
) -> tuple[object, ...]:
    return (
        artifact.dataset_fingerprint,
        artifact.builder_version,
        artifact.cohort,
        artifact.entity_key,
        artifact.row_count,
    )


def _select_champion_run_id(
    records: list[UpliftExperimentRecord],
    *,
    primary_metric: str,
    higher_is_better: bool,
) -> str | None:
    if not records:
        return None

    scored_records = [
        (record, _numeric_metric(record, primary_metric)) for record in records
    ]
    finite_records = [
        (record, score)
        for record, score in scored_records
        if score is not None and math.isfinite(score)
    ]
    if not finite_records:
        return records[0].run_id

    if higher_is_better:
        return max(finite_records, key=lambda item: item[1])[0].run_id
    return min(finite_records, key=lambda item: item[1])[0].run_id


def _numeric_metric(record: UpliftExperimentRecord, metric_name: str) -> float | None:
    value = getattr(record, metric_name, None)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _wave_status(
    *,
    records: list[UpliftExperimentRecord],
    failed_trial_ids: list[str],
    blocked_reason: str | None,
) -> str:
    if blocked_reason is not None:
        return "blocked"
    if not records:
        return "failed"
    successful_count = sum(record.status == "success" for record in records)
    if failed_trial_ids and successful_count:
        return "partially_completed"
    if failed_trial_ids:
        return "failed"
    return "completed"
