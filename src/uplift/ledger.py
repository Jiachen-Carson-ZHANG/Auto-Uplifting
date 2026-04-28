"""Append-only uplift experiment ledger."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from src.models.uplift import UpliftExperimentRecord, UpliftTrialSpec, _stable_hash


def params_hash(params: Dict[str, object]) -> str:
    """Stable short hash for trial parameters."""
    return _stable_hash(params)


class UpliftLedger:
    """JSONL-backed append-only store for uplift experiment records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: UpliftExperimentRecord) -> UpliftExperimentRecord:
        """Append one record and return it."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
        return record

    def append_result(
        self,
        *,
        trial_spec: UpliftTrialSpec,
        feature_artifact_id: str,
        result_status: str,
        qini_auc: Optional[float] = None,
        uplift_auc: Optional[float] = None,
        uplift_at_k: Optional[Dict[str, float]] = None,
        policy_gain: Optional[Dict[str, float]] = None,
        held_out_qini_auc: Optional[float] = None,
        held_out_uplift_auc: Optional[float] = None,
        held_out_uplift_at_k: Optional[Dict[str, float]] = None,
        held_out_policy_gain: Optional[Dict[str, float]] = None,
        artifact_paths: Optional[Dict[str, str]] = None,
        error: Optional[str] = None,
        verdict: str = "baseline",
    ) -> UpliftExperimentRecord:
        """Create and append a record from one trial outcome."""
        record = UpliftExperimentRecord(
            hypothesis_id=trial_spec.hypothesis_id,
            feature_recipe_id=trial_spec.feature_recipe_id,
            feature_artifact_id=feature_artifact_id,
            template_name=trial_spec.template_name,
            uplift_learner_family=trial_spec.learner_family,
            base_estimator=trial_spec.base_estimator,
            params_hash=params_hash(trial_spec.params),
            split_seed=trial_spec.split_seed,
            status=result_status,
            error=error,
            qini_auc=qini_auc,
            uplift_auc=uplift_auc,
            uplift_at_k=uplift_at_k or {},
            policy_gain=policy_gain or {},
            held_out_qini_auc=held_out_qini_auc,
            held_out_uplift_auc=held_out_uplift_auc,
            held_out_uplift_at_k=held_out_uplift_at_k or {},
            held_out_policy_gain=held_out_policy_gain or {},
            verdict=verdict,
            artifact_paths=artifact_paths or {},
        )
        return self.append(record)

    def load(self) -> List[UpliftExperimentRecord]:
        """Load records from disk."""
        if not self.path.exists():
            return []
        records: List[UpliftExperimentRecord] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(UpliftExperimentRecord.model_validate_json(line))
        return records
