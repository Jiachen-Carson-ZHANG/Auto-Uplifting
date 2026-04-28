"""Hypothesis lifecycle helpers for the uplift supervisor."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable

from src.models.uplift import (
    UpliftActionType,
    UpliftExperimentRecord,
    UpliftHypothesis,
    UpliftHypothesisStatus,
)


class InvalidHypothesisTransitionError(ValueError):
    """Raised when a hypothesis lifecycle transition is not allowed."""


_ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "proposed": {"under_test", "retired"},
    "under_test": {"supported", "contradicted", "inconclusive", "retired"},
    "inconclusive": {"under_test", "retired"},
    "supported": {"retired"},
    "contradicted": {"retired"},
    "retired": set(),
}

_TERMINAL_STATUSES = {"supported", "contradicted", "retired"}


def transition_hypothesis(
    hypothesis: UpliftHypothesis,
    new_status: UpliftHypothesisStatus,
    *,
    wave_id: str | None = None,
    trial_ids: Iterable[str] | None = None,
    next_action: str | None = None,
) -> UpliftHypothesis:
    """Return a copy of hypothesis after validating a lifecycle transition."""
    allowed = _ALLOWED_TRANSITIONS[hypothesis.status]
    if new_status == hypothesis.status and hypothesis.status in _TERMINAL_STATUSES:
        raise InvalidHypothesisTransitionError(
            f"invalid hypothesis transition: {hypothesis.status} -> {new_status}"
        )
    if new_status not in allowed and new_status != hypothesis.status:
        raise InvalidHypothesisTransitionError(
            f"invalid hypothesis transition: {hypothesis.status} -> {new_status}"
        )

    wave_ids = list(hypothesis.wave_ids)
    if wave_id:
        wave_ids.append(wave_id)

    merged_trial_ids = list(hypothesis.trial_ids)
    if trial_ids:
        merged_trial_ids.extend(trial_ids)

    return hypothesis.model_copy(
        update={
            "status": new_status,
            "wave_ids": list(dict.fromkeys(wave_ids)),
            "trial_ids": list(dict.fromkeys(merged_trial_ids)),
            "next_action": next_action,
            "updated_at": datetime.now(),
        }
    )


def link_ledger_records(
    hypothesis: UpliftHypothesis,
    records: Iterable[UpliftExperimentRecord],
) -> UpliftHypothesis:
    """Return a copy linked to ledger records for the same hypothesis."""
    run_ids: list[str] = []
    for record in records:
        if record.hypothesis_id != hypothesis.hypothesis_id:
            raise ValueError(
                f"ledger record {record.run_id} does not match hypothesis "
                f"{hypothesis.hypothesis_id}"
            )
        run_ids.append(record.run_id)
    return transition_hypothesis(
        hypothesis,
        hypothesis.status,
        trial_ids=run_ids,
        next_action=hypothesis.next_action,
    )


class UpliftHypothesisStore:
    """JSONL-backed append-only store for uplift hypothesis snapshots."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, hypothesis: UpliftHypothesis) -> UpliftHypothesis:
        """Append one hypothesis snapshot and return it."""
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(hypothesis.model_dump_json() + "\n")
        return hypothesis

    def load_snapshots(self) -> list[UpliftHypothesis]:
        """Load every hypothesis snapshot from disk in append order."""
        if not self.path.exists():
            return []
        snapshots: list[UpliftHypothesis] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    snapshots.append(UpliftHypothesis.model_validate_json(line))
        return snapshots

    def latest_by_id(self) -> dict[str, UpliftHypothesis]:
        """Return the latest snapshot for each hypothesis ID."""
        latest: dict[str, UpliftHypothesis] = {}
        for hypothesis in self.load_snapshots():
            latest[hypothesis.hypothesis_id] = hypothesis
        return latest

    def get_latest(self, hypothesis_id: str) -> UpliftHypothesis | None:
        """Return the latest snapshot for one hypothesis ID, if present."""
        return self.latest_by_id().get(hypothesis_id)

    def query_by_status(
        self, status: UpliftHypothesisStatus
    ) -> list[UpliftHypothesis]:
        """Return latest hypotheses with the requested status."""
        return [
            hypothesis
            for hypothesis in self.latest_by_id().values()
            if hypothesis.status == status
        ]

    def query_by_action_type(
        self, action_type: UpliftActionType
    ) -> list[UpliftHypothesis]:
        """Return latest hypotheses with the requested action type."""
        return [
            hypothesis
            for hypothesis in self.latest_by_id().values()
            if hypothesis.action_type == action_type
        ]

    def query_by_trial_id(self, trial_id: str) -> list[UpliftHypothesis]:
        """Return latest hypotheses linked to a ledger run ID."""
        return [
            hypothesis
            for hypothesis in self.latest_by_id().values()
            if trial_id in hypothesis.trial_ids
        ]
