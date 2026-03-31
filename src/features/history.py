"""
FeatureHistoryStore — JSONL append-only empirical experiment memory.

This is empirical memory for the feature engineering loop: what we tried,
what happened, and what to remember. Static external knowledge belongs
in references/.

Follows src/memory/run_store.py pattern exactly.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Union

from src.models.feature_engineering import FeatureHistoryEntry


class FeatureHistoryStore:
    """Append-only JSONL store for feature engineering telemetry."""

    def __init__(self, journal_path: Union[str, Path]) -> None:
        self._path = Path(journal_path)
        self._entries: List[FeatureHistoryEntry] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None:
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self._entries.append(
                        FeatureHistoryEntry.model_validate_json(line)
                    )

    def add(self, entry: FeatureHistoryEntry) -> None:
        """Append entry to in-memory list and write to disk atomically."""
        self._entries.append(entry)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get_history(self) -> List[FeatureHistoryEntry]:
        """Return copy of all entries."""
        return list(self._entries)

    def get_by_dataset(self, dataset_name: str) -> List[FeatureHistoryEntry]:
        """Return entries filtered by dataset_name."""
        return [e for e in self._entries if e.dataset_name == dataset_name]

    def get_recent(self, n: int = 10) -> List[FeatureHistoryEntry]:
        """Return last n entries."""
        return list(self._entries[-n:])
