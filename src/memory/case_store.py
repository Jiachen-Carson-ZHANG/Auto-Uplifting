"""
RETIRED FROM ACTIVE ARCHITECTURE: This module is retained only for backward
compatibility with CampaignOrchestrator. New feature engineering campaigns use
empirical experiment memory (FeatureHistoryStore) plus static reference packs instead.
See docs/plans/2026-03-30-ecommerce-feature-engineering-design.md for rationale.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
from src.models.nodes import CaseEntry


class CaseStore:
    """Append-only JSONL store for cross-session CaseEntry knowledge."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[CaseEntry] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    self._entries.append(CaseEntry.model_validate_json(line))

    def add(self, case: CaseEntry) -> None:
        self._entries.append(case)
        with self._path.open("a") as f:
            f.write(case.model_dump_json() + "\n")

    def get_all(self) -> List[CaseEntry]:
        return list(self._entries)
