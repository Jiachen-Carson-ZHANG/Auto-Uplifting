"""
Append-only JSONL store for cross-session PreprocessingEntry records.

Symmetric with CaseStore (case_bank.jsonl) — mirrors the same 25-line pattern.
get_similar() returns entries filtered by task_type; the caller is responsible for
ranking (EmbeddingRetriever) or using the naive order.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from src.models.preprocessing import PreprocessingEntry

if TYPE_CHECKING:
    from src.llm.providers.openai import OpenAIBackend

logger = logging.getLogger(__name__)


class PreprocessingStore:
    """Append-only JSONL store for cross-session PreprocessingEntry knowledge."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: List[PreprocessingEntry] = []
        if self._path.exists():
            for line in self._path.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        self._entries.append(PreprocessingEntry.model_validate_json(line))
                    except Exception as exc:
                        logger.warning("PreprocessingStore: skipping malformed entry: %s", exc)

    def add(self, entry: PreprocessingEntry) -> None:
        self._entries.append(entry)
        try:
            with self._path.open("a") as f:
                f.write(entry.model_dump_json() + "\n")
        except OSError as exc:
            logger.warning("PreprocessingStore: could not write entry %s: %s", entry.entry_id, exc)

    def get_all(self) -> List[PreprocessingEntry]:
        return list(self._entries)

    def get_similar(self, task_type: str, n: int = 3) -> List[PreprocessingEntry]:
        """
        Return up to n entries matching task_type.
        Filters by task_type only — caller may re-rank with EmbeddingRetriever.
        """
        matched = [e for e in self._entries if e.task_type == task_type]
        return matched[:n]

    def seed_from_file(
        self, path: str, embed_backend: Optional["OpenAIBackend"] = None
    ) -> None:
        """
        Load seed entries from a JSONL file and add any not already in the store.

        Idempotent: entries whose entry_id already exists in the bank are skipped.
        Atomic: each seed is appended individually; a crash leaves the bank partially
        seeded and the next run picks up where it left off via the ID check.

        If embed_backend is provided, all new seeds are embedded in a single batch
        API call before being written. Seeds with failed embeddings are stored with
        embedding=None and will score 0.0 in EmbeddingRetriever.
        """
        seed_path = Path(path)
        if not seed_path.exists():
            logger.warning("PreprocessingStore: seed file not found: %s", path)
            return

        seeds: List[PreprocessingEntry] = []
        for line in seed_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    seeds.append(PreprocessingEntry.model_validate_json(line))
                except Exception as exc:
                    logger.warning("PreprocessingStore: skipping malformed seed: %s", exc)

        existing_ids = {e.entry_id for e in self._entries}
        new_seeds = [s for s in seeds if s.entry_id not in existing_ids]
        if not new_seeds:
            return

        if embed_backend is not None:
            logger.info("PreprocessingStore: embedding %d seeds...", len(new_seeds))
            summaries = [s.transformation_summary for s in new_seeds]
            embeddings = embed_backend.embed_batch(summaries)
            for seed, emb in zip(new_seeds, embeddings):
                seed.embedding = emb  # None if batch failed — scores 0.0 in retrieval

        for seed in new_seeds:
            self.add(seed)

        logger.info("PreprocessingStore: added %d seeds from %s", len(new_seeds), path)
