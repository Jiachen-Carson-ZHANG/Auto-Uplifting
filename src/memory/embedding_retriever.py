"""
RETIRED FROM ACTIVE ARCHITECTURE: This module is retained only for backward
compatibility with CampaignOrchestrator. New feature engineering campaigns use
empirical experiment memory (FeatureHistoryStore) plus static reference packs instead.
See docs/plans/2026-03-30-ecommerce-feature-engineering-design.md for rationale.

EmbeddingRetriever — ranks PreprocessingEntry candidates by embedding cosine similarity.

Uses OpenAIBackend.embed() to embed the query text, then cosine-ranks stored entries
whose embeddings are non-None. Falls back to returning all non-None candidates ranked
by metric_delta if embedding is unavailable.

A/B comparison: logs CaseRetriever-style top-3 (by task_type filter order) alongside
embedding top-3 so we can compare quality in production logs.
"""
from __future__ import annotations
import logging
from typing import List, Optional, TYPE_CHECKING

import numpy as np

from src.models.preprocessing import PreprocessingEntry

if TYPE_CHECKING:
    from src.llm.providers.openai import OpenAIBackend

logger = logging.getLogger(__name__)


def _cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(va, vb) / (norm_a * norm_b))


class EmbeddingRetriever:
    """
    Ranks PreprocessingEntry candidates by embedding cosine similarity.

    Requires OpenAIBackend (not the abstract LLMBackend) because embed() is
    only defined on the concrete class — it is not part of the LLM ABC.
    """

    def __init__(self, backend: "OpenAIBackend") -> None:
        self._backend = backend

    def rank(
        self,
        query_text: str,
        candidates: List[PreprocessingEntry],
        top_k: int = 3,
    ) -> List[PreprocessingEntry]:
        """
        Return top_k candidates ranked by embedding cosine similarity.

        A/B log: also logs the unranked (task_type order) top-3 alongside
        the embedding top-3 so both strategies are visible in the log.
        """
        if not candidates:
            return []

        # A/B baseline: naive order (same as PreprocessingStore.get_similar)
        naive_top = [e.entry_id for e in candidates[:top_k]]
        logger.debug("EmbeddingRetriever A/B — naive top-%d: %s", top_k, naive_top)

        query_vec: Optional[List[float]] = self._backend.embed(query_text)
        if query_vec is None:
            logger.warning("EmbeddingRetriever: embed() returned None, falling back to naive order")
            return candidates[:top_k]

        scored: List[tuple[float, PreprocessingEntry]] = []
        for entry in candidates:
            if entry.embedding is None:
                # no stored embedding — use metric_delta as tiebreaker (already have score=0)
                scored.append((0.0, entry))
            else:
                sim = _cosine_sim(query_vec, entry.embedding)
                scored.append((sim, entry))

        scored.sort(key=lambda t: t[0], reverse=True)
        result = [e for _, e in scored[:top_k]]

        embedding_top = [e.entry_id for e in result]
        logger.debug("EmbeddingRetriever A/B — embedding top-%d: %s", top_k, embedding_top)

        return result
