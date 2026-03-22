"""Tests for EmbeddingRetriever — mocked OpenAIBackend, cosine ranking."""
from __future__ import annotations
from typing import List, Optional
from unittest.mock import MagicMock
import pytest
from src.memory.embedding_retriever import EmbeddingRetriever
from src.models.preprocessing import PreprocessingEntry


def make_backend(embed_return: Optional[List[float]]) -> MagicMock:
    mock = MagicMock()
    mock.embed.return_value = embed_return
    return mock


def make_entry(entry_id: str, embedding: Optional[List[float]] = None) -> PreprocessingEntry:
    return PreprocessingEntry(
        entry_id=entry_id,
        task_type="binary",
        dataset_name="titanic",
        transformation_summary=f"transform {entry_id}",
        code="def preprocess(df): return df",
        embedding=embedding,
    )


# ── Happy path ─────────────────────────────────────────────────────────────────

def test_rank_returns_top_k():
    """Returns exactly top_k results when enough candidates exist."""
    backend = make_backend([1.0, 0.0])
    retriever = EmbeddingRetriever(backend)
    candidates = [
        make_entry("e1", [1.0, 0.0]),   # sim = 1.0 (identical)
        make_entry("e2", [0.0, 1.0]),   # sim = 0.0 (orthogonal)
        make_entry("e3", [0.7, 0.7]),   # sim ≈ 0.7
    ]
    result = retriever.rank("query", candidates, top_k=2)
    assert len(result) == 2


def test_rank_orders_by_cosine_similarity():
    """Highest cosine similarity entry comes first."""
    backend = make_backend([1.0, 0.0])
    retriever = EmbeddingRetriever(backend)
    candidates = [
        make_entry("low",  [0.0, 1.0]),  # sim = 0.0
        make_entry("high", [1.0, 0.0]),  # sim = 1.0
    ]
    result = retriever.rank("query", candidates, top_k=2)
    assert result[0].entry_id == "high"
    assert result[1].entry_id == "low"


def test_rank_empty_candidates_returns_empty():
    backend = make_backend([1.0, 0.0])
    assert EmbeddingRetriever(backend).rank("q", [], top_k=3) == []


# ── Embed failure fallback ─────────────────────────────────────────────────────

def test_rank_falls_back_to_naive_when_embed_returns_none():
    """If embed() returns None, returns candidates in original order."""
    backend = make_backend(None)
    retriever = EmbeddingRetriever(backend)
    candidates = [make_entry("e1"), make_entry("e2"), make_entry("e3")]
    result = retriever.rank("q", candidates, top_k=2)
    assert len(result) == 2
    assert result[0].entry_id == "e1"


# ── Entries without embeddings ─────────────────────────────────────────────────

def test_entries_without_embeddings_score_zero():
    """Entries with embedding=None are ranked last (score=0.0)."""
    backend = make_backend([1.0, 0.0])
    retriever = EmbeddingRetriever(backend)
    candidates = [
        make_entry("no_emb", None),        # no embedding → score 0
        make_entry("has_emb", [1.0, 0.0]), # sim = 1.0
    ]
    result = retriever.rank("q", candidates, top_k=2)
    assert result[0].entry_id == "has_emb"


# ── Backend call ───────────────────────────────────────────────────────────────

def test_embed_called_with_query_text():
    backend = make_backend([1.0])
    retriever = EmbeddingRetriever(backend)
    retriever.rank("my query text", [make_entry("e1", [1.0])], top_k=1)
    backend.embed.assert_called_once_with("my query text")
