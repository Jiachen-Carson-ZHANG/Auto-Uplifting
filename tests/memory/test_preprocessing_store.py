"""Tests for PreprocessingStore — append-only JSONL store for preprocessing entries."""
from __future__ import annotations
import json
from pathlib import Path
import pytest
from src.memory.preprocessing_store import PreprocessingStore
from src.models.preprocessing import PreprocessingEntry


def make_entry(entry_id: str = "e1", task_type: str = "binary") -> PreprocessingEntry:
    return PreprocessingEntry(
        entry_id=entry_id,
        task_type=task_type,
        dataset_name="titanic",
        transformation_summary="derived FamilySize",
        code="def preprocess(df):\n    return df\n",
    )


# ── Persistence ───────────────────────────────────────────────────────────────

def test_add_persists_to_disk(tmp_path):
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    store.add(make_entry("e1"))
    lines = (tmp_path / "bank.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["entry_id"] == "e1"


def test_reload_from_disk(tmp_path):
    path = str(tmp_path / "bank.jsonl")
    s1 = PreprocessingStore(path)
    s1.add(make_entry("e1"))
    s1.add(make_entry("e2"))

    s2 = PreprocessingStore(path)
    assert len(s2.get_all()) == 2
    assert s2.get_all()[0].entry_id == "e1"


def test_malformed_line_skipped(tmp_path):
    path = tmp_path / "bank.jsonl"
    path.write_text('{"entry_id": "e1", "task_type": "binary", "dataset_name": "d", "transformation_summary": "s", "code": "def preprocess(df): return df"}\nnot valid json\n')
    store = PreprocessingStore(str(path))
    assert len(store.get_all()) == 1


# ── get_similar ───────────────────────────────────────────────────────────────

def test_get_similar_filters_by_task_type(tmp_path):
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    store.add(make_entry("bin1", "binary"))
    store.add(make_entry("bin2", "binary"))
    store.add(make_entry("reg1", "regression"))

    similar = store.get_similar("binary", n=10)
    assert len(similar) == 2
    assert all(e.task_type == "binary" for e in similar)


def test_get_similar_respects_n_limit(tmp_path):
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    for i in range(5):
        store.add(make_entry(f"e{i}", "binary"))

    similar = store.get_similar("binary", n=2)
    assert len(similar) == 2


def test_get_similar_returns_empty_for_no_match(tmp_path):
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    store.add(make_entry("e1", "binary"))
    assert store.get_similar("regression") == []


# ── Seeds ─────────────────────────────────────────────────────────────────────

def test_seeds_load_without_error():
    """Verify the seed file is valid JSONL and all entries parse correctly."""
    seed_path = "data/seeds/preprocessing_seeds.jsonl"
    store = PreprocessingStore(seed_path)
    entries = store.get_all()
    assert len(entries) >= 5
    assert all(e.entry_id.startswith("seed-") for e in entries)


def test_seeds_get_similar_binary():
    store = PreprocessingStore("data/seeds/preprocessing_seeds.jsonl")
    similar = store.get_similar("binary", n=3)
    assert len(similar) == 3
    assert all(e.task_type == "binary" for e in similar)


# ── seed_from_file ────────────────────────────────────────────────────────────

def test_seed_from_file_adds_entries(tmp_path):
    """seed_from_file loads entries from the seeds JSONL into a fresh bank."""
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    assert store.get_all() == []
    store.seed_from_file("data/seeds/preprocessing_seeds.jsonl")
    assert len(store.get_all()) >= 5


def test_seed_from_file_is_idempotent(tmp_path):
    """Calling seed_from_file twice does not duplicate entries."""
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    store.seed_from_file("data/seeds/preprocessing_seeds.jsonl")
    count_first = len(store.get_all())
    store.seed_from_file("data/seeds/preprocessing_seeds.jsonl")
    assert len(store.get_all()) == count_first


def test_seed_from_file_embeds_with_mock_backend(tmp_path):
    """When embed_backend is provided, entries get non-null embeddings."""
    from unittest.mock import MagicMock
    mock_backend = MagicMock()
    mock_backend.embed_batch.return_value = [[0.1, 0.2]] * 20  # one vector per seed

    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    store.seed_from_file("data/seeds/preprocessing_seeds.jsonl", embed_backend=mock_backend)

    entries = store.get_all()
    assert all(e.embedding is not None for e in entries)
    mock_backend.embed_batch.assert_called_once()


def test_seed_from_file_missing_path_does_not_raise(tmp_path):
    """If the seed file does not exist, seed_from_file logs a warning and returns."""
    store = PreprocessingStore(str(tmp_path / "bank.jsonl"))
    store.seed_from_file("nonexistent/path/seeds.jsonl")  # must not raise
    assert store.get_all() == []
