"""Tests for src/features/history.py — FeatureHistoryStore."""
import pytest
from pathlib import Path

from src.features.history import FeatureHistoryStore
from src.models.feature_engineering import FeatureHistoryEntry


def _make_entry(entry_id: str, dataset: str = "test_ds") -> FeatureHistoryEntry:
    return FeatureHistoryEntry(
        entry_id=entry_id,
        action="add",
        feature_spec_json='{"template_name": "rfm_recency"}',
        dataset_name=dataset,
        task_type="binary",
        observed_outcome="improved",
    )


class TestFeatureHistoryStore:
    def test_add_and_get_history(self, tmp_path):
        path = tmp_path / "history.jsonl"
        store = FeatureHistoryStore(path)
        store.add(_make_entry("e1"))
        store.add(_make_entry("e2"))
        assert len(store.get_history()) == 2

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "history.jsonl"
        store1 = FeatureHistoryStore(path)
        store1.add(_make_entry("e1"))
        store1.add(_make_entry("e2"))

        store2 = FeatureHistoryStore(path)
        assert len(store2.get_history()) == 2

    def test_get_by_dataset(self, tmp_path):
        path = tmp_path / "history.jsonl"
        store = FeatureHistoryStore(path)
        store.add(_make_entry("e1", dataset="olist"))
        store.add(_make_entry("e2", dataset="uci"))
        store.add(_make_entry("e3", dataset="olist"))
        assert len(store.get_by_dataset("olist")) == 2
        assert len(store.get_by_dataset("uci")) == 1

    def test_get_recent(self, tmp_path):
        path = tmp_path / "history.jsonl"
        store = FeatureHistoryStore(path)
        for i in range(15):
            store.add(_make_entry(f"e{i}"))
        recent = store.get_recent(5)
        assert len(recent) == 5
        assert recent[-1].entry_id == "e14"

    def test_empty_store(self, tmp_path):
        path = tmp_path / "history.jsonl"
        store = FeatureHistoryStore(path)
        assert store.get_history() == []
        assert store.get_recent() == []

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "dir" / "history.jsonl"
        store = FeatureHistoryStore(path)
        store.add(_make_entry("e1"))
        assert path.exists()
