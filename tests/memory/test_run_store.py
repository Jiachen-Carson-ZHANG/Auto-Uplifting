import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.memory.run_store import RunStore
from src.models.task import ExperimentPlan, RunConfig
from src.models.results import RunResult, RunDiagnostics, RunEntry


def make_run_entry(run_id: str, metric: float, status: str = "success") -> RunEntry:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale="test"
    )
    config = RunConfig(
        autogluon_kwargs={"eval_metric": "roc_auc"},
        data_path="data/test.csv",
        output_dir=f"experiments/test/runs/{run_id}",
    )
    result = RunResult(
        status=status,
        primary_metric=metric if status == "success" else None,
        best_model_name="GBM" if status == "success" else None,
        fit_time_seconds=10.0,
    )
    return RunEntry(
        run_id=run_id, node_id="node_001",
        timestamp=datetime(2026, 3, 16, 12, 0, 0),
        config=config, result=result,
        diagnostics=RunDiagnostics(metric_vs_parent=0.05),
        agent_rationale="test rationale", agent_review=""
    )


def test_run_store_append_and_get_history(tmp_path):
    store = RunStore(journal_path=tmp_path / "decisions.jsonl")
    entry = make_run_entry("run_0001", 0.85)
    store.append(entry)
    history = store.get_history()
    assert len(history) == 1
    assert history[0].run_id == "run_0001"
    assert history[0].result.primary_metric == 0.85


def test_run_store_persists_to_jsonl(tmp_path):
    path = tmp_path / "decisions.jsonl"
    store = RunStore(journal_path=path)
    store.append(make_run_entry("run_0001", 0.85))
    store.append(make_run_entry("run_0002", 0.88))
    # Read raw file — should be 2 lines of valid JSON
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    data = json.loads(lines[1])
    assert data["run_id"] == "run_0002"


def test_run_store_get_incumbent(tmp_path):
    store = RunStore(journal_path=tmp_path / "decisions.jsonl")
    store.append(make_run_entry("run_0001", 0.85))
    store.append(make_run_entry("run_0002", 0.88))
    store.append(make_run_entry("run_0003", 0.82))
    incumbent = store.get_incumbent(higher_is_better=True)
    assert incumbent.run_id == "run_0002"


def test_run_store_get_failed(tmp_path):
    store = RunStore(journal_path=tmp_path / "decisions.jsonl")
    store.append(make_run_entry("run_0001", 0.85, status="success"))
    store.append(make_run_entry("run_0002", 0.0, status="failed"))
    failed = store.get_failed()
    assert len(failed) == 1
    assert failed[0].run_id == "run_0002"


def test_run_store_empty(tmp_path):
    store = RunStore(journal_path=tmp_path / "decisions.jsonl")
    assert store.get_history() == []
    assert store.get_incumbent(higher_is_better=True) is None
    assert store.get_failed() == []


def test_run_store_loads_existing_journal(tmp_path):
    path = tmp_path / "decisions.jsonl"
    store1 = RunStore(journal_path=path)
    store1.append(make_run_entry("run_0001", 0.85))
    # New store instance loads from same file
    store2 = RunStore(journal_path=path)
    assert len(store2.get_history()) == 1
