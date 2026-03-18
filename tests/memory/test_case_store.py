import pytest, json
from pathlib import Path
from src.memory.case_store import CaseStore
from src.models.nodes import CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory
from src.models.task import ExperimentPlan


def _make_case(case_id: str, task_type: str = "binary") -> CaseEntry:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    return CaseEntry(
        case_id=case_id,
        task_traits=TaskTraits(
            task_type=task_type, n_rows_bucket="medium",
            n_features_bucket="medium", class_balance="balanced",
        ),
        what_worked=WhatWorked(best_config=plan, best_metric=0.85),
        what_failed=WhatFailed(),
        trajectory=SessionTrajectory(n_runs=3),
    )


def test_add_and_get_all(tmp_path):
    store = CaseStore(str(tmp_path / "cases.jsonl"))
    store.add(_make_case("c1"))
    store.add(_make_case("c2"))
    all_cases = store.get_all()
    assert len(all_cases) == 2
    assert all_cases[0].case_id == "c1"


def test_persists_across_instances(tmp_path):
    path = str(tmp_path / "cases.jsonl")
    store = CaseStore(path)
    store.add(_make_case("c1"))
    store2 = CaseStore(path)
    assert len(store2.get_all()) == 1


def test_empty_store_returns_empty_list(tmp_path):
    store = CaseStore(str(tmp_path / "cases.jsonl"))
    assert store.get_all() == []
