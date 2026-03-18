import pytest
from src.memory.retrieval import CaseRetriever
from src.models.nodes import CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory
from src.models.task import ExperimentPlan


def _plan():
    return ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )

def _case(case_id, task_type, n_rows_bucket, class_balance):
    return CaseEntry(
        case_id=case_id,
        task_traits=TaskTraits(
            task_type=task_type, n_rows_bucket=n_rows_bucket,
            n_features_bucket="medium", class_balance=class_balance,
            feature_types={"numeric": 8, "categorical": 2},
        ),
        what_worked=WhatWorked(best_config=_plan(), best_metric=0.85),
        what_failed=WhatFailed(),
        trajectory=SessionTrajectory(n_runs=3),
    )


def test_returns_top_k(tmp_path):
    cases = [
        _case("c1", "binary", "medium", "balanced"),
        _case("c2", "multiclass", "large", "severe"),
        _case("c3", "binary", "small", "moderate"),
    ]
    query = TaskTraits(
        task_type="binary", n_rows_bucket="medium",
        n_features_bucket="medium", class_balance="balanced",
    )
    retriever = CaseRetriever()
    results = retriever.rank(query, cases, top_k=2)
    assert len(results) == 2
    assert results[0].case_id == "c1"  # exact match should be first


def test_returns_empty_on_no_candidates():
    retriever = CaseRetriever()
    results = retriever.rank(
        TaskTraits(task_type="binary", n_rows_bucket="medium",
                   n_features_bucket="medium", class_balance="balanced"),
        candidates=[],
        top_k=3,
    )
    assert results == []


def test_top_k_capped_at_candidates():
    cases = [_case("c1", "binary", "medium", "balanced")]
    retriever = CaseRetriever()
    results = retriever.rank(
        TaskTraits(task_type="binary", n_rows_bucket="medium",
                   n_features_bucket="medium", class_balance="balanced"),
        candidates=cases,
        top_k=5,
    )
    assert len(results) == 1
