# tests/memory/test_context_builder.py
import pytest
from src.memory.context_builder import ContextBuilder
from src.models.nodes import SearchContext, ExperimentNode, NodeStage
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry, RunResult, RunDiagnostics, RunConfig


def _make_node() -> ExperimentNode:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    return ExperimentNode(node_id="n1", plan=plan, stage=NodeStage.OPTIMIZE)


def _make_task() -> TaskSpec:
    return TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test", constraints={},
    )


def _make_profile() -> DataProfile:
    return DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.6, missing_rate=0.05)


def _make_run(run_id: str, status: str, metric: float = None) -> RunEntry:
    config = RunConfig(
        run_id=run_id, node_id="n1", autogluon_kwargs={},
        data_path="data/train.csv", output_dir=f"experiments/runs/{run_id}",
    )
    result = RunResult(
        run_id=run_id, status=status,
        primary_metric=metric, fit_time_seconds=10.0, artifacts_dir="",
    )
    return RunEntry(run_id=run_id, node_id="n1", config=config,
                    result=result, diagnostics=RunDiagnostics())


def test_build_returns_search_context():
    builder = ContextBuilder()
    context = builder.build(
        task=_make_task(), data_profile=_make_profile(),
        history=[], incumbent=None, current_node=_make_node(),
        stage="optimize", budget_remaining=4, budget_used=1, similar_cases=[],
    )
    assert isinstance(context, SearchContext)
    assert context.stage == "optimize"
    assert context.budget_remaining == 4
    assert context.budget_used == 1


def test_failed_attempts_filtered_from_history():
    builder = ContextBuilder()
    failed = _make_run("r1", "failed")
    success = _make_run("r2", "success", metric=0.85)

    context = builder.build(
        task=_make_task(), data_profile=_make_profile(),
        history=[failed, success], incumbent=None,
        current_node=_make_node(), stage="optimize",
        budget_remaining=3, budget_used=2, similar_cases=[],
    )
    assert len(context.failed_attempts) == 1
    assert context.failed_attempts[0].run_id == "r1"
    assert len(context.history) == 2  # full history preserved


def test_similar_cases_passed_through():
    from src.models.nodes import CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    case = CaseEntry(
        case_id="c1",
        task_traits=TaskTraits(task_type="binary", n_rows_bucket="medium",
                               n_features_bucket="medium", class_balance="balanced"),
        what_worked=WhatWorked(best_config=plan, best_metric=0.85),
        what_failed=WhatFailed(), trajectory=SessionTrajectory(n_runs=2),
    )
    builder = ContextBuilder()
    context = builder.build(
        task=_make_task(), data_profile=_make_profile(),
        history=[], incumbent=None, current_node=_make_node(),
        stage="warmup", budget_remaining=5, budget_used=0, similar_cases=[case],
    )
    assert len(context.similar_cases) == 1
    assert context.similar_cases[0].case_id == "c1"
