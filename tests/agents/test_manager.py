import pytest
from unittest.mock import MagicMock
from src.agents.manager import ExperimentManager, Action, ActionType
from src.orchestration.scheduler import Scheduler
from src.orchestration.accept_reject import AcceptReject
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus, SearchContext
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunResult, RunEntry, RunDiagnostics, RunConfig
from src.llm.backend import LLMBackend
from datetime import datetime


def make_plan(metric="roc_auc") -> ExperimentPlan:
    return ExperimentPlan(
        eval_metric=metric, model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale="test"
    )


def make_node(node_id="n1", parent_id=None, stage=NodeStage.WARMUP, metric=None) -> ExperimentNode:
    node = ExperimentNode(
        node_id=node_id, parent_id=parent_id, children=[], edge_label=None,
        stage=stage, status=NodeStatus.SUCCESS if metric else NodeStatus.PENDING,
        plan=make_plan(), depth=0 if parent_id is None else 1,
        debug_depth=0, created_at=datetime.now()
    )
    if metric is not None:
        config = RunConfig(autogluon_kwargs={}, data_path="d", output_dir="o")
        result = RunResult(status="success", primary_metric=metric,
                          best_model_name="GBM", fit_time_seconds=10.0)
        entry = RunEntry(run_id="r1", node_id=node_id,
                        config=config, result=result,
                        diagnostics=RunDiagnostics(), timestamp=datetime.now())
        node = node.model_copy(update={"entry": entry})
    return node


def make_context(stage="warmup", budget_remaining=5) -> SearchContext:
    task = TaskSpec(task_name="t", task_type="binary", data_path="d",
                   target_column="label", eval_metric="roc_auc", description="test")
    profile = DataProfile(n_rows=891, n_features=11)
    node = make_node()
    return SearchContext(
        task=task, data_profile=profile, current_node=node,
        stage=stage, budget_remaining=budget_remaining, budget_used=0
    )


# --- Scheduler tests ---

def test_scheduler_starts_in_warmup():
    scheduler = Scheduler(num_candidates=3, min_warmup_runs=1, max_optimize_iterations=5)
    assert scheduler.stage == "warmup"


def test_scheduler_transitions_to_optimize():
    scheduler = Scheduler(num_candidates=2, min_warmup_runs=1, max_optimize_iterations=5)
    n1 = make_node("n1", metric=0.85)
    n2 = make_node("n2", metric=0.88)
    scheduler.record_warmup_run(n1)
    scheduler.record_warmup_run(n2)
    assert scheduler.should_advance_to_optimization()


def test_scheduler_not_ready_if_missing_candidates():
    scheduler = Scheduler(num_candidates=3, min_warmup_runs=1, max_optimize_iterations=5)
    scheduler.record_warmup_run(make_node("n1", metric=0.85))
    assert not scheduler.should_advance_to_optimization()


def test_scheduler_detects_budget_exhausted():
    scheduler = Scheduler(num_candidates=2, min_warmup_runs=1, max_optimize_iterations=2)
    scheduler.record_warmup_run(make_node("n1", metric=0.85))
    scheduler.record_warmup_run(make_node("n2", metric=0.88))
    scheduler.advance_to_optimization()
    scheduler.record_optimize_run()
    scheduler.record_optimize_run()
    assert scheduler.should_stop()


# --- AcceptReject tests ---

def test_accept_reject_accepts_improvement():
    ar = AcceptReject(higher_is_better=True, min_delta=0.001)
    parent = make_node(metric=0.85)
    child = make_node(metric=0.88)
    assert ar.evaluate(parent, child) is True


def test_accept_reject_rejects_regression():
    ar = AcceptReject(higher_is_better=True, min_delta=0.001)
    parent = make_node(metric=0.88)
    child = make_node(metric=0.85)
    assert ar.evaluate(parent, child) is False


def test_accept_reject_accepts_root_node():
    ar = AcceptReject(higher_is_better=True, min_delta=0.001)
    root = make_node(metric=0.85)  # no parent
    assert ar.evaluate(None, root) is True


# --- ExperimentManager tests ---

def test_manager_returns_select_action_in_warmup():
    mock_llm = MagicMock(spec=LLMBackend)
    manager = ExperimentManager(llm=mock_llm)
    context = make_context(stage="warmup", budget_remaining=5)
    action = manager.next_action(context)
    assert action.action_type == ActionType.SELECT


def test_manager_returns_stop_when_budget_zero():
    mock_llm = MagicMock(spec=LLMBackend)
    manager = ExperimentManager(llm=mock_llm)
    context = make_context(stage="optimize", budget_remaining=0)
    action = manager.next_action(context)
    assert action.action_type == ActionType.STOP
