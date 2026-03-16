import pytest
from datetime import datetime
from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import ModelEntry, RunResult, DataProfile, RunDiagnostics, RunEntry
from src.models.nodes import ExperimentNode, NodeStatus, NodeStage, SearchContext, CaseEntry


def test_task_spec_from_dict():
    spec = TaskSpec(
        task_name="test",
        task_type="binary",
        data_path="data/test.csv",
        target_column="label",
        eval_metric="roc_auc",
        constraints={"max_time_per_run": 120},
        description="Test task"
    )
    assert spec.task_name == "test"
    assert spec.eval_metric == "roc_auc"


def test_experiment_plan_validation():
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM", "XGB"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Baseline test"
    )
    assert plan.time_limit == 120
    assert "GBM" in plan.model_families


def test_run_config_has_node_id():
    config = RunConfig(
        run_id="run_0001",
        node_id="node_abc",
        autogluon_kwargs={"eval_metric": "roc_auc", "time_limit": 120},
        data_path="data/test.csv",
        output_dir="experiments/test/runs/run_0001"
    )
    assert config.node_id == "node_abc"


def test_run_result_failed():
    result = RunResult(
        run_id="run_0001",
        status="failed",
        primary_metric=None,
        leaderboard=[],
        best_model_name=None,
        fit_time_seconds=0.0,
        artifacts_dir="experiments/test/runs/run_0001",
        error="AutoGluon crashed: OOM",
        raw_info={}
    )
    assert result.status == "failed"
    assert result.primary_metric is None


def test_experiment_node_root():
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Root node"
    )
    node = ExperimentNode(
        node_id="node_001",
        parent_id=None,
        children=[],
        edge_label=None,
        stage=NodeStage.WARMUP,
        status=NodeStatus.PENDING,
        plan=plan,
        config=None,
        entry=None,
        depth=0,
        debug_depth=0,
        created_at=datetime.now()
    )
    assert node.parent_id is None
    assert node.depth == 0
    assert node.edge_label is None


def test_experiment_node_child_has_edge_label():
    plan = ExperimentPlan(
        eval_metric="f1_macro",
        model_families=["GBM"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Changed metric"
    )
    node = ExperimentNode(
        node_id="node_002",
        parent_id="node_001",
        children=[],
        edge_label="changed eval_metric from roc_auc to f1_macro due to class imbalance",
        stage=NodeStage.OPTIMIZE,
        status=NodeStatus.PENDING,
        plan=plan,
        config=None,
        entry=None,
        depth=1,
        debug_depth=0,
        created_at=datetime.now()
    )
    assert node.parent_id == "node_001"
    assert "eval_metric" in node.edge_label
