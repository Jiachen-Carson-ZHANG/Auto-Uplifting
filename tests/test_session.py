import pytest
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime
from src.session import ExperimentSession
from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import DataProfile, RunResult, RunDiagnostics, RunEntry
from src.models.nodes import NodeStage, NodeStatus
from src.llm.backend import LLMBackend


def make_task(tmp_path) -> TaskSpec:
    # Create minimal CSV for data profiling
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("feature1,feature2,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
    return TaskSpec(
        task_name="test", task_type="binary",
        data_path=str(csv_path), target_column="label",
        eval_metric="roc_auc", description="Test task"
    )


def make_valid_plan_json(metric="roc_auc") -> str:
    return json.dumps({
        "eval_metric": metric, "model_families": ["GBM"],
        "presets": "medium_quality", "time_limit": 60,
        "feature_policy": {"exclude_columns": [], "include_columns": []},
        "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0},
        "hyperparameters": None, "use_fit_extra": False,
        "rationale": "Test plan"
    })


def test_session_initializes(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    session = ExperimentSession(
        task=task, llm=mock_llm,
        experiments_dir=str(tmp_path / "experiments"),
        num_candidates=2, max_optimize_iterations=2
    )
    assert session.task.task_name == "test"
    assert session.tree is not None
    assert session.run_store is not None


def test_session_profiles_data(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    session = ExperimentSession(
        task=task, llm=mock_llm,
        experiments_dir=str(tmp_path / "experiments"),
        num_candidates=2, max_optimize_iterations=2
    )
    profile = session.profile_data()
    assert profile.n_rows == 4
    assert profile.n_features == 2  # excludes target
    assert 0.0 <= profile.class_balance_ratio <= 1.0


def test_session_creates_root_nodes_from_hypotheses(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    mock_llm.complete.return_value = make_valid_plan_json()
    task = make_task(tmp_path)
    session = ExperimentSession(
        task=task, llm=mock_llm,
        experiments_dir=str(tmp_path / "experiments"),
        num_candidates=2, max_optimize_iterations=2
    )
    hypotheses = [
        {"hypothesis": "Try GBM baseline", "rationale": "Good default"},
        {"hypothesis": "Try RF for robustness", "rationale": "Less overfitting"},
    ]
    profile = session.profile_data()
    nodes = session.create_candidate_nodes(hypotheses=hypotheses, data_profile=profile)
    assert len(nodes) == 2
    assert all(n.parent_id is None for n in nodes)
    assert all(n.stage == NodeStage.WARMUP for n in nodes)


def test_session_run_store_records_entry(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    session = ExperimentSession(
        task=task, llm=mock_llm,
        experiments_dir=str(tmp_path / "experiments"),
        num_candidates=2, max_optimize_iterations=2
    )
    # Build a minimal RunEntry and record it
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=60, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale="test"
    )
    config = RunConfig(run_id="r001", node_id="n001",
                      autogluon_kwargs={}, data_path="d", output_dir="o")
    result = RunResult(run_id="r001", status="success", primary_metric=0.85,
                      leaderboard=[], best_model_name="GBM", fit_time_seconds=5.0,
                      artifacts_dir="o")
    entry = RunEntry(run_id="r001", node_id="n001", config=config, result=result,
                    diagnostics=RunDiagnostics(), timestamp=datetime.now())
    session.run_store.append(entry)
    assert len(session.run_store.get_history()) == 1
    assert session.run_store.get_incumbent(higher_is_better=True).run_id == "r001"
