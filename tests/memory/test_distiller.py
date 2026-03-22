# tests/memory/test_distiller.py
import pytest
from unittest.mock import MagicMock
from src.memory.distiller import Distiller
from src.models.nodes import CaseEntry
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, ExperimentRun, RunResult, RunDiagnostics, RunConfig


def _make_run_entry(run_id: str, metric: float, families: list) -> ExperimentRun:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=families, presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    config = RunConfig(
        autogluon_kwargs={"eval_metric": "roc_auc", "presets": "medium_quality", "time_limit": 120, "hyperparameters": {f: {} for f in families}},
        data_path="data/train.csv",
        output_dir=f"experiments/runs/{run_id}",
    )
    result = RunResult(
        status="success",
        primary_metric=metric,
        best_model_name="WeightedEnsemble_L2",
        fit_time_seconds=10.0,
    )
    return ExperimentRun(
        run_id=run_id, node_id="n1", config=config, result=result,
        diagnostics=RunDiagnostics(), agent_rationale="test rationale",
    )


LLM_RESPONSE = '''{
  "what_worked": {
    "key_decisions": ["Run 2: switching to f1_macro improved +0.08"],
    "important_features": ["Age", "Fare"],
    "effective_presets": "medium_quality"
  },
  "what_failed": {
    "failed_approaches": ["GBM alone scored 0.83"],
    "failure_patterns": ["Single model families underperform ensembles on small tabular data"]
  },
  "trajectory": {
    "turning_points": ["Run 2: adding CAT to ensemble was the key improvement"]
  }
}'''


def test_distill_returns_case_entry():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    task = TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test task", constraints={},
    )
    profile = DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.6, missing_rate=0.05)
    runs = [
        _make_run_entry("run_0001", 0.83, ["GBM"]),
        _make_run_entry("run_0002", 0.87, ["GBM", "CAT"]),
    ]

    distiller = Distiller(llm=mock_llm)
    case = distiller.distill(task=task, data_profile=profile, run_history=runs)

    assert isinstance(case, CaseEntry)
    assert case.what_worked.best_metric == 0.87
    assert case.trajectory.n_runs == 2
    assert "Age" in case.what_worked.important_features
    assert mock_llm.complete.called


def test_distill_computes_traits_locally():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    task = TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test task", constraints={},
    )
    profile = DataProfile(n_rows=200, n_features=5, class_balance_ratio=0.5, missing_rate=0.0)
    runs = [_make_run_entry("run_0001", 0.83, ["GBM"])]

    distiller = Distiller(llm=mock_llm)
    case = distiller.distill(task=task, data_profile=profile, run_history=runs)

    assert case.task_traits.task_type == "binary"
    assert case.task_traits.n_rows_bucket == "small"   # 200 < 1000
    assert case.task_traits.class_balance == "moderate"  # 0.5 is in [0.4, 0.8) → moderate


def _make_task_and_profile():
    task = TaskSpec(
        task_name="test", task_type="binary", data_path="data/train.csv",
        target_column="label", eval_metric="roc_auc",
        description="test task", constraints={},
    )
    profile = DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.6, missing_rate=0.05)
    runs = [_make_run_entry("run_0001", 0.87, ["GBM", "CAT"])]
    return task, profile, runs


def test_distill_without_embed_backend_sets_embedding_none():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE
    task, profile, runs = _make_task_and_profile()

    case = Distiller(llm=mock_llm).distill(task, profile, runs)
    assert case.embedding is None
    assert case.description_for_embedding != ""  # description is still built


def test_distill_with_embed_backend_calls_embed():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE
    mock_embed = MagicMock()
    mock_embed.embed.return_value = [0.1, 0.2, 0.3]
    task, profile, runs = _make_task_and_profile()

    case = Distiller(llm=mock_llm, embed_backend=mock_embed).distill(task, profile, runs)
    mock_embed.embed.assert_called_once()
    assert case.embedding == [0.1, 0.2, 0.3]


def test_distill_embed_failure_returns_none_embedding():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE
    mock_embed = MagicMock()
    mock_embed.embed.side_effect = RuntimeError("API down")
    task, profile, runs = _make_task_and_profile()

    case = Distiller(llm=mock_llm, embed_backend=mock_embed).distill(task, profile, runs)
    assert case.embedding is None  # graceful degradation
