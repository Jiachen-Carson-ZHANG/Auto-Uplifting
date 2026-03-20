import pytest
from unittest.mock import MagicMock, patch
from src.models.task import ExperimentPlan, RunConfig
from src.execution.config_mapper import ConfigMapper
from src.execution.result_parser import ResultParser
from src.models.results import RunResult, ModelEntry


# --- ConfigMapper tests ---

def make_plan(**overrides):
    defaults = dict(
        eval_metric="roc_auc",
        model_families=["GBM", "XGB"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": ["id"], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="test"
    )
    defaults.update(overrides)
    return ExperimentPlan(**defaults)


def test_config_mapper_basic():
    plan = make_plan()
    config = ConfigMapper.to_run_config(
        plan=plan,
        data_path="data/test.csv",
        output_dir="experiments/test/runs/run_0001",
    )
    assert config.autogluon_kwargs["eval_metric"] == "roc_auc"
    assert config.autogluon_kwargs["time_limit"] == 120
    assert config.autogluon_kwargs["presets"] == "medium_quality"


def test_config_mapper_model_families():
    plan = make_plan(model_families=["GBM", "CAT"])
    config = ConfigMapper.to_run_config(
        plan=plan, data_path="d", output_dir="o"
    )
    hp = config.autogluon_kwargs.get("hyperparameters", {})
    assert "GBM" in hp
    assert "CAT" in hp


def test_config_mapper_kfold_validation():
    plan = make_plan(
        validation_policy={"holdout_frac": 0.0, "num_bag_folds": 5}
    )
    config = ConfigMapper.to_run_config(
        plan=plan, data_path="d", output_dir="o"
    )
    assert config.autogluon_kwargs.get("num_bag_folds") == 5


def test_config_mapper_excludes_columns():
    plan = make_plan(
        feature_policy={"exclude_columns": ["PassengerId", "Name"], "include_columns": []}
    )
    config = ConfigMapper.to_run_config(
        plan=plan, data_path="d", output_dir="o"
    )
    assert config.autogluon_kwargs.get("excluded_columns") == ["PassengerId", "Name"]


# --- ResultParser tests ---

def test_result_parser_success():
    mock_predictor = MagicMock()
    mock_predictor.eval_metric = "roc_auc"
    mock_leaderboard = MagicMock()
    mock_leaderboard.itertuples.return_value = [
        MagicMock(model="GBM", score_val=0.87, fit_time=10.0, pred_time=0.1, stack_level=1),
    ]
    mock_predictor.leaderboard.return_value = mock_leaderboard
    mock_predictor.model_best = "GBM"
    mock_predictor.info.return_value = {"version": "1.0"}

    result = ResultParser.from_predictor(
        predictor=mock_predictor,
        run_id="run_0001",
        fit_time=12.5,
        artifacts_dir="experiments/test/runs/run_0001",
        primary_metric_value=0.87
    )
    assert result.status == "success"
    assert result.primary_metric == 0.87
    assert result.best_model_name == "GBM"
    assert len(result.leaderboard) == 1


def test_result_parser_failed():
    result = ResultParser.from_error("AutoGluon OOM")
    assert result.status == "failed"
    assert result.primary_metric is None
    assert "OOM" in result.error
