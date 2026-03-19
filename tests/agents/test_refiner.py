import json
import pytest
from unittest.mock import MagicMock
from src.agents.refiner import RefinerAgent
from src.models.task import ExperimentPlan, TaskSpec, RunConfig
from src.models.results import RunResult, RunEntry, RunDiagnostics


def _make_task():
    return TaskSpec(
        task_name="demo",
        task_type="binary",
        data_path="data/train.csv",
        target_column="Survived",
        eval_metric="roc_auc",
    )


def _make_incumbent(overfitting_gap=0.02, metric=0.87):
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM", "XGB"],
        presets="medium_quality",
        time_limit=120,
        rationale="initial",
    )
    config = RunConfig(
        run_id="run_0001",
        node_id="node_abc",
        autogluon_kwargs={"hyperparameters": {"GBM": {}, "XGB": {}}},
        data_path="data/train.csv",
        output_dir="/tmp",
    )
    result = RunResult(
        run_id="run_0001",
        status="success",
        primary_metric=metric,
        diagnostics_overfitting_gap=overfitting_gap,
    )
    diagnostics = RunDiagnostics(overfitting_gap=overfitting_gap)
    return RunEntry(run_id="run_0001", node_id="node_abc", plan=plan, config=config, result=result, diagnostics=diagnostics)


def _valid_plan_json():
    return json.dumps({
        "eval_metric": "roc_auc",
        "model_families": ["GBM", "XGB", "CAT"],
        "presets": "medium_quality",
        "time_limit": 120,
        "feature_policy": {"exclude_columns": [], "include_columns": []},
        "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0},
        "hyperparameters": None,
        "use_fit_extra": False,
        "rationale": "Added CAT to diversify ensemble",
    })


def test_refine_returns_experiment_plan():
    llm = MagicMock()
    llm.complete.return_value = _valid_plan_json()
    agent = RefinerAgent(llm=llm)
    plan = agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[_make_incumbent()])
    assert isinstance(plan, ExperimentPlan)
    assert "CAT" in plan.model_families


def test_refine_retries_on_invalid_json():
    llm = MagicMock()
    llm.complete.side_effect = ["not json at all", _valid_plan_json()]
    agent = RefinerAgent(llm=llm, max_retries=3)
    plan = agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[])
    assert isinstance(plan, ExperimentPlan)
    assert llm.complete.call_count == 2


def test_refine_strips_markdown_fences():
    llm = MagicMock()
    llm.complete.return_value = f"```json\n{_valid_plan_json()}\n```"
    agent = RefinerAgent(llm=llm)
    plan = agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[])
    assert isinstance(plan, ExperimentPlan)


def test_refine_raises_after_max_retries():
    llm = MagicMock()
    llm.complete.return_value = "garbage"
    agent = RefinerAgent(llm=llm, max_retries=2)
    with pytest.raises(ValueError, match="Failed to get"):
        agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[])


def test_user_message_includes_overfitting_gap():
    llm = MagicMock()
    llm.complete.return_value = _valid_plan_json()
    agent = RefinerAgent(llm=llm)
    agent.refine(incumbent=_make_incumbent(overfitting_gap=0.08), task=_make_task(), prior_runs=[])
    call_args = llm.complete.call_args
    user_msg = call_args[1]["messages"][1].content
    assert "overfitting_gap=0.08" in user_msg


def test_user_message_includes_incumbent_model_families():
    llm = MagicMock()
    llm.complete.return_value = _valid_plan_json()
    agent = RefinerAgent(llm=llm)
    agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[])
    call_args = llm.complete.call_args
    user_msg = call_args[1]["messages"][1].content
    assert "GBM" in user_msg
    assert "XGB" in user_msg
