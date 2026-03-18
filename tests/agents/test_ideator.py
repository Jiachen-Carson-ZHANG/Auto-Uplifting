# tests/agents/test_ideator.py
import pytest
from unittest.mock import MagicMock
from src.agents.ideator import IdeatorAgent
from src.models.task import TaskSpec
from src.models.results import DataProfile


LLM_RESPONSE = '''[
  {"id": "h1", "model_focus": "GBM", "metric_focus": "roc_auc",
   "hypothesis": "Start with GBM baseline.", "rationale": "Reliable default."},
  {"id": "h2", "model_focus": "RF", "metric_focus": "f1_macro",
   "hypothesis": "Try RF with f1_macro for class imbalance.",
   "rationale": "class_balance_ratio=0.62 suggests mild imbalance."},
  {"id": "h3", "model_focus": "GBM+XGB", "metric_focus": "accuracy",
   "hypothesis": "Diverse boosting ensemble.", "rationale": "Variety improves ensemble."}
]'''


def _task():
    return TaskSpec(
        task_name="titanic", task_type="binary", data_path="data/train.csv",
        target_column="Survived", eval_metric="roc_auc",
        description="Predict survival.", constraints={},
    )

def _profile():
    return DataProfile(n_rows=891, n_features=10, class_balance_ratio=0.62, missing_rate=0.08)


def test_ideate_returns_hypothesis_list():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    agent = IdeatorAgent(llm=mock_llm)
    hypotheses = agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[])

    assert len(hypotheses) == 3
    assert all("hypothesis" in h for h in hypotheses)
    assert all("rationale" in h for h in hypotheses)


def test_ideate_calls_llm_once():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    agent = IdeatorAgent(llm=mock_llm)
    agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[])

    assert mock_llm.complete.call_count == 1


def test_ideate_respects_num_hypotheses():
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE  # returns 3

    agent = IdeatorAgent(llm=mock_llm, num_hypotheses=3)
    hypotheses = agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[])
    assert len(hypotheses) == 3


def test_ideate_includes_similar_cases_in_message():
    """When similar_cases provided, the user message must reference them."""
    from src.models.nodes import CaseEntry, TaskTraits, WhatWorked, WhatFailed, SessionTrajectory
    from src.models.task import ExperimentPlan

    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={}, validation_policy={"holdout_frac": 0.2},
    )
    case = CaseEntry(
        case_id="c1",
        task_traits=TaskTraits(task_type="binary", n_rows_bucket="medium",
                               n_features_bucket="medium", class_balance="balanced"),
        what_worked=WhatWorked(best_config=plan, best_metric=0.85,
                               key_decisions=["GBM+CAT improved by 0.04"]),
        what_failed=WhatFailed(failure_patterns=["NN_TORCH overfit on small data"]),
        trajectory=SessionTrajectory(n_runs=3),
    )
    mock_llm = MagicMock()
    mock_llm.complete.return_value = LLM_RESPONSE

    agent = IdeatorAgent(llm=mock_llm)
    agent.ideate(task=_task(), data_profile=_profile(), similar_cases=[case])

    call_args = mock_llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    user_content = next(m.content for m in messages if m.role == "user")
    assert "Similar Past Cases" in user_content
    assert "GBM+CAT improved" in user_content
