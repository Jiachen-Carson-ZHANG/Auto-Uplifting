"""Tests for PreprocessingAgent — mocked LLM, real ValidationHarness."""
from __future__ import annotations
import json
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from src.agents.preprocessing_agent import PreprocessingAgent
from src.llm.backend import LLMBackend, Message
from src.models.preprocessing import PreprocessingPlan
from src.models.results import DataProfile
from src.models.task import TaskSpec


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def titanic_csv(tmp_path):
    df = pd.DataFrame({
        "PassengerId": range(1, 11),
        "Survived": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath",
            "Allen, Mr. William Henry",
            "Moran, Mr. James",
            "McCarthy, Mr. Timothy J",
            "Palsson, Master. Gosta Leonard",
            "Johnson, Mrs. Oscar W",
            "Nasser, Mrs. Nicholas",
        ],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0, 14.0],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    })
    path = tmp_path / "titanic.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture()
def task():
    return TaskSpec(
        task_name="titanic",
        task_type="binary",
        data_path="irrelevant",
        target_column="Survived",
        eval_metric="roc_auc",
    )


@pytest.fixture()
def data_profile():
    return DataProfile(
        n_rows=10,
        n_features=7,
        feature_types={"numeric": 5, "categorical": 2},
    )


def make_mock_llm(*responses):
    """Return a mock LLMBackend that yields responses in order."""
    mock = MagicMock(spec=LLMBackend)
    mock.complete.side_effect = list(responses)
    return mock


_VALID_PREPROCESS_CODE = (
    "def preprocess(df):\n"
    "    import pandas as pd\n"
    "    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n"
    "    return df\n"
)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_happy_path_generates_valid_plan(titanic_csv, task, data_profile):
    """Agent inspects one column, then generates valid code → strategy=generated."""
    inspect_response = json.dumps({
        "thought": "I want to see the Name column.",
        "action": "inspect_column",
        "input": "Name",
    })
    generate_response = json.dumps({
        "thought": "I'll derive FamilySize from SibSp and Parch.",
        "action": "generate_code",
        "input": _VALID_PREPROCESS_CODE,
    })
    mock_llm = make_mock_llm(inspect_response, generate_response)
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, titanic_csv, similar_cases=[])

    assert plan.strategy == "generated"
    assert plan.validation_passed is True
    assert plan.turns_used >= 1
    assert plan.code == _VALID_PREPROCESS_CODE


def test_all_attempts_fail_returns_identity(titanic_csv, task, data_profile):
    """All 3 turns exhaust with bad code → strategy=identity."""
    bad_code = "def preprocess(df):\n    return df  # identity"
    bad_response = json.dumps({
        "thought": "Here's my code.",
        "action": "generate_code",
        "input": bad_code,
    })
    # 3 turns of forced generate (diff check fails each time)
    mock_llm = make_mock_llm(bad_response, bad_response, bad_response)
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, titanic_csv, similar_cases=[])

    assert plan.strategy == "identity"
    assert plan.validation_passed is False


def test_malformed_json_retries_and_succeeds(titanic_csv, task, data_profile):
    """Turn 1 is malformed JSON → agent retries → turn 2 succeeds."""
    bad_json = "not valid json at all"
    good_response = json.dumps({
        "thought": "Generate FamilySize.",
        "action": "generate_code",
        "input": _VALID_PREPROCESS_CODE,
    })
    mock_llm = make_mock_llm(bad_json, good_response, good_response)
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, titanic_csv, similar_cases=[])

    assert plan.strategy == "generated"
    assert plan.validation_passed is True


def test_inspect_missing_column_returns_error_dict(titanic_csv, task, data_profile):
    """inspect_column with non-existent column → returns error dict, agent continues."""
    inspect_bad = json.dumps({
        "thought": "Inspect a column that doesn't exist.",
        "action": "inspect_column",
        "input": "NonExistentCol",
    })
    generate_response = json.dumps({
        "thought": "Ok, I'll use SibSp and Parch instead.",
        "action": "generate_code",
        "input": _VALID_PREPROCESS_CODE,
    })
    mock_llm = make_mock_llm(inspect_bad, generate_response)
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, titanic_csv, similar_cases=[])

    # Agent should still succeed — error dict is fed back, not an exception
    assert plan.strategy == "generated"
    assert plan.validation_passed is True


def test_turns_used_populated(titanic_csv, task, data_profile):
    generate_response = json.dumps({
        "thought": "Generate code.",
        "action": "generate_code",
        "input": _VALID_PREPROCESS_CODE,
    })
    mock_llm = make_mock_llm(generate_response)
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, titanic_csv, similar_cases=[])

    assert plan.turns_used >= 1


def test_validation_passed_populated_on_failure(titanic_csv, task, data_profile):
    """Identity code fails validation → validation_passed=False."""
    bad_code = "def preprocess(df):\n    return df"
    responses = [json.dumps({"thought": ".", "action": "generate_code", "input": bad_code})] * 3
    mock_llm = make_mock_llm(*responses)
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, titanic_csv, similar_cases=[])

    assert plan.validation_passed is False
    assert plan.turns_used > 0


def test_similar_cases_included_in_initial_message(titanic_csv, task, data_profile):
    """Similar cases appear in the initial user message to the LLM."""
    from unittest.mock import MagicMock
    similar = [MagicMock(transformation_summary="extracted title from Name")]

    generate_response = json.dumps({
        "thought": "Use similar case patterns.",
        "action": "generate_code",
        "input": _VALID_PREPROCESS_CODE,
    })
    mock_llm = make_mock_llm(generate_response)
    agent = PreprocessingAgent(llm=mock_llm)

    agent.generate(task, data_profile, titanic_csv, similar_cases=similar)

    # Verify the initial user message contains the similar case summary
    first_call_messages = mock_llm.complete.call_args_list[0][0][0]
    user_content = next(m.content for m in first_call_messages if m.role == "user")
    assert "extracted title from Name" in user_content


def test_generate_never_raises_on_bad_data_path(task, data_profile):
    """generate() with a non-existent CSV path returns identity, never raises."""
    mock_llm = make_mock_llm(json.dumps({
        "thought": "Generate code.",
        "action": "generate_code",
        "input": "def preprocess(df):\n    return df",
    }))
    agent = PreprocessingAgent(llm=mock_llm)

    plan = agent.generate(task, data_profile, "/nonexistent/path.csv", similar_cases=[])

    assert plan.strategy == "identity"
