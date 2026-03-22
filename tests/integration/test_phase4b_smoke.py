"""
Phase 4b end-to-end smoke test.

Tests the full preprocessing pipeline WITHOUT AutoGluon:
  mock LLM → PreprocessingAgent → ValidationHarness → PreprocessingExecutor
  → SessionSummary → PreprocessingStore

Two scenarios:
  1. Happy path   — agent generates valid code → validation passes → entry stored
  2. Failing code — agent generates identity code → validation fails → identity fallback

Uses the real Titanic CSV and the real ValidationHarness subprocess.
ExperimentSession is mocked to skip AutoGluon entirely.
"""
from __future__ import annotations
import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agents.preprocessing_agent import PreprocessingAgent
from src.execution.preprocessing_runner import PreprocessingExecutor
from src.execution.validation_harness import ValidationHarness
from src.memory.preprocessing_store import PreprocessingStore
from src.models.campaign import CampaignConfig, SessionSummary
from src.models.preprocessing import PreprocessingPlan, PreprocessingEntry
from src.models.results import DataProfile
from src.models.task import TaskSpec
from src.orchestration.campaign import CampaignOrchestrator


# ── Shared fixtures ────────────────────────────────────────────────────────────

TITANIC_PATH = "data/titanic_train.csv"

_FAMILY_SIZE_CODE = (
    "def preprocess(df):\n"
    "    import pandas as pd\n"
    "    if 'SibSp' in df.columns and 'Parch' in df.columns:\n"
    "        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n"
    "        df['IsAlone'] = (df['FamilySize'] == 1).astype(int)\n"
    "    if 'Age' in df.columns:\n"
    "        df['Age'] = df['Age'].fillna(df['Age'].median())\n"
    "    return df\n"
)

_TITLE_CODE = (
    "def preprocess(df):\n"
    "    import pandas as pd\n"
    "    if 'Name' in df.columns:\n"
    "        df['Title'] = df['Name'].str.extract(r', (\\w+)\\.', expand=False)\n"
    "        title_map = {'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3}\n"
    "        df['Title'] = df['Title'].map(title_map).fillna(4).astype(int)\n"
    "    return df\n"
)

_IDENTITY_CODE = "def preprocess(df):\n    return df\n"


@pytest.fixture()
def titanic_task() -> TaskSpec:
    return TaskSpec(
        task_name="titanic",
        task_type="binary",
        data_path=TITANIC_PATH,
        target_column="Survived",
        eval_metric="roc_auc",
        description="Titanic survival prediction.",
    )


@pytest.fixture()
def titanic_profile() -> DataProfile:
    df = pd.read_csv(TITANIC_PATH, nrows=100)
    numeric = int(df.select_dtypes(include="number").shape[1])
    return DataProfile(
        n_rows=891,
        n_features=df.shape[1],
        feature_types={"numeric": numeric, "categorical": df.shape[1] - numeric},
    )


def make_mock_llm(*responses: str) -> MagicMock:
    mock = MagicMock()
    mock.complete.side_effect = list(responses)
    return mock


# ── Test 1: Agent generates valid code → ValidationHarness passes ─────────────

def test_agent_generates_valid_titanic_code(titanic_task, titanic_profile):
    """
    Mock LLM inspects 'Name', then generates FamilySize code.
    Verify: strategy=generated, validation_passed=True, code applied to real CSV.
    """
    inspect_resp = json.dumps({
        "thought": "I want to see the Name column for title extraction.",
        "action": "inspect_column",
        "input": "Name",
    })
    generate_resp = json.dumps({
        "thought": "I'll derive FamilySize from SibSp and Parch, and fill Age nulls.",
        "action": "generate_code",
        "input": _FAMILY_SIZE_CODE,
    })
    agent = PreprocessingAgent(llm=make_mock_llm(inspect_resp, generate_resp))

    plan = agent.generate(
        task=titanic_task,
        data_profile=titanic_profile,
        data_path=TITANIC_PATH,
        similar_cases=[],
    )

    assert plan.strategy == "generated", f"Expected generated, got {plan.strategy}"
    assert plan.validation_passed is True, "Expected ValidationHarness to pass"
    assert plan.turns_used >= 1
    assert "FamilySize" in plan.code


def test_agent_title_extraction_passes_validation(titanic_task, titanic_profile):
    """Title extraction from Name column passes all 6 validation checks."""
    generate_resp = json.dumps({
        "thought": "Extract title from Name — encodes gender and social class.",
        "action": "generate_code",
        "input": _TITLE_CODE,
    })
    agent = PreprocessingAgent(llm=make_mock_llm(generate_resp))

    plan = agent.generate(
        task=titanic_task,
        data_profile=titanic_profile,
        data_path=TITANIC_PATH,
        similar_cases=[],
    )

    assert plan.strategy == "generated"
    assert plan.validation_passed is True


# ── Test 2: Identity code fails validation → identity fallback ─────────────────

def test_identity_code_falls_back_to_identity_strategy(titanic_task, titanic_profile):
    """
    If agent generates identity-only code, ValidationHarness fails the diff check,
    and after all 3 turns exhausted, strategy=identity is returned.
    """
    resp = json.dumps({
        "thought": "No transformation needed.",
        "action": "generate_code",
        "input": _IDENTITY_CODE,
    })
    agent = PreprocessingAgent(llm=make_mock_llm(resp, resp, resp))

    plan = agent.generate(
        task=titanic_task,
        data_profile=titanic_profile,
        data_path=TITANIC_PATH,
        similar_cases=[],
    )

    assert plan.strategy == "identity"
    assert plan.validation_passed is False
    assert plan.turns_used == 3


# ── Test 3: PreprocessingExecutor applies generated code to real CSV ───────────

def test_executor_applies_family_size_to_titanic(tmp_path, titanic_task):
    """Generated FamilySize code adds the new column to the real Titanic CSV."""
    plan = PreprocessingPlan(
        strategy="generated",
        code=_FAMILY_SIZE_CODE,
        validation_passed=True,
    )
    out_path = PreprocessingExecutor().run(
        data_path=TITANIC_PATH,
        plan=plan,
        output_dir=str(tmp_path / "prep"),
    )
    result = pd.read_csv(out_path)
    assert "FamilySize" in result.columns
    assert "IsAlone" in result.columns
    assert result["Age"].isna().sum() == 0   # nulls filled


# ── Test 4: PreprocessingStore accumulates entries across agent runs ───────────

def test_preprocessing_store_accumulates_entries(tmp_path, titanic_task, titanic_profile):
    """Two successful agent runs → two entries in the store, get_similar returns them."""
    store_path = str(tmp_path / "bank.jsonl")
    store = PreprocessingStore(store_path)

    for code, summary in [
        (_FAMILY_SIZE_CODE, "FamilySize from SibSp+Parch"),
        (_TITLE_CODE, "Title from Name"),
    ]:
        plan = PreprocessingPlan(strategy="generated", code=code, validation_passed=True)
        entry = PreprocessingEntry(
            entry_id=str(uuid.uuid4())[:8],
            task_type="binary",
            dataset_name="titanic",
            transformation_summary=summary,
            code=code,
            metric_delta=0.01,
        )
        store.add(entry)

    # Reload from disk
    store2 = PreprocessingStore(store_path)
    similar = store2.get_similar("binary", n=3)
    assert len(similar) == 2
    assert similar[0].transformation_summary == "FamilySize from SibSp+Parch"


# ── Test 5: similar_cases are passed to agent and appear in the prompt ─────────

def test_similar_cases_from_store_passed_to_agent(titanic_task, titanic_profile, tmp_path):
    """
    When PreprocessingStore has titanic binary entries, get_similar() returns them
    and they appear in the LLM's initial message.
    """
    store_path = str(tmp_path / "bank.jsonl")
    store = PreprocessingStore(store_path)
    store.add(PreprocessingEntry(
        entry_id="seed-test",
        task_type="binary",
        dataset_name="titanic",
        transformation_summary="extracted title from Name column",
        code=_TITLE_CODE,
    ))

    similar_cases = store.get_similar("binary", n=3)

    generate_resp = json.dumps({
        "thought": "I'll use the title extraction pattern from similar cases.",
        "action": "generate_code",
        "input": _FAMILY_SIZE_CODE,
    })
    mock_llm = make_mock_llm(generate_resp)
    agent = PreprocessingAgent(llm=mock_llm)

    agent.generate(
        task=titanic_task,
        data_profile=titanic_profile,
        data_path=TITANIC_PATH,
        similar_cases=similar_cases,
    )

    # The transformation_summary from the store entry should be in the LLM call
    first_call_messages = mock_llm.complete.call_args_list[0][0][0]
    user_content = next(m.content for m in first_call_messages if m.role == "user")
    assert "extracted title from Name column" in user_content


# ── Test 6: CampaignOrchestrator session-0 always identity ────────────────────

def test_campaign_session_0_always_identity(titanic_task, tmp_path):
    """
    Session 0 should always use identity preprocessing regardless of agent.
    The agent should NOT be called for session 0.
    """
    mock_llm = MagicMock()

    config = CampaignConfig(
        max_sessions=1,
        preprocessing_bank_path=str(tmp_path / "bank.jsonl"),
    )
    orchestrator = CampaignOrchestrator(
        task=titanic_task,
        llm=mock_llm,
        config=config,
        experiments_dir=str(tmp_path / "experiments"),
    )

    plan = orchestrator._preprocessing_plan(session_index=0)
    assert plan.strategy == "identity"
    # Agent (LLM) should NOT have been called for session 0
    mock_llm.complete.assert_not_called()


# ── Test 7: CampaignOrchestrator session 1+ calls agent ───────────────────────

def test_campaign_session_1_calls_preprocessing_agent(titanic_task, tmp_path):
    """
    Session index 1+ should call PreprocessingAgent.
    Mock the LLM to return valid code and verify plan.strategy == 'generated'.
    """
    inspect_resp = json.dumps({
        "thought": "Check Age column.",
        "action": "inspect_column",
        "input": "Age",
    })
    generate_resp = json.dumps({
        "thought": "FamilySize is a useful feature.",
        "action": "generate_code",
        "input": _FAMILY_SIZE_CODE,
    })
    mock_llm = make_mock_llm(inspect_resp, generate_resp)

    config = CampaignConfig(
        max_sessions=2,
        preprocessing_bank_path=str(tmp_path / "bank.jsonl"),
    )
    orchestrator = CampaignOrchestrator(
        task=titanic_task,
        llm=mock_llm,
        config=config,
        experiments_dir=str(tmp_path / "experiments"),
    )

    plan = orchestrator._preprocessing_plan(session_index=1)
    assert plan.strategy == "generated"
    assert plan.validation_passed is True
    assert mock_llm.complete.called


# ── Test 8: _store_preprocessing_entry writes to bank ─────────────────────────

def test_store_preprocessing_entry_persists(titanic_task, tmp_path):
    """After a successful session, _store_preprocessing_entry writes to bank.jsonl."""
    bank_path = str(tmp_path / "bank.jsonl")
    config = CampaignConfig(
        max_sessions=1,
        preprocessing_bank_path=bank_path,
    )
    orchestrator = CampaignOrchestrator(
        task=titanic_task,
        llm=MagicMock(),
        config=config,
        experiments_dir=str(tmp_path / "experiments"),
    )

    plan = PreprocessingPlan(
        strategy="generated",
        code=_FAMILY_SIZE_CODE,
        rationale="FamilySize and IsAlone from SibSp+Parch",
        validation_passed=True,
        turns_used=2,
    )
    orchestrator._store_preprocessing_entry(plan, metric_value=0.83)

    # Reload store and verify the agent-generated entry is there
    # (bank also contains auto-seeded entries from init)
    store = PreprocessingStore(bank_path)
    entries = store.get_all()
    agent_entries = [e for e in entries if e.metric_delta == 0.83]
    assert len(agent_entries) == 1
    assert agent_entries[0].task_type == "binary"
    assert agent_entries[0].dataset_name == "titanic"
