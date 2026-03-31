"""Tests for src/agents/feature_engineer.py — FeatureEngineeringAgent."""
import json
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.agents.feature_engineer import (
    FeatureEngineeringAgent,
    _blocked_decision,
    _blocked_audit,
    _blocked_result,
    _failed_result,
    _strip_fences,
)
from src.models.feature_engineering import (
    FeatureAuditVerdict,
    FeatureDecision,
    FeatureExecutionResult,
    FeatureHistoryEntry,
    TemplateFeatureSpec,
)
from src.models.results import DataProfile
from src.models.task import TaskSpec


# ── Helpers ───────────────────────────────────────────────────────────

def _make_task():
    return TaskSpec(
        task_name="churn",
        task_type="binary",
        data_path="data/churn.csv",
        target_column="churned",
        eval_metric="roc_auc",
    )


def _make_profile():
    return DataProfile(n_rows=1000, n_features=20)


def _valid_decision_json(**overrides):
    d = {
        "status": "proposed",
        "action": "add",
        "reasoning": "Add RFM recency feature",
        "feature_spec": {
            "spec_type": "template",
            "template_name": "rfm_recency",
            "params": {"entity_key": "customer_id", "time_col": "date"},
        },
        "expected_impact": "moderate",
    }
    d.update(overrides)
    return json.dumps(d)


def _valid_audit_json(verdict="pass"):
    return json.dumps({"verdict": verdict, "reasons": [], "required_fixes": []})


# ── Helper function tests ─────────────────────────────────────────────

class TestHelpers:
    def test_blocked_decision(self):
        d = _blocked_decision("test reason")
        assert d.status == "blocked"
        assert d.action == "blocked"

    def test_blocked_audit(self):
        a = _blocked_audit("test reason")
        assert a.verdict == "block"
        assert "test reason" in a.reasons

    def test_blocked_result(self):
        r = _blocked_result("test reason")
        assert r.status == "blocked"

    def test_failed_result(self):
        r = _failed_result("test reason")
        assert r.status == "failed"

    def test_strip_fences_json(self):
        raw = '```json\n{"key": "val"}\n```'
        assert _strip_fences(raw) == '{"key": "val"}'

    def test_strip_fences_no_fences(self):
        raw = '{"key": "val"}'
        assert _strip_fences(raw) == '{"key": "val"}'

    def test_strip_fences_plain(self):
        raw = '```\nhello\n```'
        assert _strip_fences(raw) == "hello"


# ── Agent tests ───────────────────────────────────────────────────────

class TestFeatureEngineeringAgent:
    @pytest.fixture
    def mock_llm(self):
        return MagicMock()

    @pytest.fixture
    def agent(self, mock_llm, tmp_path):
        # Write minimal prompt files
        prompt_path = tmp_path / "decision.md"
        prompt_path.write_text("You are a feature engineer.")
        leakage_path = tmp_path / "leakage.md"
        leakage_path.write_text("You audit for leakage.")

        return FeatureEngineeringAgent(
            llm=mock_llm,
            prompt_path=str(prompt_path),
            leakage_prompt_path=str(leakage_path),
            max_retries=2,
        )

    def test_successful_proposal(self, agent, mock_llm):
        import pandas as pd
        mock_llm.complete = MagicMock(side_effect=[
            _valid_decision_json(),
            _valid_audit_json("pass"),
        ])

        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "date": ["2025-01-01", "2025-06-01"],
            "churned": [0, 1],
        })

        decision, audit, result, result_df = agent.propose_and_execute(
            task=_make_task(), data_profile=_make_profile(),
            df=df, leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            budget_remaining=10, budget_used=0,
        )
        assert decision.status == "proposed"
        assert audit.verdict == "pass"
        assert result.status == "success"

    def test_leakage_block(self, agent, mock_llm):
        import pandas as pd
        mock_llm.complete = MagicMock(side_effect=[
            _valid_decision_json(),
            _valid_audit_json("block"),
        ])

        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "date": ["2025-01-01", "2025-06-01"],
            "churned": [0, 1],
        })

        decision, audit, result, result_df = agent.propose_and_execute(
            task=_make_task(), data_profile=_make_profile(),
            df=df, leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            budget_remaining=10, budget_used=0,
        )
        assert audit.verdict == "block"
        assert result.status == "blocked"

    def test_codegen_escalation_blocked_phase1(self, agent, mock_llm):
        import pandas as pd
        mock_llm.complete = MagicMock(return_value=_valid_decision_json(
            action="escalate_codegen",
            feature_spec={
                "spec_type": "codegen",
                "description": "custom logic",
                "reason_bounded_insufficient": "need sequence",
            },
        ))

        df = pd.DataFrame({"customer_id": ["A"], "churned": [0]})
        decision, audit, result, result_df = agent.propose_and_execute(
            task=_make_task(), data_profile=_make_profile(),
            df=df, leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            budget_remaining=10, budget_used=0,
        )
        assert decision.action == "escalate_codegen"
        assert result.status == "blocked"
        assert "Phase 1" in result.failure_reason

    def test_decision_parse_failure_retries(self, agent, mock_llm):
        import pandas as pd
        mock_llm.complete = MagicMock(side_effect=[
            "not valid json",
            _valid_decision_json(),
            _valid_audit_json("pass"),
        ])

        df = pd.DataFrame({
            "customer_id": ["A", "B"],
            "date": ["2025-01-01", "2025-06-01"],
            "churned": [0, 1],
        })

        decision, audit, result, result_df = agent.propose_and_execute(
            task=_make_task(), data_profile=_make_profile(),
            df=df, leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            budget_remaining=10, budget_used=0,
        )
        assert decision.status == "proposed"
        assert mock_llm.complete.call_count >= 2

    def test_all_retries_exhausted(self, agent, mock_llm):
        import pandas as pd
        mock_llm.complete = MagicMock(return_value="garbage")

        df = pd.DataFrame({"customer_id": ["A"], "churned": [0]})
        decision, audit, result, result_df = agent.propose_and_execute(
            task=_make_task(), data_profile=_make_profile(),
            df=df, leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            budget_remaining=10, budget_used=0,
        )
        assert decision.status == "blocked"

    def test_never_raises(self, agent, mock_llm):
        import pandas as pd
        mock_llm.complete = MagicMock(side_effect=RuntimeError("LLM down"))

        df = pd.DataFrame({"customer_id": ["A"], "churned": [0]})
        # Should not raise
        decision, audit, result, result_df = agent.propose_and_execute(
            task=_make_task(), data_profile=_make_profile(),
            df=df, leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            budget_remaining=10, budget_used=0,
        )
        assert decision.status == "blocked"
        assert result.status == "failed"
