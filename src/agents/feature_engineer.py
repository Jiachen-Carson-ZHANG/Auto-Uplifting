"""
FeatureEngineeringAgent — internal pipeline: decision → leakage audit → execute.

Externally: one call to propose_and_execute() returns a triple of
(FeatureDecision, FeatureAuditVerdict, FeatureExecutionResult).

Internally: up to 3 LLM calls (decision, leakage audit, and later codegen).
All are implementation details of this single node.

Never raises — returns safe defaults on exhaustion.
Follow src/agents/refiner.py for retry pattern, src/agents/preprocessing_agent.py for never-raises.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.features.context_builder import FeatureContextBuilder
from src.features.executor import BoundedExecutor
from src.features.registry import TemplateRegistry, build_default_registry
from src.features.validator import FeatureValidator
from src.llm.backend import LLMBackend, Message
from src.models.feature_engineering import (
    FeatureAuditVerdict,
    FeatureDecision,
    FeatureExecutionResult,
    FeatureHistoryEntry,
)
from src.models.results import DataProfile
from src.models.task import TaskSpec

logger = logging.getLogger(__name__)


def _blocked_decision(reason: str) -> FeatureDecision:
    return FeatureDecision(
        status="blocked", action="blocked", reasoning=reason
    )


def _blocked_audit(reason: str) -> FeatureAuditVerdict:
    return FeatureAuditVerdict(verdict="block", reasons=[reason])


def _blocked_result(reason: str) -> FeatureExecutionResult:
    return FeatureExecutionResult(status="blocked", failure_reason=reason)


def _failed_result(reason: str) -> FeatureExecutionResult:
    return FeatureExecutionResult(status="failed", failure_reason=reason)


def _strip_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.split("\n")[1:])
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    return cleaned


class FeatureEngineeringAgent:
    """
    Proposes, audits, and executes one feature engineering action per call.

    Pipeline:
      1. _decision_call → FeatureDecision
      2. _leakage_audit_call → FeatureAuditVerdict
      3. _execute_bounded → FeatureExecutionResult

    Phase 2 will add _codegen_call and _codegen_guardrail_call.
    """

    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/feature_engineering/feature_engineer_full.md",
        leakage_prompt_path: str = "prompts/feature_engineering/feature_leakage_audit.md",
        registry: Optional[TemplateRegistry] = None,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._leakage_prompt = Path(leakage_prompt_path).read_text()
        self._executor = BoundedExecutor(registry or build_default_registry())
        self._validator = FeatureValidator()
        self._context_builder = FeatureContextBuilder()
        self._max_retries = max_retries

    def propose_and_execute(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        df: pd.DataFrame,
        leaderboard: List[dict],
        feature_importances: Dict[str, float],
        history: List[FeatureHistoryEntry],
        incumbent_metric: Optional[float],
        budget_remaining: int,
        budget_used: int,
    ) -> Tuple[FeatureDecision, FeatureAuditVerdict, FeatureExecutionResult, Optional[pd.DataFrame]]:
        """
        Propose one feature engineering action, audit it, and execute it.

        Returns (decision, audit, exec_result, result_df).
        result_df is the modified DataFrame on success, None on failure.
        Never raises. Returns safe defaults on any failure.
        """
        try:
            # 1. Build context
            context = self._context_builder.build(
                task=task,
                data_profile=data_profile,
                leaderboard=leaderboard,
                feature_importances=feature_importances,
                history=history,
                incumbent_metric=incumbent_metric,
                available_templates=self._executor._registry.list_templates(),
                budget_remaining=budget_remaining,
                budget_used=budget_used,
            )

            # 2. Decision call
            decision = self._decision_call(context)

            # 3. Early returns
            if decision.status in ("blocked", "skip") or decision.action == "blocked":
                return (
                    decision,
                    FeatureAuditVerdict(verdict="pass"),
                    _blocked_result("Decision is blocked or skip."),
                    None,
                )

            if decision.action == "escalate_codegen":
                return (
                    decision,
                    FeatureAuditVerdict(verdict="pass"),
                    _blocked_result("Codegen not available in Phase 1."),
                    None,
                )

            # 4. Leakage audit
            audit = self._leakage_audit_call(decision, data_profile)
            if audit.verdict == "block":
                return (
                    decision,
                    audit,
                    _blocked_result(f"Leakage audit blocked: {audit.reasons}"),
                    None,
                )

            # 5. Execute bounded
            result_df, result = self._executor.execute(df, decision)

            # 6. Validate if successful
            if result.status == "success" and result_df is not None:
                warnings = self._validator.validate_result(
                    df, result_df, task.target_column
                )
                if warnings:
                    result.warnings.extend(warnings)
                    logger.info("Validation warnings: %s", warnings)

            return decision, audit, result, result_df

        except Exception as exc:
            logger.error("FeatureEngineeringAgent failed: %s", exc, exc_info=True)
            return (
                _blocked_decision(f"Agent error: {exc}"),
                _blocked_audit(f"Agent error: {exc}"),
                _failed_result(f"Agent error: {exc}"),
                None,
            )

    # ── Private LLM calls ──────────────────────────────────────────

    def _decision_call(self, context: str) -> FeatureDecision:
        """LLM call to get a FeatureDecision. Retries on JSON parse failure."""
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=context),
        ]

        for attempt in range(self._max_retries):
            response = self._llm.complete(messages=messages, temperature=0.3)
            cleaned = _strip_fences(response)
            try:
                return FeatureDecision.model_validate_json(cleaned)
            except Exception as e:
                logger.warning(
                    "Decision parse failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, e,
                )
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(
                    role="user",
                    content=(
                        f"Your response was not valid JSON matching the FeatureDecision schema. "
                        f"Error: {e}. Respond with ONLY the JSON object, no markdown fences."
                    ),
                ))

        return _blocked_decision("Failed to parse FeatureDecision after retries.")

    def _leakage_audit_call(
        self, decision: FeatureDecision, data_profile: DataProfile
    ) -> FeatureAuditVerdict:
        """LLM call to audit a feature decision for leakage."""
        user_content = (
            f"## Proposed Feature\n{decision.model_dump_json(indent=2)}\n\n"
            f"## Data Profile\n"
            f"- Rows: {data_profile.n_rows}\n"
            f"- Columns: {data_profile.n_features}\n"
        )

        messages = [
            Message(role="system", content=self._leakage_prompt),
            Message(role="user", content=user_content),
        ]

        for attempt in range(self._max_retries):
            response = self._llm.complete(messages=messages, temperature=0.1)
            cleaned = _strip_fences(response)
            try:
                return FeatureAuditVerdict.model_validate_json(cleaned)
            except Exception as e:
                logger.warning(
                    "Audit parse failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries, e,
                )
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(
                    role="user",
                    content=(
                        f"Your response was not valid JSON matching the FeatureAuditVerdict schema. "
                        f"Error: {e}. Respond with ONLY the JSON object."
                    ),
                ))

        # Fail closed on audit parse failure
        return _blocked_audit("Failed to parse audit verdict after retries.")
