"""
PreprocessingAgent — generates preprocess(df) code via a ReAct tool-use loop.

ReAct loop:
  ┌─────────────────────────────────────────────────────────────────┐
  │  messages = [system, initial_user]                              │
  │  for turn in range(MAX_TURNS=3):                                │
  │    response = llm.complete(messages)                            │
  │    parsed = json.loads(response)                                │
  │    if parsed["action"] == "inspect_column":                     │
  │        result = _inspect_column(parsed["input"])                │
  │        messages += [assistant(response), user(str(result))]     │
  │    elif parsed["action"] == "generate_code":                    │
  │        code = parsed["input"]                                   │
  │        vr = harness.validate(code, ...)                         │
  │        if vr.passed: return PreprocessingPlan(generated, ✓)     │
  │        messages += [assistant(response), user(vr.error)]        │
  │    turn 3 + still inspecting → force generate_code message      │
  │  return PreprocessingPlan(identity)  # all turns exhausted      │
  └─────────────────────────────────────────────────────────────────┘

Wire format (JSON, one object per LLM response):
  {"thought": "...", "action": "inspect_column", "input": "<col_name>"}
  {"thought": "...", "action": "generate_code",  "input": "def preprocess(df):..."}
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import pandas as pd

from src.llm.backend import LLMBackend, Message
from src.models.preprocessing import PreprocessingPlan
from src.execution.validation_harness import ValidationHarness

if TYPE_CHECKING:
    from src.models.task import TaskSpec
    from src.models.results import DataProfile

logger = logging.getLogger(__name__)

_MAX_TURNS = 3
_SAMPLE_ROWS = 10_000   # max rows to read for inspect_column stats
_SAMPLE_VALUES = 5      # non-null sample values to show agent


class PreprocessingAgent:
    """
    Generates a validated preprocess(df) function via a JSON ReAct loop.

    Caches the raw DataFrame on first generate() call. Max 3 turns before
    falling back to identity. Never raises — all errors return identity plan.
    """

    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/preprocessing_agent.md",
        timeout: int = 30,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._harness = ValidationHarness(timeout=timeout)
        self._df: Optional[pd.DataFrame] = None
        self._data_path: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        task: "TaskSpec",
        data_profile: "DataProfile",
        data_path: str,
        similar_cases: list,
    ) -> PreprocessingPlan:
        """
        Run the ReAct loop. Returns a PreprocessingPlan (never raises).
        Falls back to identity on any error or if all turns are exhausted.
        """
        try:
            return self._run(task, data_profile, data_path, similar_cases)
        except Exception as exc:
            logger.warning("PreprocessingAgent: unexpected error, falling back to identity: %s", exc)
            return PreprocessingPlan(strategy="identity")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run(
        self,
        task: "TaskSpec",
        data_profile: "DataProfile",
        data_path: str,
        similar_cases: list,
    ) -> PreprocessingPlan:
        # Load and cache the DataFrame once
        self._data_path = data_path
        self._df = pd.read_csv(data_path, nrows=_SAMPLE_ROWS)

        initial_user = self._build_initial_message(task, data_profile, similar_cases)
        messages: List[Message] = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=initial_user),
        ]

        turns_used = 0
        for turn in range(_MAX_TURNS):
            turns_used = turn + 1

            # On the last turn, force generate_code if agent is still inspecting
            if turn == _MAX_TURNS - 1:
                messages.append(Message(
                    role="user",
                    content=(
                        "You have reached the final turn. "
                        "You must now call generate_code with your best preprocess() function. "
                        "No more inspect_column calls."
                    ),
                ))

            response = self._llm.complete(messages, temperature=0.2)

            try:
                parsed = json.loads(response)
            except json.JSONDecodeError:
                # Strip markdown fences if present and retry parse
                cleaned = response.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                try:
                    parsed = json.loads(cleaned)
                except json.JSONDecodeError:
                    logger.warning("PreprocessingAgent turn %d: could not parse JSON, retrying", turn + 1)
                    messages.append(Message(role="assistant", content=response))
                    messages.append(Message(
                        role="user",
                        content='Your response was not valid JSON. Respond with exactly one JSON object: {"thought": "...", "action": "...", "input": "..."}',
                    ))
                    continue

            action = parsed.get("action", "")
            input_val = parsed.get("input", "")
            thought = parsed.get("thought", "")
            logger.debug("PreprocessingAgent turn %d: action=%s thought=%s", turn + 1, action, thought[:80])

            if action == "inspect_column":
                result = self._inspect_column(input_val)
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(role="user", content=json.dumps(result)))

            elif action == "generate_code":
                code = input_val
                vr = self._harness.validate(code, data_path, task.target_column)
                if vr.passed:
                    logger.info("PreprocessingAgent: validation PASS on turn %d", turn + 1)
                    return PreprocessingPlan(
                        strategy="generated",
                        code=code,
                        rationale=thought,
                        validation_passed=True,
                        turns_used=turns_used,
                    )
                else:
                    logger.debug("PreprocessingAgent turn %d: validation FAIL — %s", turn + 1, vr.error)
                    messages.append(Message(role="assistant", content=response))
                    messages.append(Message(
                        role="user",
                        content=f"Validation failed: {vr.error}\n\nFix the issue and call generate_code again.",
                    ))

            else:
                # Unknown action — log and prompt for correction
                logger.warning("PreprocessingAgent: unknown action %r on turn %d", action, turn + 1)
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(
                    role="user",
                    content=f'Unknown action {action!r}. Use "inspect_column" or "generate_code".',
                ))

        logger.info("PreprocessingAgent: all %d turns exhausted, falling back to identity", _MAX_TURNS)
        return PreprocessingPlan(strategy="identity", turns_used=turns_used)

    def _inspect_column(self, col_name: str) -> dict:
        """Return stats and sample values for col_name. Returns error dict if col missing."""
        if self._df is None:
            return {"error": "DataFrame not loaded"}

        if col_name not in self._df.columns:
            available = sorted(self._df.columns.tolist())
            return {"error": f"column {col_name!r} not found. Available: {available}"}

        col = self._df[col_name]
        non_null = col.dropna()
        sample = non_null.head(_SAMPLE_VALUES).tolist()
        null_pct = round(col.isnull().mean(), 4)

        return {
            "col": col_name,
            "dtype": str(col.dtype),
            "n_unique": int(col.nunique()),
            "null_pct": null_pct,
            "sample_values": sample,
        }

    def _build_initial_message(
        self,
        task: "TaskSpec",
        data_profile: "DataProfile",
        similar_cases: list,
    ) -> str:
        # Build column list from cached DataFrame (most informative) or DataProfile fallback
        if self._df is not None:
            col_list = "\n".join(
                f"  - {col}: {str(self._df[col].dtype)}"
                for col in self._df.columns
            )
        else:
            type_summary = ", ".join(f"{v} {k}" for k, v in data_profile.feature_types.items())
            col_list = f"  (types: {type_summary})"

        lines = [
            f"Task: {task.task_type} — predict '{task.target_column}'",
            f"Dataset: {data_profile.n_rows} rows, {data_profile.n_features} features",
            "",
            "Columns:",
            col_list,
            "",
            "Instructions: Call inspect_column for any columns that look interesting, then generate_code.",
        ]

        if similar_cases:
            lines.append("")
            lines.append("Similar preprocessing patterns from past sessions:")
            for case in similar_cases[:3]:
                lines.append(f"  - {case.transformation_summary}")

        return "\n".join(lines)
