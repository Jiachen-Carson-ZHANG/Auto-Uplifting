# src/agents/ideator.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict
from src.llm.backend import LLMBackend, Message
from src.models.task import TaskSpec
from src.models.results import DataProfile
from src.models.nodes import CaseEntry


class IdeatorAgent:
    """
    Generates initial experiment hypotheses from task + data profile + similar past cases.
    Returns List[Dict] with 'hypothesis' and 'rationale' keys,
    compatible with session.create_candidate_nodes().
    """

    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/ideator.md",
        num_hypotheses: int = 3,
        temperature: float = 0.5,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._num_hypotheses = num_hypotheses
        self._temperature = temperature

    def ideate(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        similar_cases: List[CaseEntry],
    ) -> List[Dict[str, str]]:
        user_msg = self._build_user_message(task, data_profile, similar_cases)
        response = self._llm.complete(
            messages=[
                Message(role="system", content=self._system_prompt),
                Message(role="user", content=user_msg),
            ],
            temperature=self._temperature,
        )
        raw = json.loads(response)
        return [{"hypothesis": h["hypothesis"], "rationale": h["rationale"]} for h in raw]

    def _build_user_message(
        self,
        task: TaskSpec,
        profile: DataProfile,
        similar_cases: List[CaseEntry],
    ) -> str:
        cases_text = ""
        if similar_cases:
            lines = []
            for c in similar_cases:
                lines.append(
                    f"- Past case ({c.task_traits.task_type}, {c.task_traits.n_rows_bucket} rows): "
                    f"best={c.what_worked.best_metric:.3f} | "
                    f"worked: {'; '.join(c.what_worked.key_decisions[:2])} | "
                    f"failed: {'; '.join(c.what_failed.failure_patterns[:1])}"
                )
            cases_text = "\n## Similar Past Cases\n" + "\n".join(lines)

        return (
            f"## Task\n"
            f"Name: {task.task_name} | Type: {task.task_type}\n"
            f"Target: {task.target_column} | Metric: {task.eval_metric}\n"
            f"Description: {task.description}\n\n"
            f"## Data Profile\n"
            f"{profile.summary}\n"
            f"Rows: {profile.n_rows} | Features: {profile.n_features}\n"
            f"Class balance ratio: {profile.class_balance_ratio:.2f}\n"
            f"Missing rate: {profile.missing_rate:.2%}\n"
            f"Feature types: {profile.feature_types}\n"
            f"{cases_text}\n\n"
            f"Generate exactly {self._num_hypotheses} diverse hypotheses as a JSON array."
        )
