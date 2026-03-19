from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from src.llm.backend import LLMBackend, Message
from src.models.task import ExperimentPlan, TaskSpec
from src.models.results import RunEntry


class RefinerAgent:
    """
    Proposes a targeted one-step refinement of the incumbent config.
    Receives full incumbent state (config + leaderboard + diagnostics) + history.
    Returns ExperimentPlan. Retries on invalid JSON, strips markdown fences.
    """

    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/refiner.md",
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._temperature = temperature
        self._max_retries = max_retries

    def refine(
        self,
        incumbent: RunEntry,
        task: TaskSpec,
        prior_runs: List[RunEntry],
    ) -> ExperimentPlan:
        user_msg = self._build_user_message(incumbent, task, prior_runs)
        messages = [
            Message(role="system", content=self._system_prompt),
            Message(role="user", content=user_msg),
        ]
        last_error: Optional[Exception] = None
        for _ in range(self._max_retries):
            response = self._llm.complete(messages=messages, temperature=self._temperature)
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(cleaned.split("\n")[1:])
                cleaned = cleaned.rsplit("```", 1)[0].strip()
            try:
                return ExperimentPlan.model_validate_json(cleaned)
            except Exception as e:
                last_error = e
                messages.append(Message(role="assistant", content=response))
                messages.append(Message(
                    role="user",
                    content=(
                        f"Your response was not valid JSON matching the ExperimentPlan schema. "
                        f"Error: {e}. Respond with ONLY the JSON object, no markdown fences."
                    ),
                ))
        raise ValueError(
            f"Failed to get valid ExperimentPlan after {self._max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _build_user_message(
        self,
        incumbent: RunEntry,
        task: TaskSpec,
        prior_runs: List[RunEntry],
    ) -> str:
        families = incumbent.plan.model_families
        overfitting_gap = incumbent.diagnostics.overfitting_gap
        metric = incumbent.result.primary_metric

        leaderboard_text = ""
        if incumbent.result.leaderboard:
            rows = [
                f"  {e.model_name}: val={e.score_val:.4f}"
                + (f" train={e.score_train:.4f}" if e.score_train is not None else "")
                for e in incumbent.result.leaderboard[:5]
            ]
            leaderboard_text = "\nLeaderboard (top 5):\n" + "\n".join(rows)

        history_text = ""
        if prior_runs:
            lines = []
            for r in prior_runs[-5:]:
                m = r.result.primary_metric
                fams = r.plan.model_families if r.plan else []
                lines.append(
                    f"  run={r.run_id} metric={m} families={fams} status={r.result.status}"
                )
            history_text = "\n## Prior Runs\n" + "\n".join(lines)

        return (
            f"## Task\n"
            f"Name: {task.task_name} | Type: {task.task_type} | Metric: {task.eval_metric}\n\n"
            f"## Incumbent Config\n"
            f"model_families={families}\n"
            f"presets={incumbent.plan.presets}\n"
            f"time_limit={incumbent.plan.time_limit}\n"
            f"validation_policy={incumbent.plan.validation_policy}\n"
            f"metric={metric:.4f}\n"
            f"overfitting_gap={overfitting_gap}"
            f"{leaderboard_text}"
            f"{history_text}\n\n"
            f"Propose ONE targeted improvement as a JSON ExperimentPlan."
        )
