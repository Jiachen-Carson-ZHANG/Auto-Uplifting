# src/memory/context_builder.py
from __future__ import annotations
from typing import List, Optional
from src.models.nodes import SearchContext, ExperimentNode, CaseEntry
from src.models.task import TaskSpec
from src.models.results import DataProfile, RunEntry


class ContextBuilder:
    """Assembles a SearchContext briefing from session state. No LLM, no IO."""

    def build(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        history: List[RunEntry],
        incumbent: Optional[RunEntry],
        current_node: ExperimentNode,
        stage: str,
        budget_remaining: int,
        budget_used: int,
        similar_cases: List[CaseEntry],
    ) -> SearchContext:
        failed = [r for r in history if r.result.status == "failed"]
        return SearchContext(
            task=task,
            data_profile=data_profile,
            history=history,
            incumbent=incumbent,
            current_node=current_node,
            tree_summary={},
            similar_cases=similar_cases,
            failed_attempts=failed,
            stage=stage,
            budget_remaining=budget_remaining,
            budget_used=budget_used,
        )
