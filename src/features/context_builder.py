"""
Assembles feature engineering context for the agent prompt.

No LLM, no IO — pure string assembly.
Follow pattern from src/memory/context_builder.py.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src.models.feature_engineering import FeatureHistoryEntry
from src.models.results import DataProfile
from src.models.task import TaskSpec


class FeatureContextBuilder:
    """Builds a formatted text context string for FeatureEngineeringAgent."""

    def build(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        leaderboard: List[dict],
        feature_importances: Dict[str, float],
        history: List[FeatureHistoryEntry],
        incumbent_metric: Optional[float],
        available_templates: List[str],
        budget_remaining: int,
        budget_used: int,
    ) -> str:
        sections: List[str] = []

        # Task info
        sections.append(self._task_section(task))

        # Data profile
        sections.append(self._profile_section(data_profile))

        # Model performance
        sections.append(self._performance_section(leaderboard, incumbent_metric))

        # Feature importances
        if feature_importances:
            sections.append(self._importances_section(feature_importances))

        # Available templates
        sections.append(self._templates_section(available_templates))

        # Feature history (empirical experiment memory)
        if history:
            sections.append(self._history_section(history))

        # Budget
        sections.append(self._budget_section(budget_remaining, budget_used))

        return "\n\n".join(sections)

    # ── Section builders ────────────────────────────────────────────

    def _task_section(self, task: TaskSpec) -> str:
        return (
            f"## Task\n"
            f"- Name: {task.task_name}\n"
            f"- Type: {task.task_type}\n"
            f"- Target: {task.target_column}\n"
            f"- Metric: {task.eval_metric}\n"
            f"- Description: {task.description or 'none'}"
        )

    def _profile_section(self, profile: DataProfile) -> str:
        lines = [
            "## Data Profile",
            f"- Rows: {profile.n_rows}",
            f"- Columns: {profile.n_features}",
        ]
        if profile.feature_types:
            lines.append(f"- Feature types: {profile.feature_types}")
        if profile.missing_rate > 0:
            lines.append(f"- Missing rate: {profile.missing_rate:.2%}")
        if profile.high_cardinality_cols:
            lines.append(f"- High-cardinality columns: {profile.high_cardinality_cols}")
        if profile.suspected_leakage_cols:
            lines.append(f"- Suspected leakage columns: {profile.suspected_leakage_cols}")
        return "\n".join(lines)

    def _performance_section(
        self, leaderboard: List[dict], incumbent_metric: Optional[float]
    ) -> str:
        lines = ["## Current Model Performance"]
        if incumbent_metric is not None:
            lines.append(f"- Best metric so far: {incumbent_metric:.4f}")
        if leaderboard:
            lines.append("- Leaderboard (top 3):")
            for entry in leaderboard[:3]:
                name = entry.get("model_name", entry.get("model", "unknown"))
                score = entry.get("score_val", entry.get("metric", "?"))
                lines.append(f"  - {name}: {score}")
        else:
            lines.append("- No leaderboard available yet.")
        return "\n".join(lines)

    def _importances_section(self, importances: Dict[str, float]) -> str:
        lines = ["## Feature Importances (top 10)"]
        sorted_imp = sorted(importances.items(), key=lambda x: -x[1])[:10]
        for name, score in sorted_imp:
            lines.append(f"  - {name}: {score:.4f}")
        return "\n".join(lines)

    def _templates_section(self, templates: List[str]) -> str:
        lines = ["## Available Templates"]
        for t in templates:
            lines.append(f"  - {t}")
        return "\n".join(lines)

    def _history_section(self, history: List[FeatureHistoryEntry]) -> str:
        lines = ["## Feature History (recent experiments)"]
        for entry in history[-5:]:
            delta = ""
            if entry.metric_before is not None and entry.metric_after is not None:
                d = entry.metric_after - entry.metric_before
                delta = f" (delta: {d:+.4f})"
            takeaway = f" — {entry.distilled_takeaway}" if entry.distilled_takeaway else ""
            lines.append(
                f"  - [{entry.action}] {entry.observed_outcome}{delta}{takeaway}"
            )
        return "\n".join(lines)

    def _budget_section(self, remaining: int, used: int) -> str:
        return (
            f"## Budget\n"
            f"- Used: {used}\n"
            f"- Remaining: {remaining}"
        )
