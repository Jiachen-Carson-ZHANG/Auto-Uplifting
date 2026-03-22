from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import ExperimentRun, DataProfile


class NodeStage(str, Enum):
    """Which phase of the session this node belongs to."""
    IDEATION = "ideation"
    WARMUP = "warmup"
    OPTIMIZE = "optimize"
    DEBUG = "debug"


class NodeStatus(str, Enum):
    """Lifecycle state of a node — updated by session.execute_node as the run progresses."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"  # ran successfully but did not beat the incumbent


class ExperimentNode(BaseModel):
    """One vertex in the experiment search tree — represents a single config to try.

    Created by: session.create_candidate_nodes (warm-up) and session.run (optimize loop),
    each time an agent produces an ExperimentPlan.
    Managed by: OrchestrationState (tree), which tracks parent-child relationships.
    Updated by: session.execute_node, which attaches RunConfig and ExperimentRun after the run.

    parent_id / children form the tree structure. edge_label records what changed from the
    parent config — designed for future Graph RAG traversal across sessions.
    """
    node_id: str
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    edge_label: Optional[str] = None   # what changed from parent — for future Graph RAG
    stage: NodeStage = NodeStage.WARMUP
    status: NodeStatus = NodeStatus.PENDING
    plan: ExperimentPlan
    config: Optional[RunConfig] = None   # set after ConfigMapper translates the plan
    entry: Optional[ExperimentRun] = None  # set after the run completes
    depth: int = 0
    debug_depth: int = 0
    created_at: datetime = Field(default_factory=datetime.now)

    def is_root(self) -> bool:
        return self.parent_id is None

    def has_result(self) -> bool:
        return self.entry is not None and self.entry.result.status == "success"

    def primary_metric(self) -> Optional[float]:
        if self.entry and self.entry.result:
            return self.entry.result.primary_metric
        return None


class TaskTraits(BaseModel):
    """Bucketed description of a task used for similarity search across sessions.

    Created by: trait_utils helpers (rows_bucket, features_bucket, balance_bucket),
    called by Distiller when building a CaseEntry at session end.
    Used by: CaseRetriever, which converts TaskTraits into a 7-dim vector and computes
    cosine similarity to find past cases relevant to the current task.
    """
    task_type: str           # "binary", "multiclass", "regression"
    n_rows_bucket: str       # "small" / "medium" / "large"
    n_features_bucket: str   # "small" / "medium" / "large"
    class_balance: str       # "balanced" / "moderate" / "severe"
    feature_types: Dict[str, int] = Field(default_factory=dict)
    domain_tags: List[str] = Field(default_factory=list)


class WhatWorked(BaseModel):
    """LLM-distilled summary of what succeeded in a session.

    Created by: Distiller (LLM call at session end), which reads the full ExperimentRun
    history and produces natural-language key_decisions and effective_presets.
    Stored in: CaseEntry.what_worked.
    Read by: IdeatorAgent at the start of a new session via similar_cases, to bias
    initial hypotheses toward known-good approaches for similar tasks.
    """
    best_config: ExperimentPlan
    best_metric: float
    key_decisions: List[str] = Field(default_factory=list)
    important_features: List[str] = Field(default_factory=list)
    effective_presets: str = ""


class WhatFailed(BaseModel):
    """LLM-distilled summary of what failed in a session.

    Created by: Distiller alongside WhatWorked. Captures failed run descriptions
    and the patterns behind them (e.g. "RF underperforms on imbalanced data").
    Stored in: CaseEntry.what_failed.
    Read by: IdeatorAgent to avoid proposing approaches that already failed on similar tasks.
    """
    failed_approaches: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)


class SessionTrajectory(BaseModel):
    """Numeric summary of the session's search path — no LLM involved.

    Created by: Distiller, computed directly from the ExperimentRun history
    (metric_progression, timing, turning points).
    Stored in: CaseEntry.trajectory. Useful for understanding how fast the session
    converged and where the biggest improvements happened.
    """
    n_runs: int = 0
    total_time_seconds: float = 0.0
    metric_progression: List[float] = Field(default_factory=list)
    turning_points: List[str] = Field(default_factory=list)


class TreeSummary(BaseModel):
    """Structural summary of the ExperimentNode tree for a session.

    Created by: Distiller from the node tree. Currently mostly empty (winning_path,
    edge_labels not yet populated) — pre-built for future Graph RAG use where traversal
    across sessions requires the branching structure, not just the flat run log.
    Stored in: CaseEntry.tree_summary.
    """
    n_nodes: int = 0
    n_branches: int = 0
    max_depth: int = 0
    winning_path: List[str] = Field(default_factory=list)
    edge_labels_on_winning_path: List[str] = Field(default_factory=list)


class CaseEntry(BaseModel):
    """Long-term memory record for one completed session — persisted to case_bank.jsonl.

    Created by: Distiller at session end, compressing the full ExperimentRun history
    into a structured summary (TaskTraits + WhatWorked + WhatFailed + trajectory).
    Stored in: CaseStore (case_bank.jsonl), one entry per session.
    Read by: CaseRetriever at the start of future sessions to surface relevant past experience,
    which is then passed to IdeatorAgent as similar_cases context.

    description_for_embedding: human-readable text summary used as input to embed().
                                Built by Distiller from task traits + key decisions.
                                Empty string means embedding was not attempted.
    embedding: populated by Distiller when OpenAIBackend is available; None otherwise.
    """
    case_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    task_traits: TaskTraits
    what_worked: WhatWorked
    what_failed: WhatFailed
    trajectory: SessionTrajectory
    tree_summary: TreeSummary = Field(default_factory=TreeSummary)
    description_for_embedding: str = ""   # text used as embed() input (Step 6)
    embedding: Optional[List[float]] = None  # populated by Distiller when OpenAIBackend available


class SearchContext(BaseModel):
    """Full briefing assembled by ContextBuilder before each agent decision.

    Created by: ContextBuilder.build, which collects all live session state into one object.
    Sent to: agents (RefinerAgent, SelectorAgent) as their single structured input,
    so agents receive everything they need without accessing session internals directly.
    Contains: task definition, data profile, full run history, current incumbent,
    similar past cases from CaseStore, and budget information.
    """
    task: TaskSpec
    data_profile: DataProfile
    history: List[ExperimentRun] = Field(default_factory=list)
    incumbent: Optional[ExperimentRun] = None
    current_node: ExperimentNode
    tree_summary: Dict[str, Any] = Field(default_factory=dict)
    similar_cases: List[CaseEntry] = Field(default_factory=list)
    failed_attempts: List[ExperimentRun] = Field(default_factory=list)
    stage: str = "warmup"
    budget_remaining: int = 5
    budget_used: int = 0
