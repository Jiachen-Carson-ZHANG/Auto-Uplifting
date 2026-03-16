from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import RunEntry, DataProfile


class NodeStage(str, Enum):
    IDEATION = "ideation"
    WARMUP = "warmup"
    OPTIMIZE = "optimize"
    DEBUG = "debug"


class NodeStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    REJECTED = "rejected"


class ExperimentNode(BaseModel):
    node_id: str
    parent_id: Optional[str] = None
    children: List[str] = Field(default_factory=list)
    edge_label: Optional[str] = None   # what changed from parent — for Graph RAG
    stage: NodeStage = NodeStage.WARMUP
    status: NodeStatus = NodeStatus.PENDING
    plan: ExperimentPlan
    config: Optional[RunConfig] = None
    entry: Optional[RunEntry] = None
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
    task_type: str
    n_rows_bucket: str
    n_features_bucket: str
    class_balance: str
    feature_types: Dict[str, int] = Field(default_factory=dict)
    domain_tags: List[str] = Field(default_factory=list)


class WhatWorked(BaseModel):
    best_config: ExperimentPlan
    best_metric: float
    key_decisions: List[str] = Field(default_factory=list)
    important_features: List[str] = Field(default_factory=list)
    effective_presets: str = ""


class WhatFailed(BaseModel):
    failed_approaches: List[str] = Field(default_factory=list)
    failure_patterns: List[str] = Field(default_factory=list)


class SessionTrajectory(BaseModel):
    n_runs: int = 0
    total_time_seconds: float = 0.0
    metric_progression: List[float] = Field(default_factory=list)
    turning_points: List[str] = Field(default_factory=list)


class TreeSummary(BaseModel):
    n_nodes: int = 0
    n_branches: int = 0
    max_depth: int = 0
    winning_path: List[str] = Field(default_factory=list)
    edge_labels_on_winning_path: List[str] = Field(default_factory=list)


class CaseEntry(BaseModel):
    case_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    task_traits: TaskTraits
    what_worked: WhatWorked
    what_failed: WhatFailed
    trajectory: SessionTrajectory
    tree_summary: TreeSummary = Field(default_factory=TreeSummary)
    embedding: Optional[List[float]] = None


class SearchContext(BaseModel):
    task: TaskSpec
    data_profile: DataProfile
    history: List[RunEntry] = Field(default_factory=list)
    incumbent: Optional[RunEntry] = None
    current_node: ExperimentNode
    tree_summary: Dict[str, Any] = Field(default_factory=dict)
    similar_cases: List[CaseEntry] = Field(default_factory=list)
    failed_attempts: List[RunEntry] = Field(default_factory=list)
    stage: str = "warmup"
    budget_remaining: int = 5
    budget_used: int = 0
