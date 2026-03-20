from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from src.models.task import RunConfig, ExperimentPlan


class ModelEntry(BaseModel):
    model_name: str
    score_val: float
    fit_time: float
    pred_time: float
    stack_level: int = 1
    score_train: Optional[float] = None


class RunResult(BaseModel):
    """What AutoGluon returned from a fit() call. No IDs — those live in ExperimentRun."""
    status: Literal["success", "failed"]
    primary_metric: Optional[float] = None
    leaderboard: List[ModelEntry] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    fit_time_seconds: float = 0.0
    error: Optional[str] = None


class DataProfile(BaseModel):
    n_rows: int
    n_features: int
    feature_types: Dict[str, int] = Field(default_factory=dict)
    target_distribution: Dict[str, Any] = Field(default_factory=dict)
    class_balance_ratio: float = 1.0
    missing_rate: float = 0.0
    high_cardinality_cols: List[str] = Field(default_factory=list)
    suspected_leakage_cols: List[str] = Field(default_factory=list)
    summary: str = ""


class RunDiagnostics(BaseModel):
    """Computed observations after a run completes. All fields optional — never required for correctness."""
    overfitting_gap: Optional[float] = None
    metric_vs_parent: Optional[float] = None
    failure_mode: Optional[str] = None


class RunEntry(BaseModel):
    run_id: str
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    config: RunConfig
    result: RunResult
    diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics)
    plan: Optional[ExperimentPlan] = None
    agent_rationale: str = ""
    agent_review: str = ""
