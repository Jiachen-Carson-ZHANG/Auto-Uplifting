from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    task_name: str
    task_type: str  # "binary", "multiclass", "regression"
    data_path: str
    target_column: str
    eval_metric: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class ExperimentPlan(BaseModel):
    eval_metric: str
    model_families: List[str]
    presets: str
    time_limit: int
    feature_policy: Dict[str, List[str]] = Field(
        default_factory=lambda: {"exclude_columns": [], "include_columns": []}
    )
    validation_policy: Dict[str, Any] = Field(
        default_factory=lambda: {"holdout_frac": 0.2, "num_bag_folds": 0}
    )
    hyperparameters: Optional[Dict[str, Any]] = None
    use_fit_extra: bool = False
    rationale: str = ""


class RunConfig(BaseModel):
    run_id: str
    node_id: str
    autogluon_kwargs: Dict[str, Any]
    data_path: str
    output_dir: str
