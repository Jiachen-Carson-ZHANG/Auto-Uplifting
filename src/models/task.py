from __future__ import annotations
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """Human-written problem definition — the starting point for every session.

    Created by: the user in project.yaml / main.py.
    Used by: Session.__init__ to configure the entire run; IdeatorAgent to generate
    hypotheses; SelectorAgent and RefinerAgent to keep proposals on-metric.
    """
    task_name: str
    task_type: str  # "binary", "multiclass", "regression"
    data_path: str
    target_column: str
    eval_metric: str
    constraints: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""


class ExperimentPlan(BaseModel):
    """One agent-proposed experiment — the decision unit of the optimization loop.

    Created by: IdeatorAgent (warm-up hypotheses), SelectorAgent (hypothesis → plan),
    or RefinerAgent (incumbent → improved plan). Each plan includes a rationale field
    explaining why the agent chose these settings.
    Sent to: ConfigMapper, which translates it into AutoGluon kwargs (RunConfig).
    Stored in: ExperimentNode.plan and ExperimentRun.plan for traceability.
    """
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
    """AutoGluon-ready execution config — the translated form of an ExperimentPlan.

    Created by: ConfigMapper.to_run_config, which unpacks ExperimentPlan fields
    into the exact kwargs AutoGluon's TabularPredictor.fit() expects.
    Sent to: AutoGluonRunner.run, which passes autogluon_kwargs directly to fit().
    IDs (run_id, node_id) live in ExperimentRun, not here.
    """
    autogluon_kwargs: Dict[str, Any]
    data_path: str
    output_dir: str
