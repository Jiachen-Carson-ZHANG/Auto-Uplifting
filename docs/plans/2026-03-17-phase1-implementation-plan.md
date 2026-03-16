# Hybrid Agentic ML Framework — Phase 1 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a thin but complete slice through all 4 layers (Agent → Orchestration → Execution → Memory) that runs 3–5 real AutoGluon iterations on a demo tabular dataset.

**Architecture:** Four layers with clean interfaces. Agent outputs structured ExperimentPlan JSON → Orchestration creates ExperimentNode in a graph tree and maps to RunConfig → Execution runs AutoGluon → Memory stores RunEntry with diagnostics. No LangChain. No free-form code generation from the agent.

**Tech Stack:** Python 3.11+, Pydantic v2, AutoGluon (tabular), anthropic SDK, openai SDK, scikit-learn (cosine similarity for Phase 2 retrieval), PyYAML, pandas, pytest

---

## Phase 1 Task Map

```
Task 1: Project scaffold (configs, folders, requirements)
Task 2: Data models — src/models/ (Pydantic)
Task 3: LLM backend — protocol + Anthropic provider
Task 4: Execution layer — config_mapper, autogluon_runner, result_parser
Task 5: Memory — RunStore (append-only JSONL)
Task 6: Orchestration — ExperimentNode + ExperimentTree (state.py)
Task 7: Agent — selector.py (hypothesis → ExperimentPlan via LLM)
Task 8: Agent — manager.py (routes decisions, warmup loop)
Task 9: Wire it — main.py running 3 iterations end-to-end
Task 10: LLM backend — OpenAI provider + factory routing
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `configs/project.yaml`
- Create: `configs/search.yaml`
- Create: `configs/models.yaml`
- Create: `configs/seed_ideas.json`
- Create: `prompts/selector.md`
- Create: `prompts/refiner.md`
- Create: `prompts/reviewer.md`
- Create: `prompts/ideator.md`
- Create: `prompts/distiller.md`
- Create all `__init__.py` files

**Step 1: Create requirements.txt**

```
autogluon.tabular>=1.2.0
pydantic>=2.0.0
anthropic>=0.25.0
openai>=1.30.0
pyyaml>=6.0
pandas>=2.0.0
scikit-learn>=1.3.0
pytest>=8.0.0
python-dotenv>=1.0.0
numpy>=1.26.0
```

**Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

**Step 3: Create configs/project.yaml**

```yaml
task:
  name: "demo-titanic"
  type: "binary"
  data_path: "data/titanic_train.csv"
  target_column: "Survived"
  eval_metric: "roc_auc"
  description: "Predict passenger survival on the Titanic. Binary classification."
  constraints:
    max_time_per_run: 120
    forbidden_models: []

llm:
  provider: "anthropic"
  model: "claude-sonnet-4-6"
  temperature: 0.3

session:
  case_store_path: "experiments/case_bank.jsonl"
  experiments_dir: "experiments"
```

**Step 4: Create configs/search.yaml**

```yaml
search:
  num_candidates: 3          # warmup candidates
  min_warmup_runs: 1         # runs per candidate before advancing
  max_optimize_iterations: 5
  plateau_threshold: 0.001   # stop if improvement < this for N steps
  plateau_patience: 3        # N steps of no improvement before stopping
  higher_is_better:
    roc_auc: true
    f1_macro: true
    f1: true
    accuracy: true
    rmse: false
    mae: false
    mse: false
```

**Step 5: Create configs/models.yaml**

```yaml
allowed_families:
  - GBM
  - XGB
  - CAT
  - RF
  - XT
  - NN_TORCH
  - KNN
  - LR

presets:
  fast: "medium_quality"
  balanced: "good_quality"
  thorough: "best_quality"

default_time_limit: 120
default_presets: "medium_quality"
```

**Step 6: Create configs/seed_ideas.json**

```json
[
  {
    "hypothesis": "Start with gradient boosting (GBM) on all features with roc_auc metric and holdout validation as a reliable baseline.",
    "rationale": "GBM is a strong default for tabular binary classification. roc_auc is robust to class imbalance."
  },
  {
    "hypothesis": "Try tree ensemble methods (RF + XT) with f1_macro metric to handle potential class imbalance.",
    "rationale": "F1 macro penalizes both false positives and negatives equally, useful when classes are unbalanced."
  },
  {
    "hypothesis": "Use a diverse model mix (GBM + XGB + CAT) with accuracy metric and 5-fold cross-validation.",
    "rationale": "Multiple boosting algorithms with CV gives a more stable estimate than holdout."
  }
]
```

**Step 7: Create prompts/selector.md**

````markdown
# Selector Agent

You are an ML experiment designer. Your job is to turn a hypothesis into a concrete experiment plan.

## Input
You will receive:
1. A hypothesis (natural language description of what to try)
2. A task description (what problem we're solving)
3. A data profile (dataset statistics)
4. Prior run history (what has already been tried, may be empty)

## Output
You MUST output a valid JSON object matching this schema exactly. No prose, no explanation outside the JSON.

```json
{
  "eval_metric": "roc_auc",
  "model_families": ["GBM", "XGB"],
  "presets": "medium_quality",
  "time_limit": 120,
  "feature_policy": {
    "exclude_columns": [],
    "include_columns": []
  },
  "validation_policy": {
    "holdout_frac": 0.2,
    "num_bag_folds": 0
  },
  "hyperparameters": null,
  "use_fit_extra": false,
  "rationale": "Why this plan makes sense given the task and history"
}
```

## Rules
- eval_metric must be one of: roc_auc, f1_macro, f1, accuracy, rmse, mae
- model_families must be a subset of: GBM, XGB, CAT, RF, XT, NN_TORCH, KNN, LR
- presets must be one of: medium_quality, good_quality, best_quality
- time_limit must be between 60 and 3600 seconds
- If num_bag_folds > 0, set holdout_frac to 0.0
- If holdout_frac > 0, set num_bag_folds to 0
- use_fit_extra should be false for the first run, true only when explicitly improving an existing run
- Do NOT suggest NN_TORCH if n_rows < 1000 (overfitting risk on small data)
- rationale must be grounded in the task description, data profile, or prior results
- Respond with ONLY the JSON object. No markdown fences, no explanation.
````

**Step 8: Create prompts/refiner.md**

````markdown
# Refiner Agent

You are an ML experiment optimizer. You analyze the current best run and propose one targeted improvement.

## Input
You will receive:
1. The current incumbent run (best result so far)
2. All prior run history with diagnostics
3. The task description and data profile
4. The current search context (stage, budget remaining, similar cases)

## Output
Output a valid JSON ExperimentPlan proposing ONE clear change from the incumbent.
Focus on the change most likely to improve the primary metric based on the diagnostics.

```json
{
  "eval_metric": "roc_auc",
  "model_families": ["GBM"],
  "presets": "good_quality",
  "time_limit": 180,
  "feature_policy": {
    "exclude_columns": ["column_with_leakage"],
    "include_columns": []
  },
  "validation_policy": {
    "holdout_frac": 0.0,
    "num_bag_folds": 5
  },
  "hyperparameters": null,
  "use_fit_extra": false,
  "rationale": "The diagnostics showed a 0.15 train-val gap suggesting overfitting. Switching to 5-fold CV should give a more stable estimate and reduce overfitting."
}
```

## Improvement Axes (try ONE per iteration)
1. Metric: if class_balance_ratio < 0.3, consider f1_macro over accuracy
2. Validation: if overfitting_gap > 0.05, switch to k-fold; if slow, use holdout
3. Features: if suspected_leakage_cols is not empty, exclude them
4. Model families: if current best model is GBM, try adding CAT or XGB
5. Budget: if primary_metric is still improving, increase time_limit by 50%
6. Presets: if time permits, upgrade from medium_quality to good_quality

## Rules
- Same schema and rules as the selector prompt
- Change exactly ONE axis at a time so results are interpretable
- Do not suggest what has already been tried (check prior history)
- Respond with ONLY the JSON object.
````

**Step 9: Create prompts/reviewer.md**

````markdown
# Reviewer Agent

You assess whether a completed run result is valid, trustworthy, and whether to accept it as the new incumbent.

## Input
You will receive:
1. The run entry (config, result, diagnostics)
2. The parent node's result (what we're comparing against), or null if this is a root node
3. The task description

## Output
Output a JSON assessment:

```json
{
  "is_valid": true,
  "accept_as_incumbent": true,
  "issues": [],
  "observations": "GBM achieved roc_auc=0.87, improving over parent by +0.03. No overfitting detected (gap=0.02). Feature importances show Age and Fare dominating.",
  "recommended_next_axis": "validation_policy"
}
```

## Validity Checks
- is_valid = false if: status is "failed", primary_metric is null, or error is not null
- Flag issues if: overfitting_gap > 0.1 ("high overfitting gap"), class_balance_ratio < 0.2 and metric is accuracy ("misleading metric for imbalanced data")
- accept_as_incumbent = true if: is_valid AND (parent is null OR metric improved vs parent)

## Rules
- issues is a list of warning strings, may be empty
- recommended_next_axis is one of: eval_metric, model_families, validation_policy, feature_policy, time_limit, presets, stop
- Respond with ONLY the JSON object.
````

**Step 10: Create prompts/ideator.md**

````markdown
# Ideator Agent

You generate initial experiment hypotheses for a new ML task.

## Input
1. Task description and data profile
2. Similar past cases from the case bank (may be empty)
3. Seed ideas from configs (may be empty)

## Output
Output a JSON array of exactly K hypotheses (K will be specified in the prompt):

```json
[
  {
    "id": "h1",
    "model_focus": "GBM",
    "metric_focus": "roc_auc",
    "hypothesis": "Start with gradient boosting on all features as a reliable baseline.",
    "rationale": "GBM is a strong default for tabular binary classification."
  },
  {
    "id": "h2",
    "model_focus": "RF",
    "metric_focus": "f1_macro",
    "hypothesis": "Try random forest with f1_macro given the class imbalance detected in the data profile.",
    "rationale": "class_balance_ratio=0.23 suggests imbalance; f1_macro penalizes both false positive and false negative equally."
  }
]
```

## Rules
- Each hypothesis should explore a different angle (different model focus OR different metric)
- If similar cases exist, incorporate their lessons (e.g., "past similar task found NN_TORCH overfits on small data")
- Do NOT suggest NN_TORCH if n_rows < 1000
- Respond with ONLY the JSON array.
````

**Step 11: Create prompts/distiller.md**

````markdown
# Distiller Agent

You summarize a completed experiment session into a reusable case entry for the knowledge base.

## Input
1. The full run history (all RunEntry objects)
2. The experiment tree structure (nodes, edges, incumbent path)
3. The task description and data profile

## Output
```json
{
  "what_worked": {
    "key_decisions": [
      "Switching from accuracy to f1_macro improved metric by +0.08 due to class imbalance",
      "Excluding suspected leakage column 'ticket_number' improved generalization"
    ],
    "important_features": ["Age", "Fare", "Pclass"],
    "effective_presets": "good_quality"
  },
  "what_failed": {
    "failed_approaches": ["NN_TORCH crashed with batch size error on 891 rows"],
    "failure_patterns": ["Neural networks overfit on tabular datasets smaller than 1000 rows"]
  },
  "trajectory": {
    "turning_points": [
      "Run 2: switching metric from accuracy to f1_macro was the key insight",
      "Run 4: k-fold validation revealed the holdout estimate was optimistic"
    ]
  }
}
```

## Rules
- key_decisions should be specific and include the metric delta where available
- failure_patterns should be generalized (not task-specific) so they transfer to future tasks
- turning_points should describe WHY the session changed direction
- Respond with ONLY the JSON object.
````

**Step 12: Create all __init__.py files**

```bash
mkdir -p src/agents src/orchestration src/execution src/memory src/llm/providers src/models
touch src/__init__.py
touch src/agents/__init__.py
touch src/orchestration/__init__.py
touch src/execution/__init__.py
touch src/memory/__init__.py
touch src/llm/__init__.py
touch src/llm/providers/__init__.py
touch src/models/__init__.py
mkdir -p tests/agents tests/orchestration tests/execution tests/memory tests/llm tests/models
touch tests/__init__.py tests/agents/__init__.py tests/orchestration/__init__.py
touch tests/execution/__init__.py tests/memory/__init__.py tests/llm/__init__.py tests/models/__init__.py
mkdir -p experiments notebooks data
```

**Step 13: Download demo dataset**

```python
# scripts/download_demo_data.py
import pandas as pd

# Titanic dataset — classic binary classification, small (891 rows)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.to_csv("data/titanic_train.csv", index=False)
print(f"Downloaded: {len(df)} rows, {len(df.columns)} columns")
print(f"Target distribution:\n{df['Survived'].value_counts()}")
```

Run: `python scripts/download_demo_data.py`
Expected: "Downloaded: 891 rows, 12 columns"

**Step 14: Commit scaffold**

```bash
git add .
git commit -m "feat: project scaffold — configs, prompts, folder structure"
```

---

## Task 2: Data Models

**Files:**
- Create: `src/models/task.py`
- Create: `src/models/results.py`
- Create: `src/models/nodes.py`
- Test: `tests/models/test_models.py`

**Step 1: Write the failing tests**

```python
# tests/models/test_models.py
import pytest
from datetime import datetime
from src.models.task import TaskSpec, ExperimentPlan, RunConfig
from src.models.results import ModelEntry, RunResult, DataProfile, RunDiagnostics, RunEntry
from src.models.nodes import ExperimentNode, NodeStatus, NodeStage, SearchContext, CaseEntry


def test_task_spec_from_dict():
    spec = TaskSpec(
        task_name="test",
        task_type="binary",
        data_path="data/test.csv",
        target_column="label",
        eval_metric="roc_auc",
        constraints={"max_time_per_run": 120},
        description="Test task"
    )
    assert spec.task_name == "test"
    assert spec.eval_metric == "roc_auc"


def test_experiment_plan_validation():
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM", "XGB"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Baseline test"
    )
    assert plan.time_limit == 120
    assert "GBM" in plan.model_families


def test_run_config_has_node_id():
    config = RunConfig(
        run_id="run_0001",
        node_id="node_abc",
        autogluon_kwargs={"eval_metric": "roc_auc", "time_limit": 120},
        data_path="data/test.csv",
        output_dir="experiments/test/runs/run_0001"
    )
    assert config.node_id == "node_abc"


def test_run_result_failed():
    result = RunResult(
        run_id="run_0001",
        status="failed",
        primary_metric=None,
        leaderboard=[],
        best_model_name=None,
        fit_time_seconds=0.0,
        artifacts_dir="experiments/test/runs/run_0001",
        error="AutoGluon crashed: OOM",
        raw_info={}
    )
    assert result.status == "failed"
    assert result.primary_metric is None


def test_experiment_node_root():
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Root node"
    )
    node = ExperimentNode(
        node_id="node_001",
        parent_id=None,
        children=[],
        edge_label=None,
        stage=NodeStage.WARMUP,
        status=NodeStatus.PENDING,
        plan=plan,
        config=None,
        entry=None,
        depth=0,
        debug_depth=0,
        created_at=datetime.now()
    )
    assert node.parent_id is None
    assert node.depth == 0
    assert node.edge_label is None


def test_experiment_node_child_has_edge_label():
    plan = ExperimentPlan(
        eval_metric="f1_macro",
        model_families=["GBM"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Changed metric"
    )
    node = ExperimentNode(
        node_id="node_002",
        parent_id="node_001",
        children=[],
        edge_label="changed eval_metric from roc_auc to f1_macro due to class imbalance",
        stage=NodeStage.OPTIMIZE,
        status=NodeStatus.PENDING,
        plan=plan,
        config=None,
        entry=None,
        depth=1,
        debug_depth=0,
        created_at=datetime.now()
    )
    assert node.parent_id == "node_001"
    assert "eval_metric" in node.edge_label
```

**Step 2: Run to verify failure**

```bash
pytest tests/models/test_models.py -v
```
Expected: `ImportError` — modules don't exist yet.

**Step 3: Implement src/models/task.py**

```python
# src/models/task.py
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
```

**Step 4: Implement src/models/results.py**

```python
# src/models/results.py
from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from src.models.task import RunConfig


class ModelEntry(BaseModel):
    model_name: str
    score_val: float
    fit_time: float
    pred_time: float
    stack_level: int = 1


class RunResult(BaseModel):
    run_id: str
    status: str  # "success" | "failed"
    primary_metric: Optional[float] = None
    leaderboard: List[ModelEntry] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    fit_time_seconds: float = 0.0
    artifacts_dir: str = ""
    error: Optional[str] = None
    raw_info: Dict[str, Any] = Field(default_factory=dict)


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
    data_profile_ref: str = ""
    overfitting_gap: Optional[float] = None
    feature_importances: Dict[str, float] = Field(default_factory=dict)
    metric_vs_parent: Optional[float] = None
    change_description: str = ""
    failure_mode: Optional[str] = None


class RunEntry(BaseModel):
    run_id: str
    node_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    config: RunConfig
    result: RunResult
    diagnostics: RunDiagnostics = Field(default_factory=RunDiagnostics)
    agent_rationale: str = ""
    agent_review: str = ""
```

**Step 5: Implement src/models/nodes.py**

```python
# src/models/nodes.py
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
    n_rows_bucket: str   # "small" (<5k), "medium" (5k-100k), "large" (>100k)
    n_features_bucket: str  # "narrow" (<20), "wide" (>=20)
    class_balance: str   # "balanced", "moderate_imbalance", "severe_imbalance"
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
    winning_path: List[str] = Field(default_factory=list)          # node IDs
    edge_labels_on_winning_path: List[str] = Field(default_factory=list)  # for Graph RAG


class CaseEntry(BaseModel):
    case_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    task_traits: TaskTraits
    what_worked: WhatWorked
    what_failed: WhatFailed
    trajectory: SessionTrajectory
    tree_summary: TreeSummary = Field(default_factory=TreeSummary)
    embedding: Optional[List[float]] = None   # for future vector retrieval


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
```

**Step 6: Run tests**

```bash
pytest tests/models/test_models.py -v
```
Expected: All 6 tests PASS.

**Step 7: Commit**

```bash
git add src/models/ tests/models/
git commit -m "feat: data models — TaskSpec, ExperimentPlan, RunEntry, ExperimentNode, SearchContext"
```

---

## Task 3: LLM Backend

**Files:**
- Create: `src/llm/backend.py`
- Create: `src/llm/providers/anthropic.py`
- Test: `tests/llm/test_backend.py`

**Step 1: Write the failing tests**

```python
# tests/llm/test_backend.py
import pytest
from unittest.mock import MagicMock, patch
from src.llm.backend import create_backend, LLMBackend
from src.llm.providers.anthropic import AnthropicBackend


def test_anthropic_backend_implements_protocol():
    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", api_key="test-key")
    assert isinstance(backend, LLMBackend)


def test_create_backend_anthropic():
    backend = create_backend(provider="anthropic", model="claude-haiku-4-5-20251001", api_key="test-key")
    assert isinstance(backend, AnthropicBackend)


def test_create_backend_unknown_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_backend(provider="unknown_llm", model="model", api_key="key")


def test_anthropic_backend_complete(monkeypatch):
    """Test that AnthropicBackend calls the API and returns content."""
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"eval_metric": "roc_auc"}')]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_response

    backend = AnthropicBackend(model="claude-haiku-4-5-20251001", api_key="test-key")
    backend._client = mock_client

    result = backend.complete(messages=[{"role": "user", "content": "test"}])
    assert result == '{"eval_metric": "roc_auc"}'
    mock_client.messages.create.assert_called_once()
```

**Step 2: Run to verify failure**

```bash
pytest tests/llm/test_backend.py -v
```
Expected: `ImportError`

**Step 3: Implement src/llm/backend.py**

```python
# src/llm/backend.py
from __future__ import annotations
from typing import Optional, runtime_checkable, Protocol


@runtime_checkable
class LLMBackend(Protocol):
    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: Optional[dict] = None,
    ) -> str: ...


def create_backend(provider: str, model: str, api_key: str) -> LLMBackend:
    """Factory — returns the correct LLMBackend for the given provider."""
    if provider == "anthropic":
        from src.llm.providers.anthropic import AnthropicBackend
        return AnthropicBackend(model=model, api_key=api_key)
    elif provider == "openai":
        from src.llm.providers.openai import OpenAIBackend
        return OpenAIBackend(model=model, api_key=api_key)
    else:
        raise ValueError(f"Unknown provider: {provider!r}. Choose 'anthropic' or 'openai'.")
```

**Step 4: Implement src/llm/providers/anthropic.py**

```python
# src/llm/providers/anthropic.py
from __future__ import annotations
from typing import Optional
import anthropic


class AnthropicBackend:
    def __init__(self, model: str, api_key: str):
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: Optional[dict] = None,
    ) -> str:
        # Anthropic requires system message separate from messages list
        system = None
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        kwargs = dict(
            model=self._model,
            max_tokens=4096,
            messages=filtered,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text
```

**Step 5: Run tests**

```bash
pytest tests/llm/test_backend.py -v
```
Expected: All 4 tests PASS.

**Step 6: Commit**

```bash
git add src/llm/ tests/llm/
git commit -m "feat: LLM backend protocol + Anthropic provider"
```

---

## Task 4: Execution Layer

**Files:**
- Create: `src/execution/config_mapper.py`
- Create: `src/execution/autogluon_runner.py`
- Create: `src/execution/result_parser.py`
- Test: `tests/execution/test_config_mapper.py`
- Test: `tests/execution/test_result_parser.py`

**Step 1: Write the failing tests**

```python
# tests/execution/test_config_mapper.py
import pytest
from src.models.task import ExperimentPlan, RunConfig
from src.execution.config_mapper import ConfigMapper


@pytest.fixture
def basic_plan():
    return ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM", "XGB"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="Test"
    )


def test_config_mapper_produces_run_config(basic_plan):
    mapper = ConfigMapper()
    config = mapper.plan_to_config(
        plan=basic_plan,
        run_id="run_0001",
        node_id="node_001",
        data_path="data/test.csv",
        output_dir="experiments/test/runs/run_0001"
    )
    assert isinstance(config, RunConfig)
    assert config.run_id == "run_0001"
    assert config.node_id == "node_001"


def test_config_mapper_eval_metric(basic_plan):
    mapper = ConfigMapper()
    config = mapper.plan_to_config(basic_plan, "run_0001", "n1", "d.csv", "out/")
    assert config.autogluon_kwargs["eval_metric"] == "roc_auc"


def test_config_mapper_model_families(basic_plan):
    mapper = ConfigMapper()
    config = mapper.plan_to_config(basic_plan, "run_0001", "n1", "d.csv", "out/")
    hp = config.autogluon_kwargs["hyperparameters"]
    assert "GBM" in hp
    assert "XGB" in hp


def test_config_mapper_holdout_validation(basic_plan):
    mapper = ConfigMapper()
    config = mapper.plan_to_config(basic_plan, "run_0001", "n1", "d.csv", "out/")
    assert config.autogluon_kwargs["holdout_frac"] == 0.2


def test_config_mapper_kfold_validation():
    plan = ExperimentPlan(
        eval_metric="f1_macro",
        model_families=["GBM"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.0, "num_bag_folds": 5},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="k-fold test"
    )
    mapper = ConfigMapper()
    config = mapper.plan_to_config(plan, "run_0001", "n1", "d.csv", "out/")
    assert config.autogluon_kwargs["num_bag_folds"] == 5
    assert config.autogluon_kwargs.get("holdout_frac", 0.2) == 0.2  # default when kfold


def test_config_mapper_exclude_columns():
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM"],
        presets="medium_quality",
        time_limit=120,
        feature_policy={"exclude_columns": ["PassengerId", "Name"], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None,
        use_fit_extra=False,
        rationale="drop leaky cols"
    )
    mapper = ConfigMapper()
    config = mapper.plan_to_config(plan, "run_0001", "n1", "d.csv", "out/")
    assert config.autogluon_kwargs["excluded_model_types"] == []
    assert "PassengerId" in config.autogluon_kwargs.get("_exclude_columns", [])
```

```python
# tests/execution/test_result_parser.py
import pytest
from unittest.mock import MagicMock, patch
from src.execution.result_parser import ResultParser
from src.models.results import RunResult


def test_result_parser_success():
    mock_predictor = MagicMock()
    mock_predictor.leaderboard.return_value = MagicMock(
        to_dict=lambda orient: {
            "model": {"0": "LightGBM", "1": "WeightedEnsemble"},
            "score_val": {"0": 0.87, "1": 0.89},
            "fit_time": {"0": 10.0, "1": 1.0},
            "pred_time_val": {"0": 0.1, "1": 0.05},
            "stack_level": {"0": 1, "1": 2},
        }
    )
    mock_predictor.info.return_value = {"best_model": "WeightedEnsemble", "eval_metric": "roc_auc"}

    parser = ResultParser()
    result = parser.parse(
        run_id="run_0001",
        predictor=mock_predictor,
        eval_metric="roc_auc",
        fit_time=15.0,
        artifacts_dir="out/"
    )
    assert result.status == "success"
    assert result.run_id == "run_0001"
    assert result.fit_time_seconds == 15.0
    assert len(result.leaderboard) > 0


def test_result_parser_failure():
    parser = ResultParser()
    result = parser.parse_failure(
        run_id="run_0001",
        error="AutoGluon raised OOM",
        artifacts_dir="out/"
    )
    assert result.status == "failed"
    assert result.primary_metric is None
    assert "OOM" in result.error
```

**Step 2: Run to verify failure**

```bash
pytest tests/execution/ -v
```
Expected: `ImportError`

**Step 3: Implement src/execution/config_mapper.py**

```python
# src/execution/config_mapper.py
from __future__ import annotations
from typing import Any, Dict
from src.models.task import ExperimentPlan, RunConfig


class ConfigMapper:
    """Translates an ExperimentPlan into AutoGluon TabularPredictor.fit() kwargs."""

    def plan_to_config(
        self,
        plan: ExperimentPlan,
        run_id: str,
        node_id: str,
        data_path: str,
        output_dir: str,
    ) -> RunConfig:
        kwargs: Dict[str, Any] = {}

        # Metric
        kwargs["eval_metric"] = plan.eval_metric

        # Model families -> hyperparameters dict
        if plan.hyperparameters:
            kwargs["hyperparameters"] = plan.hyperparameters
        else:
            kwargs["hyperparameters"] = {family: {} for family in plan.model_families}

        # Presets and time
        kwargs["presets"] = plan.presets
        kwargs["time_limit"] = plan.time_limit

        # Validation strategy
        vp = plan.validation_policy
        if vp.get("num_bag_folds", 0) > 0:
            kwargs["num_bag_folds"] = vp["num_bag_folds"]
            kwargs["num_bag_sets"] = 1
        else:
            kwargs["holdout_frac"] = vp.get("holdout_frac", 0.2)

        # Feature exclusions (stored for data pre-filtering before fit)
        excluded = plan.feature_policy.get("exclude_columns", [])
        kwargs["_exclude_columns"] = excluded   # handled by runner, not passed to AG directly
        kwargs["excluded_model_types"] = []     # separate from column exclusion

        return RunConfig(
            run_id=run_id,
            node_id=node_id,
            autogluon_kwargs=kwargs,
            data_path=data_path,
            output_dir=output_dir,
        )
```

**Step 4: Implement src/execution/result_parser.py**

```python
# src/execution/result_parser.py
from __future__ import annotations
from typing import Any, Optional
from src.models.results import ModelEntry, RunResult


class ResultParser:
    """Converts AutoGluon predictor output into a RunResult."""

    def parse(
        self,
        run_id: str,
        predictor: Any,
        eval_metric: str,
        fit_time: float,
        artifacts_dir: str,
    ) -> RunResult:
        try:
            lb = predictor.leaderboard(silent=True)
            leaderboard = []
            for _, row in lb.iterrows():
                leaderboard.append(ModelEntry(
                    model_name=str(row.get("model", "")),
                    score_val=float(row.get("score_val", 0.0)),
                    fit_time=float(row.get("fit_time", 0.0)),
                    pred_time=float(row.get("pred_time_val", 0.0)),
                    stack_level=int(row.get("stack_level", 1)),
                ))

            primary_metric = lb["score_val"].max() if len(lb) > 0 else None
            best_model = lb.iloc[0]["model"] if len(lb) > 0 else None

            info = {}
            try:
                info = predictor.info()
            except Exception:
                pass

            return RunResult(
                run_id=run_id,
                status="success",
                primary_metric=float(primary_metric) if primary_metric is not None else None,
                leaderboard=leaderboard,
                best_model_name=str(best_model) if best_model else None,
                fit_time_seconds=fit_time,
                artifacts_dir=artifacts_dir,
                error=None,
                raw_info=info,
            )
        except Exception as e:
            return self.parse_failure(run_id=run_id, error=str(e), artifacts_dir=artifacts_dir)

    def parse_failure(self, run_id: str, error: str, artifacts_dir: str) -> RunResult:
        return RunResult(
            run_id=run_id,
            status="failed",
            primary_metric=None,
            leaderboard=[],
            best_model_name=None,
            fit_time_seconds=0.0,
            artifacts_dir=artifacts_dir,
            error=error,
            raw_info={},
        )
```

**Step 5: Implement src/execution/autogluon_runner.py**

```python
# src/execution/autogluon_runner.py
from __future__ import annotations
import os
import time
import pandas as pd
from pathlib import Path
from src.models.task import RunConfig
from src.models.results import RunResult
from src.execution.result_parser import ResultParser


class AutoGluonRunner:
    """Runs AutoGluon TabularPredictor for a given RunConfig."""

    def __init__(self, target_column: str):
        self._target = target_column
        self._parser = ResultParser()

    def run(self, config: RunConfig) -> RunResult:
        from autogluon.tabular import TabularPredictor

        out_dir = config.output_dir
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        # Load data
        try:
            df = pd.read_csv(config.data_path)
        except Exception as e:
            return self._parser.parse_failure(config.run_id, f"Data load failed: {e}", out_dir)

        # Apply column exclusions (agent-level feature policy)
        exclude_cols = config.autogluon_kwargs.pop("_exclude_columns", [])
        if exclude_cols:
            df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")

        # Separate AG kwargs (remove our internal keys)
        ag_kwargs = {k: v for k, v in config.autogluon_kwargs.items()
                     if not k.startswith("_")}

        predictor_path = os.path.join(out_dir, "predictor")
        start = time.time()
        try:
            predictor = TabularPredictor(
                label=self._target,
                path=predictor_path,
                eval_metric=ag_kwargs.pop("eval_metric"),
            ).fit(
                train_data=df,
                **ag_kwargs,
            )
            fit_time = time.time() - start
            return self._parser.parse(
                run_id=config.run_id,
                predictor=predictor,
                eval_metric=config.autogluon_kwargs.get("eval_metric", ""),
                fit_time=fit_time,
                artifacts_dir=out_dir,
            )
        except Exception as e:
            return self._parser.parse_failure(config.run_id, str(e), out_dir)
```

**Step 6: Run tests**

```bash
pytest tests/execution/ -v
```
Expected: All tests PASS. (autogluon_runner has no unit tests yet — covered in integration later)

**Step 7: Commit**

```bash
git add src/execution/ tests/execution/
git commit -m "feat: execution layer — config mapper, AutoGluon runner, result parser"
```

---

## Task 5: Memory — RunStore

**Files:**
- Create: `src/memory/run_store.py`
- Test: `tests/memory/test_run_store.py`

**Step 1: Write the failing tests**

```python
# tests/memory/test_run_store.py
import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from src.memory.run_store import RunStore
from src.models.task import ExperimentPlan, RunConfig
from src.models.results import RunResult, RunEntry, RunDiagnostics


def make_run_entry(run_id: str, metric: float, status: str = "success") -> RunEntry:
    plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale="test"
    )
    config = RunConfig(run_id=run_id, node_id="n1",
                       autogluon_kwargs={}, data_path="d.csv", output_dir="out/")
    result = RunResult(run_id=run_id, status=status, primary_metric=metric if status == "success" else None,
                       leaderboard=[], best_model_name=None, fit_time_seconds=10.0,
                       artifacts_dir="out/", error=None, raw_info={})
    return RunEntry(run_id=run_id, node_id="n1", timestamp=datetime.now(),
                    config=config, result=result, diagnostics=RunDiagnostics(),
                    agent_rationale="test", agent_review="ok")


def test_run_store_append_and_get_history(tmp_path):
    store = RunStore(session_dir=str(tmp_path))
    entry = make_run_entry("run_0001", 0.85)
    store.append(entry)
    history = store.get_history()
    assert len(history) == 1
    assert history[0].run_id == "run_0001"


def test_run_store_persists_to_disk(tmp_path):
    store = RunStore(session_dir=str(tmp_path))
    store.append(make_run_entry("run_0001", 0.85))

    # Reload from disk
    store2 = RunStore(session_dir=str(tmp_path))
    history = store2.get_history()
    assert len(history) == 1
    assert history[0].run_id == "run_0001"


def test_run_store_incumbent_is_best_metric(tmp_path):
    store = RunStore(session_dir=str(tmp_path))
    store.append(make_run_entry("run_0001", 0.80))
    store.append(make_run_entry("run_0002", 0.87))
    store.append(make_run_entry("run_0003", 0.83))
    incumbent = store.get_incumbent(higher_is_better=True)
    assert incumbent.run_id == "run_0002"


def test_run_store_get_failed(tmp_path):
    store = RunStore(session_dir=str(tmp_path))
    store.append(make_run_entry("run_0001", 0.85, status="success"))
    store.append(make_run_entry("run_0002", 0.0, status="failed"))
    failed = store.get_failed()
    assert len(failed) == 1
    assert failed[0].run_id == "run_0002"


def test_run_store_multiple_appends_persist(tmp_path):
    store = RunStore(session_dir=str(tmp_path))
    for i in range(5):
        store.append(make_run_entry(f"run_{i:04d}", 0.80 + i * 0.01))
    history = store.get_history()
    assert len(history) == 5
```

**Step 2: Run to verify failure**

```bash
pytest tests/memory/test_run_store.py -v
```
Expected: `ImportError`

**Step 3: Implement src/memory/run_store.py**

```python
# src/memory/run_store.py
from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional
from src.models.results import RunEntry


class RunStore:
    """Append-only session experiment journal. Persists to JSONL."""

    FILENAME = "decisions.jsonl"

    def __init__(self, session_dir: str):
        self._path = Path(session_dir) / self.FILENAME
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: List[RunEntry] = self._load()

    def _load(self) -> List[RunEntry]:
        if not self._path.exists():
            return []
        entries = []
        with open(self._path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(RunEntry.model_validate_json(line))
                    except Exception:
                        pass
        return entries

    def append(self, entry: RunEntry) -> None:
        self._cache.append(entry)
        with open(self._path, "a") as f:
            f.write(entry.model_dump_json() + "\n")

    def get_history(self) -> List[RunEntry]:
        return list(self._cache)

    def get_incumbent(self, higher_is_better: bool = True) -> Optional[RunEntry]:
        successful = [e for e in self._cache if e.result.status == "success"
                      and e.result.primary_metric is not None]
        if not successful:
            return None
        return max(successful, key=lambda e: (
            e.result.primary_metric if higher_is_better else -e.result.primary_metric
        ))

    def get_failed(self) -> List[RunEntry]:
        return [e for e in self._cache if e.result.status == "failed"]

    def __len__(self) -> int:
        return len(self._cache)
```

**Step 4: Run tests**

```bash
pytest tests/memory/test_run_store.py -v
```
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/memory/run_store.py tests/memory/
git commit -m "feat: RunStore — append-only session experiment journal with JSONL persistence"
```

---

## Task 6: Orchestration — ExperimentTree

**Files:**
- Create: `src/orchestration/state.py`
- Test: `tests/orchestration/test_state.py`

**Step 1: Write the failing tests**

```python
# tests/orchestration/test_state.py
import pytest
from datetime import datetime
from src.models.task import ExperimentPlan
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus
from src.orchestration.state import ExperimentTree


def make_plan(metric: str = "roc_auc", rationale: str = "test") -> ExperimentPlan:
    return ExperimentPlan(
        eval_metric=metric, model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale=rationale
    )


def test_tree_add_root():
    tree = ExperimentTree()
    node = tree.add_root(make_plan(rationale="Root hypothesis A"))
    assert node.parent_id is None
    assert node.depth == 0
    assert node.edge_label is None
    assert len(tree.get_roots()) == 1


def test_tree_add_child():
    tree = ExperimentTree()
    root = tree.add_root(make_plan())
    child = tree.add_child(
        parent_id=root.node_id,
        plan=make_plan(metric="f1_macro"),
        edge_label="changed metric from roc_auc to f1_macro due to class imbalance"
    )
    assert child.parent_id == root.node_id
    assert child.depth == 1
    assert "f1_macro" in child.edge_label
    assert child.node_id in tree.get_node(root.node_id).children


def test_tree_get_path_to_root():
    tree = ExperimentTree()
    root = tree.add_root(make_plan())
    child = tree.add_child(root.node_id, make_plan(), "change 1")
    grandchild = tree.add_child(child.node_id, make_plan(), "change 2")
    path = tree.get_path_to_root(grandchild.node_id)
    assert len(path) == 3
    assert path[0].node_id == root.node_id
    assert path[-1].node_id == grandchild.node_id


def test_tree_get_leaves():
    tree = ExperimentTree()
    root = tree.add_root(make_plan())
    child1 = tree.add_child(root.node_id, make_plan(), "change A")
    child2 = tree.add_child(root.node_id, make_plan(), "change B")
    leaves = tree.get_leaves()
    leaf_ids = {n.node_id for n in leaves}
    assert child1.node_id in leaf_ids
    assert child2.node_id in leaf_ids
    assert root.node_id not in leaf_ids


def test_tree_serialize_preserves_edge_labels():
    tree = ExperimentTree()
    root = tree.add_root(make_plan())
    tree.add_child(root.node_id, make_plan(), "switched metric to f1_macro")
    data = tree.serialize()
    assert "nodes" in data
    assert "edges" in data
    # edge_label should appear in the serialized edges
    edge_labels = [e["label"] for e in data["edges"]]
    assert any("f1_macro" in lbl for lbl in edge_labels)


def test_tree_update_node_status():
    tree = ExperimentTree()
    node = tree.add_root(make_plan())
    tree.update_status(node.node_id, NodeStatus.SUCCESS)
    assert tree.get_node(node.node_id).status == NodeStatus.SUCCESS
```

**Step 2: Run to verify failure**

```bash
pytest tests/orchestration/test_state.py -v
```
Expected: `ImportError`

**Step 3: Implement src/orchestration/state.py**

```python
# src/orchestration/state.py
from __future__ import annotations
import uuid
from typing import Dict, List, Optional
from src.models.task import ExperimentPlan
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus


class ExperimentTree:
    """
    Graph structure for experiment lineage.
    Nodes = experiments. Edges = "what changed" (edge_label).
    Designed for future Graph RAG indexing.
    """

    def __init__(self):
        self._nodes: Dict[str, ExperimentNode] = {}

    def add_root(self, plan: ExperimentPlan, stage: NodeStage = NodeStage.WARMUP) -> ExperimentNode:
        node = ExperimentNode(
            node_id=str(uuid.uuid4())[:8],
            parent_id=None,
            children=[],
            edge_label=None,
            stage=stage,
            status=NodeStatus.PENDING,
            plan=plan,
            depth=0,
        )
        self._nodes[node.node_id] = node
        return node

    def add_child(
        self,
        parent_id: str,
        plan: ExperimentPlan,
        edge_label: str,
        stage: NodeStage = NodeStage.OPTIMIZE,
    ) -> ExperimentNode:
        parent = self._nodes[parent_id]
        node = ExperimentNode(
            node_id=str(uuid.uuid4())[:8],
            parent_id=parent_id,
            children=[],
            edge_label=edge_label,
            stage=stage,
            status=NodeStatus.PENDING,
            plan=plan,
            depth=parent.depth + 1,
        )
        self._nodes[node.node_id] = node
        # Update parent's children list (Pydantic model — replace)
        updated_parent = parent.model_copy(update={"children": parent.children + [node.node_id]})
        self._nodes[parent_id] = updated_parent
        return node

    def get_node(self, node_id: str) -> ExperimentNode:
        return self._nodes[node_id]

    def get_roots(self) -> List[ExperimentNode]:
        return [n for n in self._nodes.values() if n.parent_id is None]

    def get_leaves(self) -> List[ExperimentNode]:
        return [n for n in self._nodes.values() if not n.children]

    def get_path_to_root(self, node_id: str) -> List[ExperimentNode]:
        path = []
        current = self._nodes.get(node_id)
        while current:
            path.append(current)
            current = self._nodes.get(current.parent_id) if current.parent_id else None
        return list(reversed(path))

    def update_status(self, node_id: str, status: NodeStatus) -> None:
        node = self._nodes[node_id]
        self._nodes[node_id] = node.model_copy(update={"status": status})

    def update_node(self, node_id: str, **kwargs) -> None:
        node = self._nodes[node_id]
        self._nodes[node_id] = node.model_copy(update=kwargs)

    def serialize(self) -> dict:
        """Serialize tree to dict with explicit edges for Graph RAG compatibility."""
        nodes = []
        edges = []
        for node in self._nodes.values():
            nodes.append({
                "id": node.node_id,
                "stage": node.stage.value,
                "status": node.status.value,
                "depth": node.depth,
                "eval_metric": node.plan.eval_metric,
                "model_families": node.plan.model_families,
                "primary_metric": node.primary_metric(),
                "rationale": node.plan.rationale,
            })
            if node.parent_id:
                edges.append({
                    "source": node.parent_id,
                    "target": node.node_id,
                    "label": node.edge_label or "",
                })
        return {"nodes": nodes, "edges": edges}

    def __len__(self) -> int:
        return len(self._nodes)
```

**Step 4: Run tests**

```bash
pytest tests/orchestration/test_state.py -v
```
Expected: All 6 tests PASS.

**Step 5: Commit**

```bash
git add src/orchestration/state.py tests/orchestration/
git commit -m "feat: ExperimentTree — graph structure with edge labels for future Graph RAG"
```

---

## Task 7: Agent — Selector

**Files:**
- Create: `src/agents/selector.py`
- Test: `tests/agents/test_selector.py`

**Step 1: Write the failing tests**

```python
# tests/agents/test_selector.py
import pytest
import json
from unittest.mock import MagicMock
from src.agents.selector import SelectorAgent
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile


@pytest.fixture
def task():
    return TaskSpec(
        task_name="titanic", task_type="binary",
        data_path="data/titanic_train.csv", target_column="Survived",
        eval_metric="roc_auc", constraints={"max_time_per_run": 120},
        description="Predict passenger survival. Binary classification."
    )


@pytest.fixture
def profile():
    return DataProfile(
        n_rows=891, n_features=11,
        feature_types={"numeric": 5, "categorical": 6},
        target_distribution={"0": 549, "1": 342},
        class_balance_ratio=0.62,
        missing_rate=0.02,
        high_cardinality_cols=["Name", "Ticket"],
        suspected_leakage_cols=[],
        summary="891 rows, 11 features, 38% positive class, low missing rate."
    )


def test_selector_returns_experiment_plan(task, profile):
    mock_backend = MagicMock()
    mock_backend.complete.return_value = json.dumps({
        "eval_metric": "roc_auc",
        "model_families": ["GBM", "XGB"],
        "presets": "medium_quality",
        "time_limit": 120,
        "feature_policy": {"exclude_columns": [], "include_columns": []},
        "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0},
        "hyperparameters": None,
        "use_fit_extra": False,
        "rationale": "GBM is a strong default for tabular binary classification."
    })

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    hypothesis = "Use gradient boosting as a reliable baseline."
    plan = agent.select(hypothesis=hypothesis, task=task, data_profile=profile, history=[])

    assert isinstance(plan, ExperimentPlan)
    assert plan.eval_metric == "roc_auc"
    assert "GBM" in plan.model_families
    mock_backend.complete.assert_called_once()


def test_selector_passes_hypothesis_to_llm(task, profile):
    mock_backend = MagicMock()
    mock_backend.complete.return_value = json.dumps({
        "eval_metric": "f1_macro", "model_families": ["RF"],
        "presets": "medium_quality", "time_limit": 120,
        "feature_policy": {"exclude_columns": [], "include_columns": []},
        "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0},
        "hyperparameters": None, "use_fit_extra": False, "rationale": "test"
    })

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    agent.select(hypothesis="Use RF with f1_macro", task=task, data_profile=profile, history=[])

    call_args = mock_backend.complete.call_args
    messages = call_args[1].get("messages") or call_args[0][0]
    user_content = next(m["content"] for m in messages if m["role"] == "user")
    assert "RF" in user_content or "f1_macro" in user_content


def test_selector_handles_json_in_markdown_fence(task, profile):
    """LLM sometimes wraps JSON in markdown fences despite instructions."""
    mock_backend = MagicMock()
    mock_backend.complete.return_value = '```json\n{"eval_metric": "roc_auc", "model_families": ["GBM"], "presets": "medium_quality", "time_limit": 120, "feature_policy": {"exclude_columns": [], "include_columns": []}, "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0}, "hyperparameters": null, "use_fit_extra": false, "rationale": "test"}\n```'

    agent = SelectorAgent(llm=mock_backend, prompt_path="prompts/selector.md")
    plan = agent.select("test hypothesis", task, profile, [])
    assert plan.eval_metric == "roc_auc"
```

**Step 2: Run to verify failure**

```bash
pytest tests/agents/test_selector.py -v
```
Expected: `ImportError`

**Step 3: Implement src/agents/selector.py**

```python
# src/agents/selector.py
from __future__ import annotations
import json
import re
from pathlib import Path
from typing import List
from src.llm.backend import LLMBackend
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry


class SelectorAgent:
    """
    Turns a hypothesis (natural language) into a concrete ExperimentPlan.
    Calls the LLM with structured context, parses JSON response.
    """

    def __init__(self, llm: LLMBackend, prompt_path: str = "prompts/selector.md"):
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()

    def select(
        self,
        hypothesis: str,
        task: TaskSpec,
        data_profile: DataProfile,
        history: List[RunEntry],
    ) -> ExperimentPlan:
        user_msg = self._build_user_message(hypothesis, task, data_profile, history)
        response = self._llm.complete(
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
        )
        return self._parse_response(response)

    def _build_user_message(
        self,
        hypothesis: str,
        task: TaskSpec,
        profile: DataProfile,
        history: List[RunEntry],
    ) -> str:
        history_str = "None yet."
        if history:
            lines = []
            for e in history[-5:]:   # last 5 runs for context
                lines.append(
                    f"- run {e.run_id}: metric={e.result.primary_metric}, "
                    f"models={e.config.autogluon_kwargs.get('hyperparameters', {}).keys()}, "
                    f"review={e.agent_review[:80] if e.agent_review else ''}"
                )
            history_str = "\n".join(lines)

        return f"""## Hypothesis
{hypothesis}

## Task
Name: {task.task_name}
Type: {task.task_type}
Target: {task.target_column}
Initial metric: {task.eval_metric}
Description: {task.description}

## Data Profile
Rows: {profile.n_rows}
Features: {profile.n_features}
Feature types: {profile.feature_types}
Class balance ratio: {profile.class_balance_ratio:.2f}
Missing rate: {profile.missing_rate:.2f}
High cardinality columns: {profile.high_cardinality_cols}
Suspected leakage columns: {profile.suspected_leakage_cols}
Summary: {profile.summary}

## Prior Run History
{history_str}

Output the ExperimentPlan JSON now."""

    def _parse_response(self, response: str) -> ExperimentPlan:
        # Strip markdown fences if present
        response = response.strip()
        fence_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response)
        if fence_match:
            response = fence_match.group(1)

        data = json.loads(response)
        return ExperimentPlan(**data)
```

**Step 4: Run tests**

```bash
pytest tests/agents/test_selector.py -v
```
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/agents/selector.py tests/agents/
git commit -m "feat: SelectorAgent — hypothesis to ExperimentPlan via LLM with JSON parsing"
```

---

## Task 8: Agent — ExperimentManager (Warmup Loop)

**Files:**
- Create: `src/agents/manager.py`
- Test: `tests/agents/test_manager.py`

**Step 1: Write the failing tests**

```python
# tests/agents/test_manager.py
import pytest
import json
from unittest.mock import MagicMock, patch
from src.agents.manager import ExperimentManager, Action, ActionType
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunResult, RunEntry, RunDiagnostics
from src.models.nodes import ExperimentNode, NodeStage, NodeStatus, SearchContext
from datetime import datetime


def make_search_context(stage: str = "warmup", budget: int = 5) -> SearchContext:
    task = TaskSpec(task_name="test", task_type="binary", data_path="d.csv",
                    target_column="label", eval_metric="roc_auc",
                    constraints={}, description="test task")
    profile = DataProfile(n_rows=891, n_features=11, class_balance_ratio=0.62,
                          missing_rate=0.02)
    plan = ExperimentPlan(eval_metric="roc_auc", model_families=["GBM"],
                          presets="medium_quality", time_limit=120,
                          feature_policy={"exclude_columns": [], "include_columns": []},
                          validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
                          hyperparameters=None, use_fit_extra=False, rationale="test")
    node = ExperimentNode(node_id="n1", plan=plan, stage=NodeStage.WARMUP,
                          status=NodeStatus.PENDING, depth=0)
    return SearchContext(task=task, data_profile=profile, history=[], incumbent=None,
                         current_node=node, stage=stage, budget_remaining=budget, budget_used=0)


def test_manager_select_action_in_warmup():
    """In warmup with no result on current node, manager should select."""
    mock_selector = MagicMock()
    mock_selector.select.return_value = MagicMock(spec=ExperimentPlan)
    manager = ExperimentManager(selector=mock_selector, refiner=None)
    context = make_search_context(stage="warmup")
    action = manager.next_action(context)
    assert action.type == ActionType.SELECT


def test_manager_stop_when_no_budget():
    manager = ExperimentManager(selector=MagicMock(), refiner=None)
    context = make_search_context(stage="optimize", budget=0)
    action = manager.next_action(context)
    assert action.type == ActionType.STOP


def test_manager_action_has_plan():
    mock_plan = ExperimentPlan(
        eval_metric="roc_auc", model_families=["GBM"], presets="medium_quality",
        time_limit=120, feature_policy={"exclude_columns": [], "include_columns": []},
        validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
        hyperparameters=None, use_fit_extra=False, rationale="selected"
    )
    mock_selector = MagicMock()
    mock_selector.select.return_value = mock_plan
    manager = ExperimentManager(selector=mock_selector, refiner=None)
    context = make_search_context(stage="warmup")
    action = manager.next_action(context)
    assert action.plan is not None
    assert action.plan.eval_metric == "roc_auc"
```

**Step 2: Run to verify failure**

```bash
pytest tests/agents/test_manager.py -v
```
Expected: `ImportError`

**Step 3: Implement src/agents/manager.py**

```python
# src/agents/manager.py
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from src.models.task import ExperimentPlan
from src.models.nodes import SearchContext


class ActionType(str, Enum):
    SELECT = "select"     # warmup: turn hypothesis into plan
    REFINE = "refine"     # optimize: improve the incumbent
    STOP = "stop"         # budget exhausted or converged
    DEBUG = "debug"       # fix a failed run (Phase 3)


class Action(BaseModel):
    type: ActionType
    plan: Optional[ExperimentPlan] = None
    reason: str = ""


class ExperimentManager:
    """
    Top-level decision router. Decides which sub-agent to invoke
    based on stage and context, returns an Action.
    """

    def __init__(self, selector, refiner):
        self._selector = selector   # SelectorAgent
        self._refiner = refiner     # RefinerAgent (Phase 3, may be None)

    def next_action(self, context: SearchContext) -> Action:
        # Always stop if budget is gone
        if context.budget_remaining <= 0:
            return Action(type=ActionType.STOP, reason="Budget exhausted")

        if context.stage == "warmup":
            return self._warmup_action(context)
        elif context.stage == "optimize":
            return self._optimize_action(context)
        else:
            return Action(type=ActionType.STOP, reason=f"Unknown stage: {context.stage}")

    def _warmup_action(self, context: SearchContext) -> Action:
        # In warmup, the current node has a hypothesis but no result yet.
        # Selector turns the hypothesis into a concrete ExperimentPlan.
        plan = self._selector.select(
            hypothesis=context.current_node.plan.rationale,
            task=context.task,
            data_profile=context.data_profile,
            history=context.history,
        )
        return Action(type=ActionType.SELECT, plan=plan)

    def _optimize_action(self, context: SearchContext) -> Action:
        if self._refiner is None:
            # Phase 1: no refiner yet — repeat incumbent config with minor time bump
            incumbent_plan = context.incumbent.config.autogluon_kwargs if context.incumbent else None
            if incumbent_plan is None:
                return Action(type=ActionType.STOP, reason="No incumbent to refine")
            # Simple rule-based refinement: increase time_limit by 50%
            current = context.current_node.plan
            new_plan = current.model_copy(update={
                "time_limit": int(current.time_limit * 1.5),
                "rationale": "Phase 1 fallback: increase time_limit by 50% to allow more training"
            })
            return Action(type=ActionType.REFINE, plan=new_plan,
                          reason="No refiner available — applying time budget increase")

        plan = self._refiner.refine(context=context)
        return Action(type=ActionType.REFINE, plan=plan)
```

**Step 4: Run tests**

```bash
pytest tests/agents/test_manager.py -v
```
Expected: All 3 tests PASS.

**Step 5: Commit**

```bash
git add src/agents/manager.py tests/agents/test_manager.py
git commit -m "feat: ExperimentManager — action routing for warmup/optimize stages"
```

---

## Task 9: Wire It — main.py

**Files:**
- Create: `src/session.py` — session orchestrator (cleaner than dumping everything in main.py)
- Create: `main.py` — thin entrypoint
- Test: manual end-to-end run on Titanic data

**Step 1: Implement src/session.py**

```python
# src/session.py
from __future__ import annotations
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import yaml
import pandas as pd

from src.llm.backend import create_backend
from src.agents.manager import ExperimentManager, ActionType
from src.agents.selector import SelectorAgent
from src.execution.config_mapper import ConfigMapper
from src.execution.autogluon_runner import AutoGluonRunner
from src.execution.result_parser import ResultParser
from src.memory.run_store import RunStore
from src.orchestration.state import ExperimentTree
from src.models.task import TaskSpec, ExperimentPlan
from src.models.results import DataProfile, RunEntry, RunDiagnostics
from src.models.nodes import (
    ExperimentNode, NodeStage, NodeStatus, SearchContext, CaseEntry
)


def _bucket_rows(n: int) -> str:
    if n < 5000: return "small"
    if n < 100000: return "medium"
    return "large"


def _bucket_features(n: int) -> str:
    return "narrow" if n < 20 else "wide"


def _balance_label(ratio: float) -> str:
    if ratio > 0.6: return "balanced"
    if ratio > 0.3: return "moderate_imbalance"
    return "severe_imbalance"


def profile_data(data_path: str, target_column: str) -> DataProfile:
    df = pd.read_csv(data_path)
    n_rows, n_cols = df.shape
    n_features = n_cols - 1

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(exclude="number").columns.tolist()

    target = df[target_column]
    vc = target.value_counts()
    balance_ratio = vc.min() / vc.max() if len(vc) > 1 else 1.0
    missing_rate = df.isnull().mean().mean()
    high_card = [c for c in cat_cols if c != target_column and df[c].nunique() > 20]

    return DataProfile(
        n_rows=n_rows,
        n_features=n_features,
        feature_types={"numeric": len(numeric_cols), "categorical": len(cat_cols)},
        target_distribution=vc.to_dict(),
        class_balance_ratio=round(float(balance_ratio), 3),
        missing_rate=round(float(missing_rate), 3),
        high_cardinality_cols=high_card,
        suspected_leakage_cols=[],
        summary=(
            f"{n_rows} rows, {n_features} features, "
            f"class balance ratio={balance_ratio:.2f}, "
            f"missing rate={missing_rate:.2f}"
        )
    )


def load_seed_ideas(path: str = "configs/seed_ideas.json") -> list[dict]:
    if Path(path).exists():
        with open(path) as f:
            return json.load(f)
    return []


class AgenticMLSession:
    """Orchestrates a full experiment session through all 4 layers."""

    def __init__(self, project_config_path: str, search_config_path: str):
        load_dotenv()
        with open(project_config_path) as f:
            self._project = yaml.safe_load(f)
        with open(search_config_path) as f:
            self._search = yaml.safe_load(f)

        # Build task spec
        tc = self._project["task"]
        self._task = TaskSpec(
            task_name=tc["name"],
            task_type=tc["type"],
            data_path=tc["data_path"],
            target_column=tc["target_column"],
            eval_metric=tc["eval_metric"],
            constraints=tc.get("constraints", {}),
            description=tc.get("description", ""),
        )

        # Session directory
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        exp_dir = self._project["session"]["experiments_dir"]
        self._session_dir = os.path.join(exp_dir, f"{ts}_{self._task.task_name}")
        Path(self._session_dir).mkdir(parents=True, exist_ok=True)

        # LLM backend
        llm_cfg = self._project["llm"]
        self._llm = create_backend(
            provider=llm_cfg["provider"],
            model=llm_cfg["model"],
            api_key=os.environ.get(
                "ANTHROPIC_API_KEY" if llm_cfg["provider"] == "anthropic" else "OPENAI_API_KEY"
            )
        )

        # Layers
        self._run_store = RunStore(session_dir=self._session_dir)
        self._tree = ExperimentTree()
        self._mapper = ConfigMapper()
        self._runner = AutoGluonRunner(target_column=self._task.target_column)
        selector = SelectorAgent(llm=self._llm)
        self._manager = ExperimentManager(selector=selector, refiner=None)

        self._search_cfg = self._search["search"]
        self._higher_is_better = self._search_cfg["higher_is_better"].get(
            self._task.eval_metric, True
        )

    def run(self) -> None:
        print(f"\n{'='*60}")
        print(f"Session: {self._session_dir}")
        print(f"Task: {self._task.task_name} | Metric: {self._task.eval_metric}")
        print(f"{'='*60}\n")

        # Profile data
        print("[1/4] Profiling data...")
        profile = profile_data(self._task.data_path, self._task.target_column)
        self._save_json(profile.model_dump(), "data_profile.json")
        print(f"      {profile.summary}")

        # Ideation — use seed ideas as initial hypotheses
        print("\n[2/4] Ideating candidates...")
        seeds = load_seed_ideas()
        num_candidates = min(self._search_cfg["num_candidates"], len(seeds)) if seeds else 1
        hypotheses = seeds[:num_candidates] if seeds else [
            {"hypothesis": "Use GBM as a baseline.", "rationale": "Default strong baseline."}
        ]

        root_nodes = []
        for h in hypotheses:
            plan = ExperimentPlan(
                eval_metric=self._task.eval_metric,
                model_families=["GBM"],
                presets="medium_quality",
                time_limit=self._task.constraints.get("max_time_per_run", 120),
                feature_policy={"exclude_columns": [], "include_columns": []},
                validation_policy={"holdout_frac": 0.2, "num_bag_folds": 0},
                hyperparameters=None,
                use_fit_extra=False,
                rationale=h["hypothesis"],
            )
            node = self._tree.add_root(plan, stage=NodeStage.WARMUP)
            root_nodes.append(node)
            print(f"      Candidate: {h['hypothesis'][:80]}")

        self._save_json(hypotheses, "initial_hypotheses.json")

        # Warm-up loop
        print(f"\n[3/4] Warm-up: running {len(root_nodes)} candidates...")
        run_counter = 0
        for node in root_nodes:
            run_counter += 1
            print(f"\n  Run {run_counter}: node={node.node_id}, hypothesis='{node.plan.rationale[:60]}...'")

            context = self._build_context(node, stage="warmup",
                                          budget_remaining=self._search_cfg["max_optimize_iterations"],
                                          profile=profile)
            action = self._manager.next_action(context)

            if action.type == ActionType.STOP:
                print(f"  Stopped: {action.reason}")
                break

            run_id = f"run_{run_counter:04d}"
            run_dir = os.path.join(self._session_dir, "runs", run_id)
            config = self._mapper.plan_to_config(action.plan, run_id, node.node_id,
                                                  self._task.data_path, run_dir)
            self._tree.update_node(node.node_id, config=config, status=NodeStatus.RUNNING,
                                   plan=action.plan)

            print(f"  Executing: metric={action.plan.eval_metric}, "
                  f"models={action.plan.model_families}, time_limit={action.plan.time_limit}s")
            result = self._runner.run(config)
            self._tree.update_status(node.node_id,
                                      NodeStatus.SUCCESS if result.status == "success" else NodeStatus.FAILED)

            diagnostics = RunDiagnostics(
                data_profile_ref=os.path.join(self._session_dir, "data_profile.json"),
                change_description="Initial warmup run",
            )
            entry = RunEntry(
                run_id=run_id, node_id=node.node_id, config=config,
                result=result, diagnostics=diagnostics,
                agent_rationale=action.plan.rationale,
            )
            self._run_store.append(entry)
            self._tree.update_node(node.node_id, entry=entry)

            if result.status == "success":
                print(f"  Result: {self._task.eval_metric}={result.primary_metric:.4f}, "
                      f"best_model={result.best_model_name}, fit_time={result.fit_time_seconds:.1f}s")
            else:
                print(f"  Failed: {result.error}")

        # Optimization loop
        max_opt = self._search_cfg["max_optimize_iterations"]
        incumbent = self._run_store.get_incumbent(higher_is_better=self._higher_is_better)
        if not incumbent:
            print("\n[4/4] No valid runs — skipping optimization.")
        else:
            print(f"\n[4/4] Optimization: incumbent={incumbent.run_id}, "
                  f"metric={incumbent.result.primary_metric:.4f}")
            plateau_count = 0
            for opt_i in range(max_opt):
                run_counter += 1
                budget_remaining = max_opt - opt_i - 1
                print(f"\n  Opt run {opt_i+1}/{max_opt}: budget_remaining={budget_remaining}")

                incumbent_node = self._tree.get_node(incumbent.node_id)
                context = self._build_context(incumbent_node, stage="optimize",
                                               budget_remaining=budget_remaining, profile=profile)
                action = self._manager.next_action(context)

                if action.type == ActionType.STOP:
                    print(f"  Stopped: {action.reason}")
                    break

                # Compute edge label
                prev_plan = incumbent_node.plan
                changes = []
                if action.plan.eval_metric != prev_plan.eval_metric:
                    changes.append(f"metric {prev_plan.eval_metric}→{action.plan.eval_metric}")
                if action.plan.time_limit != prev_plan.time_limit:
                    changes.append(f"time_limit {prev_plan.time_limit}→{action.plan.time_limit}s")
                if action.plan.model_families != prev_plan.model_families:
                    changes.append(f"models {prev_plan.model_families}→{action.plan.model_families}")
                edge_label = ", ".join(changes) if changes else "minor refinement"

                child_node = self._tree.add_child(
                    parent_id=incumbent.node_id,
                    plan=action.plan,
                    edge_label=edge_label,
                    stage=NodeStage.OPTIMIZE,
                )

                run_id = f"run_{run_counter:04d}"
                run_dir = os.path.join(self._session_dir, "runs", run_id)
                config = self._mapper.plan_to_config(action.plan, run_id, child_node.node_id,
                                                      self._task.data_path, run_dir)
                self._tree.update_node(child_node.node_id, config=config,
                                       status=NodeStatus.RUNNING, plan=action.plan)

                print(f"  Executing: {edge_label}, time_limit={action.plan.time_limit}s")
                result = self._runner.run(config)
                self._tree.update_status(child_node.node_id,
                                          NodeStatus.SUCCESS if result.status == "success" else NodeStatus.FAILED)

                parent_metric = incumbent.result.primary_metric or 0.0
                child_metric = result.primary_metric
                delta = None
                if child_metric is not None:
                    delta = child_metric - parent_metric if self._higher_is_better else parent_metric - child_metric

                diagnostics = RunDiagnostics(
                    data_profile_ref=os.path.join(self._session_dir, "data_profile.json"),
                    metric_vs_parent=delta,
                    change_description=edge_label,
                )
                entry = RunEntry(
                    run_id=run_id, node_id=child_node.node_id, config=config,
                    result=result, diagnostics=diagnostics,
                    agent_rationale=action.plan.rationale,
                )
                self._run_store.append(entry)
                self._tree.update_node(child_node.node_id, entry=entry)

                if result.status == "success":
                    print(f"  Result: {self._task.eval_metric}={child_metric:.4f} "
                          f"(delta={delta:+.4f})" if delta is not None else f"  Result: {child_metric}")
                    # Accept or reject
                    if delta is not None and delta > self._search_cfg["plateau_threshold"]:
                        print(f"  ✓ Accepted as new incumbent")
                        incumbent = entry
                        plateau_count = 0
                    else:
                        print(f"  ✗ Rejected (no improvement)")
                        self._tree.update_status(child_node.node_id, NodeStatus.REJECTED)
                        plateau_count += 1
                        if plateau_count >= self._search_cfg["plateau_patience"]:
                            print(f"  Plateau detected ({plateau_count} consecutive non-improvements) — stopping")
                            break
                else:
                    print(f"  Failed: {result.error}")
                    plateau_count += 1

        # Save tree and summary
        tree_data = self._tree.serialize()
        self._save_json(tree_data, "tree.json")

        final_incumbent = self._run_store.get_incumbent(higher_is_better=self._higher_is_better)
        print(f"\n{'='*60}")
        print(f"Session complete. Runs: {len(self._run_store)}")
        if final_incumbent:
            print(f"Best: {self._task.eval_metric}={final_incumbent.result.primary_metric:.4f} "
                  f"(run={final_incumbent.run_id}, model={final_incumbent.result.best_model_name})")
        print(f"Tree: {len(self._tree)} nodes, saved to {self._session_dir}/tree.json")
        print(f"{'='*60}\n")

    def _build_context(self, node: ExperimentNode, stage: str,
                        budget_remaining: int, profile: DataProfile) -> SearchContext:
        history = self._run_store.get_history()
        incumbent_entry = self._run_store.get_incumbent(self._higher_is_better)
        tree_data = self._tree.serialize()
        return SearchContext(
            task=self._task,
            data_profile=profile,
            history=history,
            incumbent=incumbent_entry,
            current_node=node,
            tree_summary={"n_nodes": len(self._tree), "stage": stage},
            similar_cases=[],   # Phase 2: CaseStore retrieval
            failed_attempts=self._run_store.get_failed(),
            stage=stage,
            budget_remaining=budget_remaining,
            budget_used=len(history),
        )

    def _save_json(self, data: dict | list, filename: str) -> None:
        path = os.path.join(self._session_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
```

**Step 2: Implement main.py**

```python
# main.py
import argparse
from src.session import AgenticMLSession


def main():
    parser = argparse.ArgumentParser(description="Hybrid Agentic ML Framework")
    parser.add_argument("--config", default="configs/project.yaml", help="Project config path")
    parser.add_argument("--search", default="configs/search.yaml", help="Search config path")
    args = parser.parse_args()

    session = AgenticMLSession(
        project_config_path=args.config,
        search_config_path=args.search,
    )
    session.run()


if __name__ == "__main__":
    main()
```

**Step 3: Run the full loop**

First ensure you have the demo data:
```bash
mkdir -p scripts data
python -c "
import pandas as pd
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
df = pd.read_csv(url)
df.to_csv('data/titanic_train.csv', index=False)
print(f'Saved {len(df)} rows to data/titanic_train.csv')
"
```

Then run the loop:
```bash
python main.py --config configs/project.yaml --search configs/search.yaml
```

Expected output (approximate):
```
============================================================
Session: experiments/2026-03-17_XXXXXX_demo-titanic
Task: demo-titanic | Metric: roc_auc
============================================================

[1/4] Profiling data...
      891 rows, 11 features, class balance ratio=0.62, missing rate=0.02

[2/4] Ideating candidates...
      Candidate: Start with gradient boosting (GBM) on all features...
      Candidate: Try random forest with f1_macro given class imbalance...
      Candidate: Use a diverse model mix (GBM + XGB + CAT)...

[3/4] Warm-up: running 3 candidates...

  Run 1: node=XXXXXXXX, hypothesis='Start with gradient boosting...'
  Executing: metric=roc_auc, models=['GBM', 'XGB'], time_limit=120s
  Result: roc_auc=0.8612, best_model=WeightedEnsemble_L2, fit_time=87.3s

  Run 2: node=XXXXXXXX, hypothesis='Try random forest with f1_macro...'
  Executing: metric=f1_macro, models=['RF', 'XT'], time_limit=120s
  Result: f1_macro=0.7234, best_model=RandomForestGini, fit_time=45.2s

  Run 3: node=XXXXXXXX, hypothesis='Use a diverse model mix...'
  Executing: metric=accuracy, models=['GBM', 'XGB', 'CAT'], time_limit=120s
  Result: accuracy=0.8361, best_model=LightGBM, fit_time=95.1s

[4/4] Optimization: incumbent=run_0001, metric=0.8612
  Opt run 1/5: budget_remaining=4
  Executing: time_limit 120→180s, minor refinement
  Result: roc_auc=0.8634 (delta=+0.0022)
  ✓ Accepted as new incumbent

  ...

============================================================
Session complete. Runs: 6
Best: roc_auc=0.8634 (run=run_0004, model=WeightedEnsemble_L2)
Tree: 7 nodes, saved to experiments/.../tree.json
============================================================
```

**Step 4: Verify tree.json was saved**

```bash
cat experiments/$(ls -t experiments/ | head -1)/tree.json | python -m json.tool | head -40
```
Expected: JSON with `nodes` array and `edges` array with `label` fields.

**Step 5: Commit**

```bash
git add src/session.py main.py
git commit -m "feat: session orchestrator + main.py — full 4-layer loop running end-to-end"
```

---

## Task 10: LLM Backend — OpenAI Provider

**Files:**
- Create: `src/llm/providers/openai.py`
- Test: `tests/llm/test_backend.py` (add tests)

**Step 1: Add the failing test**

```python
# Add to tests/llm/test_backend.py
from src.llm.providers.openai import OpenAIBackend

def test_openai_backend_implements_protocol():
    backend = OpenAIBackend(model="gpt-4o-mini", api_key="test-key")
    assert isinstance(backend, LLMBackend)

def test_create_backend_openai():
    backend = create_backend(provider="openai", model="gpt-4o-mini", api_key="test-key")
    assert isinstance(backend, OpenAIBackend)

def test_openai_backend_complete(monkeypatch):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content='{"eval_metric": "roc_auc"}'))]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response

    backend = OpenAIBackend(model="gpt-4o-mini", api_key="test-key")
    backend._client = mock_client
    result = backend.complete(messages=[{"role": "user", "content": "test"}])
    assert result == '{"eval_metric": "roc_auc"}'
```

**Step 2: Implement src/llm/providers/openai.py**

```python
# src/llm/providers/openai.py
from __future__ import annotations
from typing import Optional
from openai import OpenAI


class OpenAIBackend:
    def __init__(self, model: str, api_key: str):
        self._model = model
        self._client = OpenAI(api_key=api_key)

    def complete(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        response_format: Optional[dict] = None,
    ) -> str:
        kwargs = dict(model=self._model, messages=messages, temperature=temperature)
        if response_format:
            kwargs["response_format"] = response_format
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
```

**Step 3: Run all tests**

```bash
pytest tests/ -v
```
Expected: All tests PASS.

**Step 4: Commit**

```bash
git add src/llm/providers/openai.py tests/llm/test_backend.py
git commit -m "feat: OpenAI provider + factory routing complete"
```

---

## Phase 1 Complete — Run Full Test Suite

```bash
pytest tests/ -v --tb=short
```

Expected: All tests pass. Then run one final end-to-end smoke test:

```bash
python main.py
```

Verify:
- `experiments/` contains a new session directory
- Session directory has `data_profile.json`, `initial_hypotheses.json`, `decisions.jsonl`, `tree.json`
- `decisions.jsonl` has 3+ lines (one per run)
- `tree.json` has nodes with `edges` containing `label` fields
- Console shows final incumbent metric

---

## Phase 2 Overview: Deepen Search + Memory

*(Detailed tasks to be written after Phase 1 is working)*

**Goal:** Agent makes informed, context-grounded decisions using full warm-up→optimize scheduler, CaseStore retrieval, and rich SearchContext.

**Tasks at high level:**
1. `src/orchestration/scheduler.py` — warm-up/optimize stage transitions, plateau detection, budget tracking as a proper class (extracted from session.py)
2. `src/orchestration/accept_reject.py` — direction-aware incumbent comparison as a standalone class
3. `src/agents/ideator.py` — LLM-based hypothesis generation from DataProfile + case retrieval + seed ideas
4. `src/memory/case_store.py` — JSONL persistence + load/add/search interface
5. `src/memory/retrieval.py` — cosine similarity over task trait vectors (sklearn)
6. `src/memory/distiller.py` — LLM-assisted session → CaseEntry at session end
7. `src/memory/context_builder.py` — assembles full SearchContext from all sources
8. Refactor `session.py` to use Scheduler and ContextBuilder instead of inline logic

---

## Phase 3 Overview: Deepen Agent Reasoning

*(Detailed tasks to be written after Phase 2 is working)*

**Goal:** Agent reasons well about what to change and why, based on diagnostics.

**Tasks at high level:**
1. `src/agents/refiner.py` — LLM-based config refinement (replaces the rule-based fallback in manager.py)
2. `src/agents/reviewer.py` — run quality assessment (overfitting, imbalance, leakage flags)
3. Richer `result_parser.py` — feature importances via `predictor.feature_importance()`, overfitting gap from train score
4. `comparison_to_parent` in `RunDiagnostics` — populated by session.py after each run
5. Full `ExperimentTree.serialize()` with edge labels written at session end
6. `notebooks/01_inspect_run.ipynb` — interactive run inspection
7. `notebooks/02_replay_decision.ipynb` — replay agent reasoning with different context

---

*Plan saved. Phase 1 covers all 4 layers end-to-end with TDD. Each task is ~30-60 minutes of focused work.*
