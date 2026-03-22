# Phase 3: Principled Refinement — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the generic "refine this config" selector call in the optimize loop with a dedicated RefinerAgent that receives the incumbent's full config, leaderboard, and overfitting diagnostics — producing targeted, evidence-driven refinements.

**Architecture:** ResultParser gains overfitting_gap (score_train − score_val for best model via AutoGluon's extra_info leaderboard). session.py populates metric_vs_parent after each run. RefinerAgent is a new agent in src/agents/ that takes incumbent RunEntry + history → ExperimentPlan, replacing the SelectorAgent call in the optimize loop.

**Tech Stack:** AutoGluon leaderboard(extra_info=True), Pydantic RunDiagnostics, same LLM retry + fence-strip pattern as IdeatorAgent/SelectorAgent.

---

### Task 1: Capture score_train in ModelEntry and compute overfitting_gap in ResultParser

**Files:**
- Modify: `src/models/results.py:8-13` — add `score_train` to ModelEntry
- Modify: `src/execution/result_parser.py:26-58` — use `leaderboard(extra_info=True)`, compute overfitting_gap
- Test: `tests/test_result_parser.py`

**Step 1: Write the failing test**

```python
# tests/test_result_parser.py
import pytest
from unittest.mock import MagicMock
import pandas as pd
from src.execution.result_parser import ResultParser

def _make_predictor(val_scores, train_scores, best_model="WeightedEnsemble_L2"):
    lb_basic = pd.DataFrame({
        "model": ["WeightedEnsemble_L2", "GBM"],
        "score_val": val_scores,
        "fit_time": [10.0, 8.0],
        "pred_time": [0.1, 0.1],
        "stack_level": [2, 1],
    })
    lb_extra = lb_basic.copy()
    lb_extra["score_train"] = train_scores
    p = MagicMock()
    p.leaderboard.side_effect = lambda extra_info=False: lb_extra if extra_info else lb_basic
    p.model_best = best_model
    return p

def test_overfitting_gap_computed():
    predictor = _make_predictor(
        val_scores=[0.87, 0.85],
        train_scores=[0.95, 0.93],
    )
    result = ResultParser.from_predictor(predictor, "run_0001", 10.0, "/tmp", 0.87)
    assert result.leaderboard[0].score_train == pytest.approx(0.95)
    assert result.diagnostics_overfitting_gap == pytest.approx(0.95 - 0.87)

def test_overfitting_gap_none_when_extra_info_fails():
    predictor = _make_predictor([0.87, 0.85], [0.95, 0.93])
    predictor.leaderboard.side_effect = Exception("no extra info")
    result = ResultParser.from_predictor(predictor, "run_0001", 10.0, "/tmp", 0.87)
    assert result.diagnostics_overfitting_gap is None
```

**Step 2: Run test to verify it fails**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_result_parser.py -v 2>&1 | tail -20
```

Expected: AttributeError — `score_train` not on ModelEntry, `diagnostics_overfitting_gap` not on RunResult.

**Step 3: Add `score_train` to ModelEntry in `src/models/results.py`**

```python
class ModelEntry(BaseModel):
    model_name: str
    score_val: float
    fit_time: float
    pred_time: float
    stack_level: int = 1
    score_train: Optional[float] = None   # ← add this line
```

Also add `diagnostics_overfitting_gap: Optional[float] = None` to `RunResult`:

```python
class RunResult(BaseModel):
    run_id: str
    status: str
    primary_metric: Optional[float] = None
    leaderboard: List[ModelEntry] = Field(default_factory=list)
    best_model_name: Optional[str] = None
    fit_time_seconds: float = 0.0
    artifacts_dir: str = ""
    error: Optional[str] = None
    raw_info: Dict[str, Any] = Field(default_factory=dict)
    diagnostics_overfitting_gap: Optional[float] = None   # ← add this line
```

**Step 4: Update `ResultParser.from_predictor` in `src/execution/result_parser.py`**

Replace the leaderboard block with:

```python
@staticmethod
def from_predictor(
    predictor: Any,
    run_id: str,
    fit_time: float,
    artifacts_dir: str,
    primary_metric_value: float,
) -> RunResult:
    leaderboard_entries = []
    overfitting_gap = None
    try:
        lb = predictor.leaderboard(extra_info=True)
        best_row = lb.iloc[0]
        score_train = float(best_row["score_train"]) if "score_train" in lb.columns else None
        score_val = float(best_row["score_val"])
        if score_train is not None:
            overfitting_gap = round(score_train - score_val, 4)
        for row in lb.itertuples():
            leaderboard_entries.append(ModelEntry(
                model_name=row.model,
                score_val=row.score_val,
                fit_time=row.fit_time,
                pred_time=row.pred_time,
                stack_level=getattr(row, "stack_level", 1),
                score_train=getattr(row, "score_train", None),
            ))
    except Exception:
        try:
            lb = predictor.leaderboard()
            for row in lb.itertuples():
                leaderboard_entries.append(ModelEntry(
                    model_name=row.model,
                    score_val=row.score_val,
                    fit_time=row.fit_time,
                    pred_time=row.pred_time,
                    stack_level=getattr(row, "stack_level", 1),
                ))
        except Exception:
            pass

    return RunResult(
        run_id=run_id,
        status="success",
        primary_metric=primary_metric_value,
        leaderboard=leaderboard_entries,
        best_model_name=getattr(predictor, "model_best", None),
        fit_time_seconds=fit_time,
        artifacts_dir=artifacts_dir,
        error=None,
        raw_info={},
        diagnostics_overfitting_gap=overfitting_gap,
    )
```

**Step 5: Run tests**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_result_parser.py -v 2>&1 | tail -20
```

Expected: PASS

**Step 6: Commit**

```bash
cd "/home/tough/Agentic ML" && git add src/models/results.py src/execution/result_parser.py tests/test_result_parser.py && git commit -m "feat: capture score_train + overfitting_gap in ResultParser"
```

---

### Task 2: Populate metric_vs_parent and overfitting_gap in RunDiagnostics

The `RunDiagnostics` fields are populated in `session.py:execute_node`, not in ResultParser (since metric_vs_parent needs the parent node's metric).

**Files:**
- Modify: `src/session.py` — in `execute_node`, after building RunEntry, compute diagnostics

**Step 1: Find execute_node in session.py**

Read `src/session.py` lines 180-220 to locate where `RunEntry` is constructed.

**Step 2: Add diagnostics population after RunEntry is built**

After the line `entry = RunEntry(...)` in `execute_node`, add:

```python
# Populate diagnostics from result and parent
if result.status == "success":
    parent_node = self.tree.get_node(node.parent_id) if node.parent_id else None
    parent_metric = parent_node.primary_metric() if parent_node and parent_node.has_result() else None
    metric_vs_parent = None
    if parent_metric is not None and result.primary_metric is not None:
        metric_vs_parent = round(result.primary_metric - parent_metric, 4)
    entry.diagnostics = RunDiagnostics(
        overfitting_gap=result.diagnostics_overfitting_gap,
        metric_vs_parent=metric_vs_parent,
    )
```

Also add `RunDiagnostics` to the imports at the top of session.py if not already present.

**Step 3: Verify no test regressions**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/ -v 2>&1 | tail -20
```

**Step 4: Commit**

```bash
cd "/home/tough/Agentic ML" && git add src/session.py && git commit -m "feat: populate overfitting_gap and metric_vs_parent in RunDiagnostics"
```

---

### Task 3: Write the RefinerAgent prompt

**Files:**
- Create: `prompts/refiner.md`

**Step 1: Create the prompt**

```markdown
You are a machine learning experiment refiner. You receive the current best (incumbent) experiment config, its results, and the full session history. Your job is to propose ONE targeted improvement as a concrete ExperimentPlan JSON object.

## Decision Rules
- If overfitting_gap > 0.05: reduce model complexity (fewer families, lower time_limit, add regularisation via hyperparameters) or increase holdout_frac.
- If all prior runs use the same model families: diversify (try CAT, NN_TORCH, or FASTAI).
- If metric has plateaued for 2+ runs: change validation strategy (increase num_bag_folds from 0 to 5, or switch presets from medium_quality to high_quality).
- If a run failed (primary_metric=None): avoid the same model families from that run.
- Otherwise: try adding one model family that hasn't appeared in the top-3 leaderboard.

## Output Format
Respond with ONLY a valid JSON object matching this schema exactly:
{
  "eval_metric": "<string>",
  "model_families": ["<string>", ...],
  "presets": "<string>",
  "time_limit": <int>,
  "feature_policy": {"exclude_columns": [], "include_columns": []},
  "validation_policy": {"holdout_frac": <float>, "num_bag_folds": <int>},
  "hyperparameters": null,
  "use_fit_extra": false,
  "rationale": "<one sentence explaining the ONE change you made and why>"
}

No markdown fences. No explanation outside the JSON.
```

**Step 2: Commit**

```bash
cd "/home/tough/Agentic ML" && git add prompts/refiner.md && git commit -m "feat: add RefinerAgent prompt with decision rules"
```

---

### Task 4: Implement RefinerAgent

**Files:**
- Create: `src/agents/refiner.py`
- Test: `tests/test_refiner.py`

**Step 1: Write the failing tests**

```python
# tests/test_refiner.py
import json
import pytest
from unittest.mock import MagicMock
from src.agents.refiner import RefinerAgent
from src.models.task import ExperimentPlan, TaskSpec, RunConfig
from src.models.results import RunResult, RunEntry, RunDiagnostics


def _make_task():
    return TaskSpec(
        task_name="demo",
        task_type="binary",
        data_path="data/train.csv",
        target_column="Survived",
        eval_metric="roc_auc",
    )


def _make_incumbent(overfitting_gap=0.02, metric=0.87):
    plan = ExperimentPlan(
        eval_metric="roc_auc",
        model_families=["GBM", "XGB"],
        presets="medium_quality",
        time_limit=120,
        rationale="initial",
    )
    config = RunConfig(
        run_id="run_0001",
        node_id="node_abc",
        autogluon_kwargs={"hyperparameters": {"GBM": {}, "XGB": {}}},
        data_path="data/train.csv",
        output_dir="/tmp",
    )
    result = RunResult(
        run_id="run_0001",
        status="success",
        primary_metric=metric,
        diagnostics_overfitting_gap=overfitting_gap,
    )
    diagnostics = RunDiagnostics(overfitting_gap=overfitting_gap)
    return RunEntry(run_id="run_0001", node_id="node_abc", config=config, result=result, diagnostics=diagnostics)


def _valid_plan_json():
    return json.dumps({
        "eval_metric": "roc_auc",
        "model_families": ["GBM", "XGB", "CAT"],
        "presets": "medium_quality",
        "time_limit": 120,
        "feature_policy": {"exclude_columns": [], "include_columns": []},
        "validation_policy": {"holdout_frac": 0.2, "num_bag_folds": 0},
        "hyperparameters": None,
        "use_fit_extra": False,
        "rationale": "Added CAT to diversify ensemble",
    })


def test_refine_returns_experiment_plan():
    llm = MagicMock()
    llm.complete.return_value = _valid_plan_json()
    agent = RefinerAgent(llm=llm)
    incumbent = _make_incumbent()
    plan = agent.refine(
        incumbent=incumbent,
        task=_make_task(),
        prior_runs=[incumbent],
    )
    assert isinstance(plan, ExperimentPlan)
    assert "CAT" in plan.model_families


def test_refine_retries_on_invalid_json():
    llm = MagicMock()
    llm.complete.side_effect = [
        "not json at all",
        _valid_plan_json(),
    ]
    agent = RefinerAgent(llm=llm, max_retries=3)
    plan = agent.refine(
        incumbent=_make_incumbent(),
        task=_make_task(),
        prior_runs=[],
    )
    assert isinstance(plan, ExperimentPlan)
    assert llm.complete.call_count == 2


def test_refine_strips_markdown_fences():
    llm = MagicMock()
    llm.complete.return_value = f"```json\n{_valid_plan_json()}\n```"
    agent = RefinerAgent(llm=llm)
    plan = agent.refine(
        incumbent=_make_incumbent(),
        task=_make_task(),
        prior_runs=[],
    )
    assert isinstance(plan, ExperimentPlan)


def test_refine_raises_after_max_retries():
    llm = MagicMock()
    llm.complete.return_value = "garbage"
    agent = RefinerAgent(llm=llm, max_retries=2)
    with pytest.raises(ValueError, match="Failed to get"):
        agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[])


def test_user_message_includes_overfitting_gap():
    llm = MagicMock()
    llm.complete.return_value = _valid_plan_json()
    agent = RefinerAgent(llm=llm)
    agent.refine(incumbent=_make_incumbent(overfitting_gap=0.08), task=_make_task(), prior_runs=[])
    call_args = llm.complete.call_args
    user_msg = call_args[1]["messages"][1].content
    assert "overfitting_gap=0.08" in user_msg


def test_user_message_includes_incumbent_model_families():
    llm = MagicMock()
    llm.complete.return_value = _valid_plan_json()
    agent = RefinerAgent(llm=llm)
    agent.refine(incumbent=_make_incumbent(), task=_make_task(), prior_runs=[])
    call_args = llm.complete.call_args
    user_msg = call_args[1]["messages"][1].content
    assert "GBM" in user_msg
    assert "XGB" in user_msg
```

**Step 2: Run tests to verify they fail**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_refiner.py -v 2>&1 | tail -20
```

Expected: ImportError — `src/agents/refiner.py` doesn't exist.

**Step 3: Implement RefinerAgent**

```python
# src/agents/refiner.py
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
    Returns ExperimentPlan. Retries on invalid JSON.
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
        # Incumbent state
        families = incumbent.plan.model_families if hasattr(incumbent, "plan") else \
            list(incumbent.config.autogluon_kwargs.get("hyperparameters", {}).keys())
        overfitting_gap = incumbent.diagnostics.overfitting_gap
        metric = incumbent.result.primary_metric

        leaderboard_text = ""
        if incumbent.result.leaderboard:
            rows = [
                f"  {e.model_name}: val={e.score_val:.4f}"
                + (f" train={e.score_train:.4f}" if e.score_train else "")
                for e in incumbent.result.leaderboard[:5]
            ]
            leaderboard_text = "\nLeaderboard (top 5):\n" + "\n".join(rows)

        # Prior runs summary (last 5)
        history_text = ""
        if prior_runs:
            lines = []
            for r in prior_runs[-5:]:
                m = r.result.primary_metric
                fams = r.plan.model_families if hasattr(r, "plan") else []
                lines.append(
                    f"  run={r.run_id} metric={m} families={fams} "
                    f"status={r.result.status}"
                )
            history_text = "\n## Prior Runs\n" + "\n".join(lines)

        return (
            f"## Task\n"
            f"Name: {task.task_name} | Type: {task.task_type} | Metric: {task.eval_metric}\n\n"
            f"## Incumbent Config\n"
            f"model_families={families}\n"
            f"presets={incumbent.plan.presets if hasattr(incumbent, 'plan') else 'unknown'}\n"
            f"time_limit={incumbent.plan.time_limit if hasattr(incumbent, 'plan') else 'unknown'}\n"
            f"validation_policy={incumbent.plan.validation_policy if hasattr(incumbent, 'plan') else {}}\n"
            f"metric={metric:.4f}\n"
            f"overfitting_gap={overfitting_gap}"
            f"{leaderboard_text}"
            f"{history_text}\n\n"
            f"Propose ONE targeted improvement as a JSON ExperimentPlan."
        )
```

**Step 4: Run tests**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_refiner.py -v 2>&1 | tail -30
```

Expected: All PASS.

**Step 5: Commit**

```bash
cd "/home/tough/Agentic ML" && git add src/agents/refiner.py tests/test_refiner.py prompts/refiner.md && git commit -m "feat: add RefinerAgent with retry, fence-stripping, and overfitting-aware prompt"
```

---

### Task 5: Wire RefinerAgent into session.py optimize loop

**Files:**
- Modify: `src/session.py`

**Step 1: Add RefinerAgent import and __init__ parameter**

At the top of `session.py`, add:
```python
from src.agents.refiner import RefinerAgent
```

In `Session.__init__`, after `self._selector = SelectorAgent(llm)`, add:
```python
self._refiner = RefinerAgent(llm)
```

**Step 2: Replace the SelectorAgent call in the optimize loop**

Find this block (around line 304-314):

```python
# Use selector to propose refinement (refiner agent added in Phase 3)
plan = self._selector.select(
    hypothesis=(
        f"Refine the current best config (metric={incumbent.primary_metric():.4f}). "
        f"Try ONE improvement: consider changing validation strategy, "
        f"model families, or increasing time budget."
    ),
    task=self.task,
    data_profile=data_profile,
    prior_runs=self.run_store.get_history(),
)
```

Replace with:

```python
incumbent_entry = self.run_store.get_incumbent(self._higher_is_better)
plan = self._refiner.refine(
    incumbent=incumbent_entry,
    task=self.task,
    prior_runs=self.run_store.get_history(),
)
```

**Step 3: Verify existing tests still pass**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/ -v 2>&1 | tail -20
```

Expected: All PASS.

**Step 4: Commit**

```bash
cd "/home/tough/Agentic ML" && git add src/session.py && git commit -m "feat: wire RefinerAgent into optimize loop, replacing generic selector call"
```

---

### Task 6: Smoke-test end-to-end

**Step 1: Run main.py**

```bash
cd "/home/tough/Agentic ML" && python3 main.py 2>&1 | tee /tmp/phase3_run.log
```

**Step 2: Verify log shows targeted refinements**

```bash
grep "Optimize run\|ACCEPTED\|REJECTED\|overfitting" /tmp/phase3_run.log
```

Expected: Optimize runs show different model families across iterations (RefinerAgent diversifying), no crash.

**Step 3: Verify overfitting_gap in decisions.jsonl**

```bash
python3 -c "
import json
lines = open('experiments/$(ls -t experiments/ | grep -v case_bank | head -1)/decisions.jsonl').readlines()
for l in lines:
    e = json.loads(l)
    gap = e.get('diagnostics', {}).get('overfitting_gap')
    print(f\"{e['run_id']}: overfitting_gap={gap}\")
"
```

Expected: Non-None values for successful runs.

---

### Task 7: Update docs

**Files:**
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Update current-state.md**

Change `**Phase:** 2 (Memory & Ideation complete)` → `**Phase:** 3 (Principled Refinement complete)`

In the AGENT LAYER box, add `RefinerAgent` to the list.

In `## What is NOT yet built`, remove the RefinerAgent line. Update remaining list:
```
- ReviewerAgent — post-run quality assessment (Phase 4)
- Optuna executor (Phase 4)
- Graph RAG over ExperimentNode trees (Phase 5)
```

Update the Session Flow step 5:
```
5. Optimize: RefinerAgent reads incumbent config + leaderboard + overfitting_gap → targeted ExperimentPlan
```

**Step 2: Append to implementation-log.md**

```markdown
## 2026-03-20 — Phase 3: Principled Refinement

**What changed:**
- ResultParser now calls `leaderboard(extra_info=True)` to capture `score_train`, computes `overfitting_gap = score_train - score_val` for best model, stored on `RunResult.diagnostics_overfitting_gap`
- `ModelEntry` gains `score_train: Optional[float]`
- `session.execute_node` populates `RunDiagnostics.overfitting_gap` and `metric_vs_parent` after each run
- New `RefinerAgent` (`src/agents/refiner.py`) replaces the generic SelectorAgent call in the optimize loop; receives full incumbent state (config, leaderboard, overfitting_gap, prior_runs) and produces a targeted one-step ExperimentPlan
- `prompts/refiner.md` encodes decision rules: if overfitting → reduce complexity; if plateau → change validation; if homogeneous families → diversify

**Why:**
The optimize loop was calling SelectorAgent with a generic "refine this config" string; the LLM had no access to what the incumbent config contained, which models had trained, or whether overfitting was occurring. RefinerAgent gives the LLM the evidence it needs to make principled decisions.

**Must remain true:**
- `leaderboard(extra_info=True)` failure is caught and falls back to basic leaderboard
- RefinerAgent and SelectorAgent share the same retry + fence-strip pattern
- RunDiagnostics fields remain Optional — they are never required for correctness, only for better agent decisions
```

**Step 3: Commit**

```bash
cd "/home/tough/Agentic ML" && git add docs/architecture/current-state.md docs/changes/implementation-log.md && git commit -m "docs: update architecture and implementation log for Phase 3"
```

---

## Summary

| Task | Component | Key file(s) |
|------|-----------|-------------|
| 1 | Richer ResultParser | `src/execution/result_parser.py`, `src/models/results.py` |
| 2 | Diagnostics in session | `src/session.py` |
| 3 | Refiner prompt | `prompts/refiner.md` |
| 4 | RefinerAgent | `src/agents/refiner.py` |
| 5 | Wire into session | `src/session.py` |
| 6 | Smoke test | run `main.py` |
| 7 | Docs | `docs/architecture/`, `docs/changes/` |
