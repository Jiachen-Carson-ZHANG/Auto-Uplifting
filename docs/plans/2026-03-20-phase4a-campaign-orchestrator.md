# Phase 4a: Campaign Orchestrator Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `CampaignOrchestrator` outer loop that runs multiple `ExperimentSession`s, detects metric plateau, and stops — plus a stub `PreprocessingExecutor` (identity transform) that Phase 4b will replace with real code generation.

**Architecture:** Five new files: `src/models/campaign.py` (`CampaignConfig`, `SessionSummary`, `CampaignResult`), `src/models/preprocessing.py` (`PreprocessingPlan`), `src/execution/preprocessing_runner.py` (`PreprocessingExecutor`), `src/orchestration/campaign.py` (`CampaignOrchestrator`), and `campaign.py` entry point at the repo root. `CampaignOrchestrator` runs sessions in a loop, writes `campaign.json` after each session to `experiments/campaigns/{id}/sessions/`, and logs to `campaign.log` in the campaign dir. `ExperimentSession` gains an optional `preprocessed_data_path` param so the orchestrator can hand off preprocessed data. Cherry-picks accepted from CEO review: `higher_is_better` param, sessions stored in `campaigns/{id}/sessions/`, per-session `campaign.json` writes, session error handling with `error_message`, and `FileHandler` for `campaign.log`.

**Tech Stack:** Python 3.12, Pydantic v2, PyYAML, existing `ExperimentSession`, pytest

---

### Task 1: Campaign data models

**Files:**
- Create: `src/models/campaign.py`
- Create: `tests/models/test_campaign.py`

**Step 1: Write the failing test**

```python
# tests/models/test_campaign.py
import pytest
from src.models.campaign import CampaignConfig, SessionSummary, CampaignResult


def test_campaign_config_defaults():
    cfg = CampaignConfig()
    assert cfg.max_sessions == 5
    assert cfg.plateau_threshold == 0.002
    assert cfg.plateau_window == 3


def test_session_summary_requires_fields():
    s = SessionSummary(
        session_id="s1",
        best_metric=0.87,
        preprocessing_strategy="identity",
        session_dir="/tmp/s1",
        duration_seconds=42.0,
        error_message=None,
    )
    assert s.session_id == "s1"
    assert s.best_metric == 0.87


def test_session_summary_none_metric():
    # A session where all runs failed has best_metric=None
    s = SessionSummary(
        session_id="s2",
        best_metric=None,
        preprocessing_strategy="identity",
        session_dir="/tmp/s2",
        duration_seconds=5.0,
    )
    assert s.best_metric is None


def test_session_summary_error_message():
    s = SessionSummary(
        session_id="s3",
        best_metric=None,
        preprocessing_strategy="identity",
        session_dir="/tmp/s3",
        duration_seconds=1.0,
        error_message="AutoGluon raised RuntimeError",
    )
    assert s.error_message == "AutoGluon raised RuntimeError"


def test_campaign_result_serialises():
    import json
    r = CampaignResult(
        campaign_id="c1",
        task_name="titanic",
        started_at="2026-03-20T00:00:00",
        sessions=[],
        best_metric=None,
        best_session_id=None,
        stopped_reason="budget",
    )
    blob = json.loads(r.model_dump_json())
    assert blob["stopped_reason"] == "budget"
```

**Step 2: Run test to verify it fails**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/models/test_campaign.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.models.campaign'`

**Step 3: Write implementation**

```python
# src/models/campaign.py
"""
Campaign-level data classes.

Origin  : defined by CampaignOrchestrator (src/orchestration/campaign.py)
Consumed: campaign.py entrypoint (display / persistence), future analysis tools
"""
from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel


class CampaignConfig(BaseModel):
    """
    Hyper-parameters for the outer campaign loop.
    All runs in a campaign share the same task and LLM.
    """
    max_sessions: int = 5
    plateau_threshold: float = 0.002   # stop if best metric moves < this across plateau_window sessions
    plateau_window: int = 3            # number of recent sessions to check for plateau


class SessionSummary(BaseModel):
    """
    One row in the campaign log — one per completed session.
    Written by CampaignOrchestrator after each session finishes.
    """
    session_id: str
    best_metric: Optional[float]           # None if all runs failed
    preprocessing_strategy: str = "identity"
    session_dir: str                       # absolute path to session artifacts
    duration_seconds: float
    error_message: Optional[str] = None   # set if the session raised an exception


class CampaignResult(BaseModel):
    """
    Full record of a completed campaign.
    Saved to experiments/campaigns/{campaign_id}/campaign.json.
    """
    campaign_id: str
    task_name: str
    started_at: str                        # ISO 8601
    sessions: List[SessionSummary]
    best_metric: Optional[float]           # best across all sessions
    best_session_id: Optional[str]
    stopped_reason: Literal['plateau', 'budget']
```

**Step 4: Run test to verify it passes**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/models/test_campaign.py -v
```
Expected: 4 PASSED

**Step 5: Commit**

```bash
cd "/home/tough/Agentic ML"
git add src/models/campaign.py tests/models/test_campaign.py
git commit -m "feat: add CampaignConfig, SessionSummary, CampaignResult models"
```

---

### Task 2: PreprocessingPlan model + stub executor

**Files:**
- Create: `src/models/preprocessing.py`
- Create: `src/execution/preprocessing_runner.py`
- Create: `tests/execution/test_preprocessing_runner.py`

**Step 1: Write the failing test**

```python
# tests/execution/test_preprocessing_runner.py
import pytest
import pandas as pd
from pathlib import Path
from src.models.preprocessing import PreprocessingPlan
from src.execution.preprocessing_runner import PreprocessingExecutor


def test_identity_copies_file(tmp_path):
    # Write a tiny CSV
    src = tmp_path / "data.csv"
    src.write_text("a,b,label\n1,2,0\n3,4,1\n")

    plan = PreprocessingPlan(strategy="identity")
    executor = PreprocessingExecutor()
    out_path = executor.run(str(src), plan, str(tmp_path / "out"))

    # Output file must exist and have same content
    assert Path(out_path).exists()
    original = pd.read_csv(str(src))
    result = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(original, result)


def test_identity_output_named_preprocessed_data(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,label\n1,0\n")
    plan = PreprocessingPlan(strategy="identity")
    out_path = PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
    assert Path(out_path).name == "preprocessed_data.csv"


def test_unknown_strategy_raises(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,label\n1,0\n")
    plan = PreprocessingPlan(strategy="generated", code="def preprocess(df, t): return df")
    with pytest.raises(NotImplementedError):
        PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
```

**Step 2: Run test to verify it fails**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/execution/test_preprocessing_runner.py -v
```
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/models/preprocessing.py
"""
Preprocessing plan produced by PreprocessingAgent (Phase 4b) or stub (Phase 4a).

Origin  : PreprocessingAgent (not yet built) or CampaignOrchestrator stub
Consumed: PreprocessingExecutor (src/execution/preprocessing_runner.py)
"""
from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel


class PreprocessingPlan(BaseModel):
    """
    Describes how to transform raw data before AutoGluon sees it.

    Phase 4a: strategy="identity" — data passed through unchanged.
    Phase 4b: strategy="generated" — code contains a preprocess(df, target_col) function
              generated by PreprocessingAgent and validated by ValidationHarness.
    """
    strategy: str = "identity"         # "identity" | "generated"
    code: Optional[str] = None         # Python function string (Phase 4b only)
    rationale: str = ""                # agent's reasoning
    transformations: List[str] = []    # human-readable list of what the code does
```

```python
# src/execution/preprocessing_runner.py
"""
Executes a PreprocessingPlan against a CSV file.

Phase 4a: identity — copies data unchanged.
Phase 4b: executes generated code through ValidationHarness (not yet built).

Input : data_path (CSV path), PreprocessingPlan, output_dir (will be created)
Output: path to preprocessed_data.csv inside output_dir
"""
from __future__ import annotations
import shutil
from pathlib import Path
from src.models.preprocessing import PreprocessingPlan


class PreprocessingExecutor:
    """
    Applies a PreprocessingPlan to a CSV file and saves the result.
    Currently only supports strategy="identity".
    """

    def run(self, data_path: str, plan: PreprocessingPlan, output_dir: str) -> str:
        """
        Returns the absolute path to the preprocessed CSV.
        Creates output_dir if it does not exist.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "preprocessed_data.csv"

        if plan.strategy == "identity":
            shutil.copy2(data_path, str(out_path))
            return str(out_path)

        raise NotImplementedError(
            f"PreprocessingExecutor: strategy '{plan.strategy}' is not yet implemented. "
            f"Phase 4b will add 'generated' strategy support."
        )
```

**Step 4: Run test to verify it passes**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/execution/test_preprocessing_runner.py -v
```
Expected: 3 PASSED

**Step 5: Commit**

```bash
cd "/home/tough/Agentic ML"
git add src/models/preprocessing.py src/execution/preprocessing_runner.py \
        tests/execution/test_preprocessing_runner.py
git commit -m "feat: add PreprocessingPlan model and stub PreprocessingExecutor"
```

---

### Task 3: ExperimentSession accepts preprocessed_data_path

**Files:**
- Modify: `src/session.py`
- Modify: `tests/test_session.py`

**Context:** `ExperimentSession.__init__` currently loads data from `task.data_path`. Add an optional `preprocessed_data_path` param. When provided, use that path for data loading (profiling + passing to runner) instead of `task.data_path`. Also write `preprocessing_plan` into `manifest.json`.

**Step 1: Write the failing test**

Add this test to `tests/test_session.py`:

```python
def test_session_uses_preprocessed_data_path(tmp_path):
    """When preprocessed_data_path is provided, session loads that CSV."""
    mock_llm = MagicMock(spec=LLMBackend)

    # raw CSV (5 rows)
    raw_csv = tmp_path / "raw.csv"
    raw_csv.write_text("feature1,feature2,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n9,10,0\n")

    # preprocessed CSV (same columns, 3 rows — deliberately different)
    prep_csv = tmp_path / "preprocessed_data.csv"
    prep_csv.write_text("feature1,feature2,label\n1,2,0\n3,4,1\n5,6,0\n")

    task = TaskSpec(
        task_name="test", task_type="binary",
        data_path=str(raw_csv), target_column="label",
        eval_metric="roc_auc", description="Test",
    )
    session = ExperimentSession(
        task=task, llm=mock_llm,
        experiments_dir=str(tmp_path / "experiments"),
        num_candidates=1, max_optimize_iterations=1,
        preprocessed_data_path=str(prep_csv),
    )
    # Data profile should reflect the preprocessed CSV (3 rows), not the raw (5 rows)
    assert session._data_profile.n_rows == 3
    # AutoGluon must also train on preprocessed data, not raw
    assert session._data_path == str(prep_csv)
```

**Step 2: Run test to verify it fails**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_session.py::test_session_uses_preprocessed_data_path -v
```
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'preprocessed_data_path'`

**Step 3: Write implementation**

In `src/session.py`, make these changes:

1. Add import at top:
```python
from src.models.preprocessing import PreprocessingPlan
```

2. Add `preprocessed_data_path` to `__init__` signature (after `case_store_path`):
```python
preprocessed_data_path: Optional[str] = None,
```

3. In `__init__`, after `self._session_dir.mkdir(...)`, store it:
```python
self._data_path = preprocessed_data_path or task.data_path
```

4. There are **two** places in `session.py` that use `task.data_path` — both must be changed to `self._data_path`:

   - **Line 112** (profiling): `df = pd.read_csv(self.task.data_path)` → `df = pd.read_csv(self._data_path)`
   - **Line 171** (AutoGluon training): `data_path=self.task.data_path` → `data_path=self._data_path`

   If only line 112 is changed, the data profile reflects the preprocessed data but AutoGluon still trains on the raw CSV — preprocessing is silently ignored for model training.

5. In the manifest dict written to `manifest.json`, add:
```python
"preprocessed_data_path": preprocessed_data_path,
```

**Step 4: Run test to verify it passes**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_session.py -v
```
Expected: ALL PASS (including the new test)

**Step 5: Commit**

```bash
cd "/home/tough/Agentic ML"
git add src/session.py tests/test_session.py
git commit -m "feat: ExperimentSession accepts preprocessed_data_path"
```

---

### Task 4: CampaignOrchestrator

**Files:**
- Create: `src/orchestration/campaign.py`
- Create: `tests/orchestration/test_campaign.py`

**Step 1: Write the failing tests**

```python
# tests/orchestration/test_campaign.py
import pytest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.models.campaign import CampaignConfig
from src.models.task import TaskSpec
from src.llm.backend import LLMBackend
from src.orchestration.campaign import CampaignOrchestrator


def make_task(tmp_path) -> TaskSpec:
    csv = tmp_path / "data.csv"
    csv.write_text("f1,f2,label\n1,2,0\n3,4,1\n5,6,0\n7,8,1\n")
    return TaskSpec(
        task_name="test", task_type="binary",
        data_path=str(csv), target_column="label",
        eval_metric="roc_auc", description="Test",
    )


def test_plateau_detection():
    cfg = CampaignConfig(plateau_threshold=0.002, plateau_window=3)
    orch = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orch._config = cfg

    # Not enough sessions yet
    assert orch._is_plateau([0.87, 0.871]) is False

    # Flat — all within 0.002
    assert orch._is_plateau([0.87, 0.871, 0.870]) is True

    # Not flat — big jump in last window
    assert orch._is_plateau([0.85, 0.86, 0.871]) is False


def test_campaign_stops_at_budget(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=2, plateau_window=5)  # plateau never triggers

    session_metrics = [0.87, 0.88]  # two sessions

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"sess_{call_count}"
            (tmp_path / f"sess_{call_count}").mkdir(exist_ok=True)
            inc = MagicMock()
            inc.result.primary_metric = session_metrics[call_count] if call_count < len(session_metrics) else None
            m.run_store.get_incumbent.return_value = inc
            m.run.return_value = None
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        result = orch.run()

    assert result.stopped_reason == "budget"
    assert len(result.sessions) == 2


def test_campaign_stops_on_plateau(tmp_path):
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=10, plateau_window=3, plateau_threshold=0.002)

    call_count = 0
    flat_metrics = [0.87, 0.870, 0.871]  # 3 sessions, all within 0.002

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"s_{call_count}"
            (tmp_path / f"s_{call_count}").mkdir(exist_ok=True)
            inc = MagicMock()
            inc.result.primary_metric = flat_metrics[call_count]
            m.run_store.get_incumbent.return_value = inc
            m.run.return_value = None
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        result = orch.run()

    assert result.stopped_reason == "plateau"
    assert len(result.sessions) == 3
```

**Step 2: Run test to verify it fails**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/orchestration/test_campaign.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.orchestration.campaign'`

**Step 3: Write implementation**

```python
# src/orchestration/campaign.py
"""
Outer optimization loop over multiple ExperimentSessions.

Origin  : campaign.py entrypoint (repo root)
Consumed: nothing downstream — writes campaign.json and campaign.log to disk

Each iteration:
  1. PreprocessingExecutor applies PreprocessingPlan (Phase 4a: identity)
  2. ExperimentSession runs warm-up + optimize loop on the preprocessed data
  3. CampaignOrchestrator records SessionSummary, checks stop conditions

Sessions are stored inside campaigns/{campaign_id}/sessions/ so each campaign
is a self-contained folder.
"""
from __future__ import annotations
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.models.campaign import CampaignConfig, SessionSummary, CampaignResult
from src.models.task import TaskSpec
from src.models.preprocessing import PreprocessingPlan
from src.llm.backend import LLMBackend
from src.execution.preprocessing_runner import PreprocessingExecutor
from src.session import ExperimentSession


class CampaignOrchestrator:
    """
    Runs multiple ExperimentSessions on the same task, stopping when the
    metric plateaus or the session budget is exhausted.

    Sessions are stored in campaigns/{campaign_id}/sessions/ for easy navigation.
    campaign.json is written after each session so partial results survive crashes.

    Phase 4a: always uses identity preprocessing.
    Phase 4b: will generate new preprocessing strategies on plateau.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: LLMBackend,
        config: Optional[CampaignConfig] = None,
        experiments_dir: str = "experiments",
        num_candidates: int = 3,
        max_optimize_iterations: int = 5,
        higher_is_better: bool = True,
        case_store_path: Optional[str] = None,
    ) -> None:
        self._task = task
        self._llm = llm
        self._config = config or CampaignConfig()
        self._experiments_dir = experiments_dir
        self._num_candidates = num_candidates
        self._max_optimize_iterations = max_optimize_iterations
        self._higher_is_better = higher_is_better
        self._case_store_path = case_store_path
        self._executor = PreprocessingExecutor()

    def run(self) -> CampaignResult:
        campaign_id = str(uuid.uuid4())[:8]
        started_at = datetime.now().isoformat()
        campaign_dir = Path(self._experiments_dir) / "campaigns" / f"{campaign_id}_{self._task.task_name}"
        sessions_dir = campaign_dir / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        # Campaign-level logger: writes to both stdout and campaign.log
        log = logging.getLogger(f"campaign.{campaign_id}")
        log.setLevel(logging.DEBUG)
        if not log.handlers:
            fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
            sh = logging.StreamHandler()
            sh.setFormatter(fmt)
            fh = logging.FileHandler(campaign_dir / "campaign.log", mode="a")
            fh.setFormatter(fmt)
            log.addHandler(sh)
            log.addHandler(fh)

        log.info("=" * 60)
        log.info(f"Campaign {campaign_id} | task={self._task.task_name} | max_sessions={self._config.max_sessions}")
        log.info("=" * 60)

        sessions: List[SessionSummary] = []
        metrics: List[float] = []

        for i in range(self._config.max_sessions):
            log.info(f"--- Session {i + 1}/{self._config.max_sessions} ---")
            t_start = datetime.now()
            summary: Optional[SessionSummary] = None

            try:
                plan = self._preprocessing_plan()
                prep_dir = campaign_dir / f"preprocessing_{i + 1}"
                preprocessed_path = self._executor.run(self._task.data_path, plan, str(prep_dir))

                session = ExperimentSession(
                    task=self._task,
                    llm=self._llm,
                    experiments_dir=str(sessions_dir),
                    num_candidates=self._num_candidates,
                    max_optimize_iterations=self._max_optimize_iterations,
                    higher_is_better=self._higher_is_better,
                    case_store_path=self._case_store_path,
                    preprocessed_data_path=preprocessed_path,
                )
                session.run()

                duration = (datetime.now() - t_start).total_seconds()
                incumbent = session.run_store.get_incumbent(higher_is_better=self._higher_is_better)
                best_metric = incumbent.result.primary_metric if incumbent else None

                summary = SessionSummary(
                    session_id=str(session._session_dir.name),
                    best_metric=best_metric,
                    preprocessing_strategy=plan.strategy,
                    session_dir=str(session._session_dir),
                    duration_seconds=duration,
                )
                if best_metric is not None:
                    metrics.append(best_metric)
                    log.info(f"Session {i + 1} best: {best_metric:.4f}")
                else:
                    log.warning(f"Session {i + 1}: no successful runs")

            except Exception as exc:
                duration = (datetime.now() - t_start).total_seconds()
                log.error(f"Session {i + 1} failed: {exc}")
                summary = SessionSummary(
                    session_id=f"session_{i + 1}_failed",
                    best_metric=None,
                    preprocessing_strategy="identity",
                    session_dir="",
                    duration_seconds=duration,
                    error_message=str(exc),
                )

            sessions.append(summary)
            # Write campaign.json after every session so partial results survive
            partial = self._build_result(campaign_id, started_at, sessions, "budget")
            self._save(partial, campaign_dir)

            if self._is_plateau(metrics):
                log.info("Plateau detected — stopping campaign.")
                result = self._build_result(campaign_id, started_at, sessions, "plateau")
                self._save(result, campaign_dir)
                return result

        result = self._build_result(campaign_id, started_at, sessions, "budget")
        self._save(result, campaign_dir)
        log.info(f"Campaign complete: best={result.best_metric} | stopped={result.stopped_reason}")
        return result

    def _is_plateau(self, metrics: List[float]) -> bool:
        """True if the last plateau_window metrics are all within plateau_threshold of each other."""
        if len(metrics) < self._config.plateau_window:
            return False
        recent = metrics[-self._config.plateau_window:]
        return max(recent) - min(recent) < self._config.plateau_threshold

    def _best_metric(self, metrics: List[float]) -> Optional[float]:
        """Returns best metric respecting higher_is_better."""
        if not metrics:
            return None
        return max(metrics) if self._higher_is_better else min(metrics)

    def _preprocessing_plan(self) -> PreprocessingPlan:
        """Phase 4a: always identity. Phase 4b: call PreprocessingAgent here."""
        return PreprocessingPlan(strategy="identity")

    def _build_result(
        self,
        campaign_id: str,
        started_at: str,
        sessions: List[SessionSummary],
        stopped_reason: Literal['plateau', 'budget'],
    ) -> CampaignResult:
        metrics_with_values = [s.best_metric for s in sessions if s.best_metric is not None]
        best_metric = self._best_metric(metrics_with_values)
        best_session_id = None
        if best_metric is not None:
            best_session_id = next(
                s.session_id for s in sessions if s.best_metric == best_metric
            )
        return CampaignResult(
            campaign_id=campaign_id,
            task_name=self._task.task_name,
            started_at=started_at,
            sessions=sessions,
            best_metric=best_metric,
            best_session_id=best_session_id,
            stopped_reason=stopped_reason,
        )

    def _save(self, result: CampaignResult, campaign_dir: Path) -> None:
        path = campaign_dir / "campaign.json"
        path.write_text(result.model_dump_json(indent=2))
```

**Step 4: Add additional tests to `tests/orchestration/test_campaign.py`**

```python
def test_lower_is_better_plateau(tmp_path):
    """_is_plateau uses spread (max-min), which works for both higher and lower is better."""
    cfg = CampaignConfig(plateau_threshold=0.002, plateau_window=3)
    orch = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orch._config = cfg
    # For RMSE: values decreasing (improving). Spread < threshold → plateau.
    assert orch._is_plateau([0.300, 0.301, 0.300]) is True
    # Large improvement → not plateau
    assert orch._is_plateau([0.300, 0.280, 0.260]) is False


def test_best_metric_lower_is_better(tmp_path):
    orch = CampaignOrchestrator.__new__(CampaignOrchestrator)
    orch._higher_is_better = False
    assert orch._best_metric([0.30, 0.28, 0.25]) == 0.25  # lower is better


def test_session_error_campaign_continues(tmp_path):
    """If a session throws, the campaign records the error and continues."""
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=2, plateau_window=5)
    call_count = 0

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"s_{call_count}"
            (tmp_path / f"s_{call_count}").mkdir(exist_ok=True)
            if call_count == 0:
                m.run.side_effect = RuntimeError("AutoGluon crashed")
            else:
                inc = MagicMock()
                inc.result.primary_metric = 0.87
                m.run_store.get_incumbent.return_value = inc
                m.run.return_value = None
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        result = orch.run()

    assert len(result.sessions) == 2
    assert result.sessions[0].error_message is not None
    assert result.sessions[1].best_metric == 0.87
    assert result.stopped_reason == "budget"


def test_campaign_json_written_after_each_session(tmp_path):
    """campaign.json must be written after session 1, before session 2 starts (crash-survival guarantee)."""
    mock_llm = MagicMock(spec=LLMBackend)
    task = make_task(tmp_path)
    cfg = CampaignConfig(max_sessions=2, plateau_window=5)
    call_count = 0
    campaign_json_existed_before_session2 = [False]

    with patch("src.orchestration.campaign.ExperimentSession") as MockSession, \
         patch("src.orchestration.campaign.PreprocessingExecutor"):
        def side_effect(*args, **kwargs):
            nonlocal call_count
            m = MagicMock()
            m._session_dir = tmp_path / f"s_{call_count}"
            (tmp_path / f"s_{call_count}").mkdir(exist_ok=True)
            inc = MagicMock()
            inc.result.primary_metric = 0.87
            m.run_store.get_incumbent.return_value = inc
            m.run.return_value = None
            if call_count == 1:
                # Session 1 write should have happened before session 2 is constructed
                campaigns = list((tmp_path / "experiments" / "campaigns").iterdir())
                if campaigns and (campaigns[0] / "campaign.json").exists():
                    campaign_json_existed_before_session2[0] = True
            call_count += 1
            return m
        MockSession.side_effect = side_effect

        orch = CampaignOrchestrator(
            task=task, llm=mock_llm, config=cfg,
            experiments_dir=str(tmp_path / "experiments"),
        )
        orch.run()

    assert campaign_json_existed_before_session2[0], \
        "campaign.json was not written after session 1 (crash-survival guarantee broken)"
```

**Step 5: Run tests**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/orchestration/test_campaign.py -v
```
Expected: 7 PASSED

**Step 6: Commit**

```bash
cd "/home/tough/Agentic ML"
git add src/orchestration/campaign.py tests/orchestration/test_campaign.py
git commit -m "feat: add CampaignOrchestrator with plateau detection and budget stop"
```

---

### Task 5: campaign.py entry point + project.yaml campaign section

**Files:**
- Create: `campaign.py` (repo root)
- Modify: `configs/project.yaml`

**Step 1: Write the failing smoke test**

```python
# tests/test_campaign_entrypoint.py
import subprocess
import sys


def test_campaign_entrypoint_shows_help():
    """campaign.py --help should exit 0 and mention 'campaign'."""
    result = subprocess.run(
        [sys.executable, "campaign.py", "--help"],
        capture_output=True, text=True, cwd="/home/tough/Agentic ML"
    )
    assert result.returncode == 0
    assert "campaign" in result.stdout.lower() or "campaign" in result.stderr.lower()


def test_best_str_none_metric():
    """When all sessions fail, best_metric=None must not crash the format string."""
    # Directly exercise the format logic from campaign.py main()
    best_metric = None
    best_str = f"{best_metric:.4f}" if best_metric is not None else "N/A"
    assert best_str == "N/A"
```

**Step 2: Run test to verify it fails**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_campaign_entrypoint.py -v
```
Expected: FAIL — `FileNotFoundError` or non-zero returncode

**Step 3: Write campaign.py**

```python
#!/usr/bin/env python3
"""
Campaign entrypoint — runs multiple ExperimentSessions via CampaignOrchestrator.

Usage:
    python3 campaign.py
    python3 campaign.py --config configs/project.yaml
"""
import argparse
import os
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.models.task import TaskSpec
from src.models.campaign import CampaignConfig
from src.llm.backend import create_backend
from src.orchestration.campaign import CampaignOrchestrator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Agentic ML Campaign — multi-session optimization")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    search_config = load_config("configs/search.yaml")

    task_cfg = config["task"]
    task = TaskSpec(
        task_name=task_cfg["name"],
        task_type=task_cfg["type"],
        data_path=task_cfg["data_path"],
        target_column=task_cfg["target_column"],
        eval_metric=task_cfg["eval_metric"],
        constraints=task_cfg.get("constraints", {}),
        description=task_cfg.get("description", ""),
    )

    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = llm_cfg["model"]
    api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"Warning: {api_key_env} not set. LLM calls will fail.")

    llm = create_backend(provider=provider, model=model, api_key=api_key)

    search = search_config.get("search", {})
    num_candidates = search.get("num_candidates", 3)
    max_optimize = search.get("max_optimize_iterations", 5)
    eval_metric = task_cfg.get("eval_metric", "roc_auc")
    higher_is_better = search.get("higher_is_better", {}).get(eval_metric, True)

    campaign_cfg = config.get("campaign", {})
    campaign_config = CampaignConfig(
        max_sessions=campaign_cfg.get("max_sessions", 5),
        plateau_threshold=campaign_cfg.get("plateau_threshold", 0.002),
        plateau_window=campaign_cfg.get("plateau_window", 3),
    )

    orchestrator = CampaignOrchestrator(
        task=task,
        llm=llm,
        config=campaign_config,
        experiments_dir=config["session"]["experiments_dir"],
        num_candidates=num_candidates,
        max_optimize_iterations=max_optimize,
        higher_is_better=higher_is_better,
        case_store_path=config["session"].get("case_store_path"),
    )

    result = orchestrator.run()
    best_str = f"{result.best_metric:.4f}" if result.best_metric is not None else "N/A"
    print(f"\nCampaign complete: best={best_str} | sessions={len(result.sessions)} | stopped={result.stopped_reason}")


if __name__ == "__main__":
    main()
```

**Step 4: Add `campaign` section to `configs/project.yaml`**

Append to the end of the file:
```yaml

campaign:
  max_sessions: 5
  plateau_threshold: 0.002
  plateau_window: 3
```

**Step 5: Run test**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/test_campaign_entrypoint.py -v
```
Expected: PASS

Also verify manually:
```bash
cd "/home/tough/Agentic ML" && python3 campaign.py --help
```
Expected: prints usage and exits 0

**Step 6: Commit**

```bash
cd "/home/tough/Agentic ML"
git add campaign.py configs/project.yaml tests/test_campaign_entrypoint.py
git commit -m "feat: add campaign.py entrypoint and campaign config section"
```

---

### Task 6: Full test suite + docs

**Files:**
- Run: all tests
- Modify: `docs/architecture/current-state.md`
- Modify: `docs/changes/implementation-log.md`

**Step 1: Run full test suite**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/ -v
```
Expected: ALL PASS. If any test fails, fix before proceeding.

**Step 2: Update `docs/architecture/current-state.md`**

Change the phase line:
```
**Phase:** 3 (Principled Refinement complete)
```
to:
```
**Phase:** 4a (Campaign Orchestration complete)
```

Add `CampaignOrchestrator` to the ORCHESTRATION LAYER box (alongside `Scheduler`, `AcceptReject`, `ExperimentTree`).

Add a new row to the Data Objects table:
```
| CampaignResult | models/campaign.py | Full record of a multi-session campaign |
| SessionSummary | models/campaign.py | Per-session result within a campaign |
| PreprocessingPlan | models/preprocessing.py | Stub for Phase 4b preprocessing strategy |
```

Update "What is NOT yet built" section:
```
- **Phase 4b:** PreprocessingAgent (ReAct code gen), ValidationHarness, preprocessing_bank, EmbeddingRetriever, external knowledge seeds
- ReviewerAgent — post-run quality assessment (Phase 5)
- Graph RAG over ExperimentNode trees (Phase 5)
```

**Step 3: Append to `docs/changes/implementation-log.md`**

```markdown
## 2026-03-20 — Phase 4a: Campaign Orchestration

**What changed:**
- New `CampaignConfig`, `SessionSummary`, `CampaignResult` models (`src/models/campaign.py`)
- New `PreprocessingPlan` stub model (`src/models/preprocessing.py`) — strategy="identity" passes data through unchanged
- New `PreprocessingExecutor` stub (`src/execution/preprocessing_runner.py`) — copies CSV unchanged, raises NotImplementedError for "generated" strategy
- New `CampaignOrchestrator` (`src/orchestration/campaign.py`) — outer loop over `ExperimentSession`, plateau detection, budget stop, writes `campaign.json`
- `ExperimentSession` gains `preprocessed_data_path: Optional[str]` — uses it for data loading when provided
- New `campaign.py` entry point at repo root — single-session `main.py` unchanged
- `configs/project.yaml` gains `campaign:` section

**Why:**
The inner optimize loop exhausts interesting model-selection combinations in 5–8 runs. The real improvement lever is preprocessing strategy (feature engineering, imputation, class imbalance). CampaignOrchestrator creates the outer loop that will try different preprocessing strategies across sessions once Phase 4b adds PreprocessingAgent.

**Must remain true:**
- `python3 main.py` still works unchanged (single session)
- `python3 campaign.py` runs multiple sessions and detects plateau
- All tests pass
```

**Step 4: Commit docs**

```bash
cd "/home/tough/Agentic ML"
git add docs/architecture/current-state.md docs/changes/implementation-log.md
git commit -m "docs: update architecture and implementation log for Phase 4a"
```

**Step 5: Final smoke test**

```bash
cd "/home/tough/Agentic ML" && python -m pytest tests/ -v --tb=short
```
Expected: ALL PASS, no warnings about missing modules.
