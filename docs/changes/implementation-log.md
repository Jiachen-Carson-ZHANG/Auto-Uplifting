## 2026-03-20 — Schema cleanup: RunResult, RunDiagnostics, RunConfig, RunEntry→ExperimentRun

**What changed:**
- `RunResult` stripped to execution output only: removed `run_id`, `artifacts_dir`, `raw_info`, `diagnostics_overfitting_gap`; `status` now `Literal["success", "failed"]`
- `RunDiagnostics` stripped to computed observations: removed `data_profile_ref`, `feature_importances`, `change_description`
- `RunConfig` stripped to AutoGluon kwargs wrapper: removed `run_id`, `node_id` (owned by `ExperimentRun`)
- `RunEntry` renamed to `ExperimentRun`; removed `agent_review` (ReviewerAgent not built yet)
- `ResultParser.from_predictor` now returns `(RunResult, Optional[float])` — overfitting_gap surfaced as a separate value; `from_error` simplified to single `error_msg` arg
- `AutoGluonRunner.run` propagates the tuple; `session.execute_node` unpacks and puts gap into `RunDiagnostics`

**Why:** Fields were duplicated across 2-3 classes (run_id in ExperimentRun + RunResult + RunConfig), dead fields had accumulated (raw_info, data_profile_ref), and RunEntry's name implied a log line rather than a composite record.

## 2026-03-20 — Phase 3: Principled Refinement

**What changed:**
- `ModelEntry` gains `score_train: Optional[float]`; `RunResult` gains `diagnostics_overfitting_gap: Optional[float]`
- `ResultParser.from_predictor` calls `leaderboard(extra_info=True)` to capture train scores and compute `overfitting_gap = score_train - score_val`; falls back to basic leaderboard on failure
- `session.execute_node` populates `RunDiagnostics.overfitting_gap` and `metric_vs_parent` after each run; `RunEntry` now carries `plan: Optional[ExperimentPlan]`
- New `RefinerAgent` (`src/agents/refiner.py`) replaces the generic SelectorAgent call in the optimize loop; receives full incumbent state (config, leaderboard, overfitting_gap, prior runs) and produces a targeted one-step ExperimentPlan
- `prompts/refiner.md` encodes 5 decision rules: overfitting → reduce complexity; homogeneous families → diversify; plateau → change validation; failed run → avoid those families; otherwise → add new family

**Why:**
The optimize loop called SelectorAgent with a generic "refine this config" string; the LLM had no access to what the incumbent config contained, which models trained, or whether overfitting was occurring. RefinerAgent gives the LLM the evidence it needs for principled decisions.

**Must remain true:**
- `leaderboard(extra_info=True)` failure falls back to basic leaderboard (overfitting_gap stays None — acceptable)
- RefinerAgent, SelectorAgent, and IdeatorAgent all share the same retry + fence-strip pattern
- `RunDiagnostics` fields remain Optional — never required for correctness, only improve agent decisions

## 2026-03-19 — Phase 2: Memory & Ideation

**New modules:**
- `src/memory/case_store.py` — append-only JSONL cross-session knowledge base (CaseStore)
- `src/memory/retrieval.py` — cosine similarity ranking on TaskTraits feature vectors (CaseRetriever)
- `src/memory/distiller.py` — LLM-assisted session → CaseEntry summarisation (Distiller)
- `src/memory/context_builder.py` — pure assembly of SearchContext from session state (ContextBuilder)
- `src/memory/trait_utils.py` — shared bucket helpers (rows_bucket, features_bucket, balance_bucket)
- `src/agents/ideator.py` — LLM hypothesis generation grounded in data profile + past cases (IdeatorAgent)

**Changed:**
- `src/session.py` — replaced static seed_ideas with IdeatorAgent; added CaseStore retrieval before ideation; replaced inline SearchContext with ContextBuilder; added distillation to CaseStore at session end (non-fatal)
- `main.py` — passes case_store_path from configs/project.yaml to ExperimentSession

**Effect:** The agent is no longer stateless. Each session retrieves similar past cases via cosine similarity on task traits and starts with informed hypotheses. At session end, the run history is distilled into a CaseEntry and persisted to CaseStore for future sessions.

**Tests:** 66 passing

## Phase 4a: Campaign Orchestration (2026-03-20)

**What changed:**
- Added `CampaignOrchestrator` (`src/orchestration/campaign.py`) — outer loop over multiple `ExperimentSession`s, stops on metric plateau or session budget
- Added `CampaignConfig`, `SessionSummary`, `CampaignResult` models (`src/models/campaign.py`)
- Added `PreprocessingPlan` model + stub `PreprocessingExecutor` (`src/models/preprocessing.py`, `src/execution/preprocessing_runner.py`)
- `ExperimentSession` gains `preprocessed_data_path` param; data profile cached eagerly as `self._data_profile`
- Added `campaign.py` entrypoint (root); `configs/project.yaml` gains `campaign:` section

**Why:**
- Enable multi-session search campaigns that improve over time (plateau → stop)
- Phase 4b will replace identity preprocessing with LLM-generated transforms

**Tradeoffs:**
- `PreprocessingExecutor` is a stub (identity copy only); Phase 4b adds code gen
- `CampaignOrchestrator` has 6 pass-through params to `ExperimentSession` (deferred `SessionConfig` refactor, see TODOS.md)
- Cross-session IdeatorAgent history deferred to Phase 4b (no signal when all sessions use same data)

**Must remain true:**
- `session._data_path` must be used in BOTH `profile_data()` (line ~121) AND `execute_node()` (line ~180) — otherwise preprocessing is silently ignored for training
- `campaign.json` must be written after EACH session (crash-survival guarantee)
- `get_incumbent(higher_is_better=...)` kwarg must be passed explicitly
