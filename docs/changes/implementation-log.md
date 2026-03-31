## 2026-03-31 — Phase 5: Ecommerce Feature Engineering Subsystem

**What changed:**
- Added `FeatureEngineeringAgent` with internal pipeline: decision_call → leakage_audit_call → execute_bounded. Follows refiner.py retry pattern, preprocessing_agent.py never-raises pattern.
- Added bounded template + DSL execution path: 14 operators, 20 template functions across 5 modules (customer, order, temporal, transforms, composites). Time-based features enforce entity_key + time_col as leakage defense.
- Added `FeatureCampaignOrchestrator` as sibling to `CampaignOrchestrator`. Runs baseline session, then iterates feature proposals with plateau/budget/block stop conditions.
- Added `FeatureHistoryStore` (JSONL append-only) for empirical experiment memory. Static reference packs under `references/feature_engineering/` for domain knowledge.
- Added Pydantic contracts: FeatureDecision, FeatureSpec (discriminated union), FeatureAuditVerdict, FeatureExecutionResult, FeatureHistoryEntry.
- Retired `CaseStore`, `PreprocessingStore`, `EmbeddingRetriever` from active architecture (deprecation docstrings, not deletion — existing CampaignOrchestrator still imports them).

**Why:** Replace RAG/vector retrieval with simpler split between empirical experiment memory and static reference packs. Focus on ecommerce lifecycle tasks (churn, repurchase, LTV). Bounded templates give safer leakage guarantees than broad codegen.

**Tradeoff:** Less flexible than free codegen default, but deterministic bounded features with mandatory leakage audit. Codegen escape hatch (Phase 2) remains available for schemas the bounded path cannot handle.

**Must remain true:** Leakage audit is mandatory before any execution path — no bypass. Existing CampaignOrchestrator and preprocessing pipeline are untouched.

---

## 2026-03-22 — Phase 4c: Curated Seed Bank + Semantic Retrieval Wired End-to-End

**What changed:**
- `OpenAIBackend.embed_batch(texts)` — single batch API call to `embeddings.create(model, input=texts)`; sorted by `response.data[i].index` for guaranteed ordering; returns `[None] * len` on failure
- `PreprocessingStore.seed_from_file(path, embed_backend=None)` — loads seeds from JSONL, skips existing `entry_id`s (idempotent), calls `embed_batch` in one shot, appends each seed atomically via `add()`; logs warning if file missing
- `CampaignOrchestrator.__init__()` — computes `self._embed_backend = llm if hasattr(llm, "embed") else None`; auto-seeds empty bank from `data/seeds/preprocessing_seeds.jsonl`; warns at startup if bank has null-embedding entries and no embed_backend
- `CampaignOrchestrator._preprocessing_plan()` — two-stage retrieval: `get_similar(task_type, n=20)` filter → `EmbeddingRetriever(embed_backend).rank(query, candidates, top_k=3)` semantic ranking; query text is `"{task_type} task on {task_name}: {description}"`; falls back to `candidates[:3]` if no embed_backend
- `CampaignOrchestrator._store_preprocessing_entry()` — embeds `transformation_summary` via `embed_backend.embed()` before storing; null on failure
- `data/seeds/preprocessing_seeds.jsonl` — expanded from 5 to 20 entries across 5 archetypes: Titanic binary (seeds 001-004), credit risk binary (005-008), house prices regression (009-012), news multiclass (013-016), time series regression (017-020)

**Why:** The EmbeddingRetriever was wired but never had real vectors to rank — all 5 seeds had `embedding: null` and agent-generated entries were also stored without embeddings. Semantic retrieval was silently falling back to naive ordering on every call. This wires the full pipeline: seeds arrive pre-embedded, agent entries are embedded at write time.

**Tradeoff:** `embed_batch` is called at `CampaignOrchestrator.__init__()` time (blocking). With 20 seeds and OpenAI's batch endpoint, this is ~1 API call and takes <1s. AnthropicBackend has no `embed()` — naive ordering remains in effect when using Anthropic.

**Must remain true:**
- `PreprocessingAgent.generate()` never raises — already guaranteed
- Auto-seeding only runs when the bank is empty — non-empty banks are never modified at init
- `seed_from_file` is idempotent by `entry_id` — safe to call with overlapping seed files

---

## 2026-03-21 — Phase 4b: PreprocessingAgent + ValidationHarness + EmbeddingRetriever

**What changed:**
- `PreprocessingPlan` gains `validation_passed: bool` and `turns_used: int`
- `PreprocessingEntry` added to `src/models/preprocessing.py` — long-term memory record for successful preprocessing sessions, symmetric with `CaseEntry`
- `ValidationHarness` (`src/execution/validation_harness.py`) — validates generated `preprocess(df)` code in a subprocess (6 checks: no exception, returns DataFrame, shape ≥50%, target col present, no NaN in target, not identity)
- `PreprocessingAgent` (`src/agents/preprocessing_agent.py`) — ReAct loop (max 3 turns) that inspects columns and generates validated `preprocess(df)` code; falls back to identity on all errors; never raises
- `PreprocessingExecutor` extended with `strategy="generated"`: `exec()` the plan code, call `preprocess(df.copy())`, write result; falls back to identity on exec error
- `PreprocessingStore` (`src/memory/preprocessing_store.py`) — append-only JSONL store for `PreprocessingEntry`, symmetric with `CaseStore`; `get_similar()` filters by task_type
- `data/seeds/preprocessing_seeds.jsonl` — 5 hand-written seeds (titanic titles, family size, log transforms, house prices, news text) bootstrapping the preprocessing bank
- `OpenAIBackend.embed()` — calls `embeddings.create(model="text-embedding-3-small")`; returns `None` on failure (not ABC — only OpenAIBackend exposes this)
- `EmbeddingRetriever` (`src/memory/embedding_retriever.py`) — cosine-ranks `PreprocessingEntry` candidates by embedding similarity; A/B logs naive order vs embedding order; falls back to naive on embed failure
- `CaseEntry` gains `description_for_embedding: str` + `embedding: Optional[List[float]]`
- `Distiller` accepts optional `embed_backend: OpenAIBackend`; builds `description_for_embedding` from task traits + key decisions; calls `embed()` and stores vector; silently skips on failure
- `CampaignOrchestrator` wired end-to-end: session 0 always runs identity (warm-up baseline), subsequent sessions call `PreprocessingAgent.generate()`; logs `validation_passed` and `turns_used`; warns on 2+ consecutive preprocessing failures; stores successful entries in `PreprocessingStore`
- `SessionSummary` gains `preprocessing_validation_passed: bool` and `preprocessing_turns_used: int`
- `CampaignConfig` gains `preprocessing_bank_path: str` (default `experiments/preprocessing_bank.jsonl`)
- `configs/project.yaml` updated with `preprocessing_bank_path`

**Why:** Phase 4a always used identity preprocessing. Phase 4b gives the campaign loop an LLM-driven feature engineering step before each experiment session, grounded in memory of past transformations.

**Tradeoff:** `exec()` in PreprocessingExecutor re-runs the code outside the subprocess sandbox — acceptable because ValidationHarness already verified safety. A double-exec (subprocess + in-process) was considered but rejected as over-engineering for a research tool.

**Must remain true:**
- `PreprocessingAgent.generate()` never raises — all errors return `strategy="identity"`
- `ValidationHarness` runs in subprocess isolation — crashes in generated code cannot affect the main process
- Session 0 always runs identity — the first session is always the clean baseline

---

## 2026-03-20 — Retry logic in OpenAIBackend; NaN sanitization in ResultParser

**What changed:**
- `OpenAIBackend.complete()`: now retries up to 3 times with exponential backoff (2s, 4s) on status codes 400, 429, 500, 502, 503, 504; non-retryable errors (401, 403, etc.) raise immediately; logs warning on retryable failures, error on final failure
- `ResultParser.from_predictor`: sanitizes `score_train = NaN` → `None` (AutoGluon returns NaN for models without train scores); prevents `overfitting_gap = NaN` from propagating into `RunDiagnostics`

**Why:** Session 1 in the Phase 4a smoke test failed with a 400 "cannot parse JSON body" from OpenAI on the first optimize call. Root cause is most likely a transient API issue (all 4 warm-up calls and all 9 session-2 calls succeeded). Retry logic makes the orchestrator resilient to transient failures. NaN sanitization is defensive hygiene — pydantic v2 serializes NaN correctly but having NaN floats in diagnostics is an easy source of silent bugs.

**Tradeoff:** Adding retry with sleep increases the wall time of transient failures. Max added latency = 6s (2s + 4s). Acceptable for an ML experiment loop where each run takes seconds to minutes.

**Must remain true:** Non-retryable auth errors (401) must NOT be retried — they fail immediately.

---

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
