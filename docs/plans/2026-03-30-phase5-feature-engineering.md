# Phase 5: Ecommerce Feature Engineering Subsystem

> **For Claude:** Use obra-executing-plans to implement this plan step-by-step.
>
> **Governing architecture:** `docs/plans/2026-03-30-ecommerce-feature-engineering-design.md` — the design doc is the source of truth for architectural constraints and rationale. This plan is the execution sequence.

**Hard constraint — no vector RAG in the new path.** The old `CaseStore`, `PreprocessingStore`, and `EmbeddingRetriever` are not part of the target architecture for the feature engineering subsystem. "No memory" is wrong — "no semantic retrieval layer" is the rule. Empirical experiment logs and static reference packs are the two permitted knowledge sources.

**Goal:** Add a `FeatureEngineeringAgent` subsystem that proposes, validates, executes, and learns from candidate predictive signals for ecommerce lifecycle tasks (churn, repurchase, lifetime value). Uses bounded `template + DSL` execution by default with a codegen escape hatch. Replaces RAG/vector retrieval with a simpler split: empirical experiment memory plus static reference packs. Retires `CaseStore`, `PreprocessingStore`, and `EmbeddingRetriever` from the active architecture of the new feature system.

**Architecture additions:**
- `src/models/feature_engineering.py` — Pydantic contracts: FeatureDecision, FeatureSpec variants, FeatureAuditVerdict, FeatureExecutionResult, FeatureHistoryEntry
- `src/features/dsl.py` — DSL operator surface validation (14 operators, time-op guards)
- `src/features/registry.py` — Template registry mapping names to pandas implementations
- `src/features/templates/customer.py` — RFM features (recency, frequency, monetary)
- `src/features/templates/order.py` — Order/basket features (AOV, basket size, category diversity)
- `src/features/templates/temporal.py` — Windowed time features (days_since, count/sum/mean/nunique_in_window)
- `src/features/templates/transforms.py` — Column transforms (log1p, clip, bucketize, is_missing)
- `src/features/templates/composites.py` — Composite ops (safe_divide, subtract, add, multiply, ratio_to_baseline)
- `src/features/executor.py` — BoundedExecutor: dispatches DSL configs to template functions
- `src/features/validator.py` — Feature validation (row count, target preservation, no all-null/constant)
- `src/features/history.py` — FeatureHistoryStore: JSONL append-only empirical experiment memory for feature iterations
- `src/features/context_builder.py` — Assembles feature engineering context for agent prompt
- `src/features/sandbox.py` — Subprocess-isolated codegen executor (Phase 2)
- `src/agents/feature_engineer.py` — FeatureEngineeringAgent: decision → leakage audit → execute pipeline
- `src/orchestration/feature_campaign.py` — FeatureCampaignOrchestrator: sibling to existing campaign
- `src/memory/feature_store.py` — Re-export of FeatureHistoryStore
- `prompts/feature_engineering/feature_engineer_full.md` — Decision prompt with domain knowledge + JSON contract
- `prompts/feature_engineering/feature_engineer_router.md` — Short routing variant
- `prompts/feature_engineering/feature_leakage_audit.md` — Leakage auditor prompt (5 checks)
- `prompts/feature_engineering/feature_codegen.md` — Codegen generation prompt (Phase 2)
- `prompts/feature_engineering/feature_codegen_guardrail.md` — Codegen review prompt (Phase 2)
- `references/feature_engineering/ecommerce_features.md` — Curated ecommerce feature knowledge loaded into runtime prompts
- `references/feature_engineering/leakage_patterns.md` — Known ecommerce leakage patterns loaded into runtime prompts

**Key decisions from design doc:**
- Single `FeatureEngineeringAgent` node in campaign, multi-LLM-call pipeline is internal detail
- `leakage_audit_call` is mandatory on both bounded and codegen paths — no bypass
- DSL is config-shaped (JSON objects), not string-formula-based
- Time-based features require `entity_key`, `time_col`, explicit window, cutoff semantics — leakage defense by construction
- `escalate_codegen` is explicit escape hatch, not default — agent must explain why bounded path is insufficient
- BoundedExecutor runs in-process (no subprocess) — only pre-approved templates with validated params
- CodegenSandbox uses subprocess isolation (same pattern as ValidationHarness)
- Existing `CampaignOrchestrator` untouched — `FeatureCampaignOrchestrator` is a sibling, not a replacement
- Preprocessing stays identity inside feature campaigns for v1
- No vector retrieval in the new path — empirical memory stays via run history and `FeatureHistoryStore`
- Static reference packs complement experiment memory; they do not replace it
- Runtime references live in repo `references/`, not in developer skill folders

**Build order:** Step 1–2 (Models + DSL) → Step 3–7 (Templates + Registry + Executor + Validator + Memory) → Step 8–10 (Agent + Prompts + References) → Step 11–13 (Campaign + Cleanup + Docs) → Step 14–16 (Codegen Escape Hatch)

**Tech Stack:** Python 3.12, Pydantic v2, pandas, subprocess (codegen only), pytest

---

## Step 1: Feature Engineering Models

**Files:**
- Create: `src/models/feature_engineering.py`

**What:** All Pydantic contracts for the feature engineering subsystem. Discriminated union for `FeatureSpec` using `spec_type` literal. Follow pattern from `src/models/preprocessing.py`.

```python
class FeatureDecision(BaseModel):
    status: Literal["proposed", "blocked", "skip"]
    action: Literal["add", "drop", "transform", "composite", "request_context", "blocked", "escalate_codegen"]
    reasoning: str
    feature_spec: Optional[FeatureSpec] = None  # discriminated union
    expected_impact: str = ""
    risk_flags: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    facts_to_save: List[str] = Field(default_factory=list)

class TemplateFeatureSpec(BaseModel):
    spec_type: Literal["template"] = "template"
    template_name: str
    params: Dict[str, Any] = Field(default_factory=dict)

class TransformFeatureSpec(BaseModel):
    spec_type: Literal["transform"] = "transform"
    input_col: str
    op: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output_col: str

class CompositeFeatureSpec(BaseModel):
    spec_type: Literal["composite"] = "composite"
    name: str
    op: str
    inputs: List[Dict[str, Any]]  # [{"ref": "col"} | {"literal": val}]
    post: List[str] = Field(default_factory=list)

class CodegenEscalationSpec(BaseModel):
    spec_type: Literal["codegen"] = "codegen"
    description: str
    reason_bounded_insufficient: str
    code: Optional[str] = None

FeatureSpec = Annotated[
    Union[TemplateFeatureSpec, TransformFeatureSpec, CompositeFeatureSpec, CodegenEscalationSpec],
    Field(discriminator="spec_type"),
]

class FeatureAuditVerdict(BaseModel):
    verdict: Literal["pass", "block", "warn"]
    reasons: List[str] = Field(default_factory=list)
    required_fixes: List[str] = Field(default_factory=list)

class FeatureExecutionResult(BaseModel):
    status: Literal["success", "failed", "blocked"]
    output_path: Optional[str] = None
    produced_columns: List[str] = Field(default_factory=list)
    dropped_columns: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None

class FeatureHistoryEntry(BaseModel):
    entry_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str
    feature_spec_json: str
    dataset_name: str
    task_type: str
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None
    observed_outcome: str = ""
    distilled_takeaway: str = ""
    audit_verdict: str = ""
```

**Test:** `tests/models/test_feature_engineering.py`
```python
def test_feature_decision_round_trip()          # model_dump_json → model_validate_json
def test_discriminated_union_template()         # parse TemplateFeatureSpec via spec_type
def test_discriminated_union_transform()        # parse TransformFeatureSpec via spec_type
def test_discriminated_union_composite()        # parse CompositeFeatureSpec via spec_type
def test_discriminated_union_codegen()          # parse CodegenEscalationSpec via spec_type
def test_invalid_action_rejected()              # action="invalid" → ValidationError
def test_feature_history_entry_defaults()       # timestamp populated, optional fields None
def test_feature_audit_verdict_defaults()       # reasons and required_fixes default to []
```

---

## Step 2: DSL Validation

**Files:**
- Create: `src/features/__init__.py`
- Create: `src/features/dsl.py`

**What:** Define the 14-operator surface and validate DSL configs. Time-requiring ops enforce `entity_key`, `time_col`, `window` in params — this is a primary leakage defense.

```python
VALID_OPS = {
    "safe_divide", "subtract", "add", "multiply", "ratio_to_baseline",
    "log1p", "clip", "bucketize", "is_missing", "days_since",
    "count_in_window", "sum_in_window", "mean_in_window", "nunique_in_window",
}

TIME_REQUIRING_OPS = {
    "days_since", "count_in_window", "sum_in_window", "mean_in_window", "nunique_in_window",
}

def validate_dsl_config(spec: Union[TransformFeatureSpec, CompositeFeatureSpec]) -> List[str]:
    """Returns list of error strings (empty = valid)."""
    # 1. op in VALID_OPS
    # 2. time ops require entity_key, time_col, window in params
    # 3. composite inputs non-empty
    # 4. post ops recognized

def parse_post_ops(post: List[str]) -> List[Callable]:
    """Returns list of post-processing callables (clip_0_1, log1p, etc.)."""
```

**Test:** `tests/features/test_dsl.py`
```python
def test_valid_op_accepted()                    # "safe_divide" → no errors
def test_invalid_op_rejected()                  # "foobar" → error string
def test_time_op_requires_entity_key()          # "count_in_window" without entity_key → error
def test_time_op_requires_time_col()            # "days_since" without time_col → error
def test_time_op_requires_window()              # "sum_in_window" without window → error
def test_composite_inputs_required()            # empty inputs → error
def test_post_ops_parsed()                      # ["clip_0_1", "log1p"] → callables
def test_unknown_post_op_rejected()             # ["unknown_op"] → error
```

---

## Step 3: Template Registry

**Files:**
- Create: `src/features/registry.py`

**What:** Maps template names to implementation functions. Template functions follow signature `(df: pd.DataFrame, **params) -> pd.DataFrame`.

```python
class TemplateRegistry:
    def __init__(self) -> None:
        self._templates: Dict[str, Callable] = {}

    def register(self, name: str, fn: Callable) -> None: ...
    def get(self, name: str) -> Optional[Callable]: ...
    def list_templates(self) -> List[str]: ...
    def execute(self, name: str, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame: ...
```

**Test:** `tests/features/test_registry.py`
```python
def test_register_and_get()                     # register fn, get returns it
def test_get_missing_returns_none()             # unknown name → None
def test_list_templates()                       # returns all registered names
def test_execute_calls_template()               # verify fn called with df + params
def test_execute_missing_raises()               # unknown template → KeyError or similar
```

---

## Step 4: Template Implementations

**Files:**
- Create: `src/features/templates/__init__.py`
- Create: `src/features/templates/customer.py`
- Create: `src/features/templates/order.py`
- Create: `src/features/templates/temporal.py`
- Create: `src/features/templates/transforms.py`
- Create: `src/features/templates/composites.py`

**What:** Ecommerce feature template functions. All windowed templates require `entity_key` + `time_col` — leakage defense by construction. Each function takes a DataFrame and keyword params, returns a DataFrame with new/modified columns.

**`customer.py`:**
- `rfm_recency(df, entity_key, time_col, cutoff_col=None, output_col="recency_days")` — days since last transaction
- `rfm_frequency(df, entity_key, time_col, window_days=365, output_col="frequency")` — transaction count in window
- `rfm_monetary(df, entity_key, amount_col, window_days=365, output_col="monetary")` — sum of amounts in window

**`order.py`:**
- `avg_order_value(df, entity_key, amount_col, order_id_col, output_col="avg_order_value")`
- `basket_size(df, entity_key, order_id_col, item_col, output_col="avg_basket_size")`
- `category_diversity(df, entity_key, category_col, output_col="category_nunique")`

**`temporal.py`:**
- `days_since(df, time_col, reference_date, output_col)`
- `count_in_window(df, entity_key, time_col, window_days, output_col)`
- `sum_in_window(df, entity_key, time_col, value_col, window_days, output_col)`
- `mean_in_window(df, entity_key, time_col, value_col, window_days, output_col)`
- `nunique_in_window(df, entity_key, time_col, value_col, window_days, output_col)`

**`transforms.py`:**
- `log1p_transform(df, input_col, output_col)` — `np.log1p`
- `clip_transform(df, input_col, lower, upper, output_col)` — `np.clip`
- `bucketize_transform(df, input_col, bins, labels=None, output_col=None)` — `pd.cut`
- `is_missing_transform(df, input_col, output_col)` — `col.isnull().astype(int)`

**`composites.py`:**
- `safe_divide(df, numerator_col, denominator_col, output_col, fill_value=0.0)`
- `subtract_cols(df, col_a, col_b, output_col)`
- `add_cols(df, col_a, col_b, output_col)`
- `multiply_cols(df, col_a, col_b, output_col)`
- `ratio_to_baseline(df, col, baseline_col, output_col)`

**Test:** `tests/features/test_templates.py` — each function on a small synthetic DataFrame (5–10 rows), verify output columns exist and values are correct. Test edge cases: missing columns, zero denominators, empty windows.

---

## Step 5: BoundedExecutor

**Files:**
- Create: `src/features/executor.py`

**What:** Dispatches `FeatureDecision` to template functions via registry. Never raises — returns `FeatureExecutionResult(status="failed")` on error. Follow pattern from `src/execution/preprocessing_runner.py`.

```python
class BoundedExecutor:
    def __init__(self, registry: TemplateRegistry) -> None:
        self._registry = registry

    def execute(self, df: pd.DataFrame, decision: FeatureDecision) -> FeatureExecutionResult:
        # Dispatch on decision.action + feature_spec.spec_type:
        #   TemplateFeatureSpec → registry.execute(template_name, df, params)
        #   TransformFeatureSpec → resolve op, apply to column
        #   CompositeFeatureSpec → resolve inputs, apply op, apply post-ops
        #   "drop" action → remove specified columns
        # Returns FeatureExecutionResult (never raises)
```

**Test:** `tests/features/test_executor.py`
```python
def test_template_spec_executes()               # TemplateFeatureSpec → registry called
def test_transform_spec_executes()              # TransformFeatureSpec → op applied to column
def test_composite_spec_executes()              # CompositeFeatureSpec → inputs resolved, op applied
def test_drop_action()                          # columns removed from DataFrame
def test_unknown_template_returns_failed()      # missing registry entry → status="failed"
def test_exception_returns_failed()             # template raises → status="failed", failure_reason set
def test_produced_columns_populated()           # new columns listed in result
```

---

## Step 6: FeatureValidator

**Files:**
- Create: `src/features/validator.py`

**What:** Validates feature execution output. Returns list of warning/error strings (empty = valid).

```python
class FeatureValidator:
    def validate_result(
        self,
        original_df: pd.DataFrame,
        result_df: pd.DataFrame,
        target_col: str,
    ) -> List[str]:
        # 1. Row count preserved (within tolerance)
        # 2. Target column preserved and unchanged
        # 3. No all-null new columns
        # 4. No constant-value new columns
        # 5. No NaN explosion in existing columns
```

**Test:** `tests/features/test_validator.py`
```python
def test_valid_result_no_warnings()             # good output → empty list
def test_row_count_changed()                    # fewer rows → error
def test_target_column_missing()                # target dropped → error
def test_target_column_modified()               # target values changed → error
def test_all_null_new_column()                  # new col all NaN → warning
def test_constant_new_column()                  # new col single value → warning
def test_nan_explosion()                        # existing col gains many NaNs → warning
```

---

## Step 7: FeatureHistoryStore

**Files:**
- Create: `src/features/history.py`
- Create: `src/memory/feature_store.py`

**What:** JSONL append-only store for `FeatureHistoryEntry`. This is empirical experiment memory for the feature loop: it stores what we tried, what happened, and what to remember later. Follows `src/memory/run_store.py` pattern exactly.

```python
class FeatureHistoryStore:
    def __init__(self, journal_path: Union[str, Path]) -> None:
        self._path = Path(journal_path)
        self._entries: List[FeatureHistoryEntry] = []
        if self._path.exists():
            self._load()

    def _load(self) -> None: ...        # line-by-line model_validate_json
    def add(self, entry: FeatureHistoryEntry) -> None: ...  # append to list + atomic write
    def get_history(self) -> List[FeatureHistoryEntry]: ...
    def get_by_dataset(self, dataset_name: str) -> List[FeatureHistoryEntry]: ...
    def get_recent(self, n: int = 10) -> List[FeatureHistoryEntry]: ...
```

`src/memory/feature_store.py` is a thin re-export: `from src.features.history import FeatureHistoryStore`

**Important:** this store is memory, not a reference pack. It captures our own historical feature-engineering experience. Static external knowledge belongs under `references/feature_engineering/`.

**Test:** `tests/features/test_history.py`
```python
def test_add_and_get_history()                  # add entry, get_history returns it
def test_persists_to_jsonl()                    # add entry, new store instance loads it
def test_get_by_dataset()                       # filter by dataset_name
def test_get_recent()                           # returns last n entries
def test_empty_store()                          # no file → empty list
```

---

## Step 8: FeatureContextBuilder

**Files:**
- Create: `src/features/context_builder.py`

**What:** Assembles feature engineering context for agent prompt. No LLM, no IO — pure string assembly. Follow pattern from `src/memory/context_builder.py`.

```python
class FeatureContextBuilder:
    def build(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        leaderboard: List[dict],
        feature_importances: Dict[str, float],
        history: List[FeatureHistoryEntry],
        incumbent_metric: Optional[float],
        available_templates: List[str],
        budget_remaining: int,
        budget_used: int,
    ) -> str:
        # Returns formatted text context with sections for:
        # task info, data profile, model performance, importances,
        # available templates, feature history, loaded references, budget
```

**Test:** `tests/features/test_context_builder.py`
```python
def test_context_includes_all_sections()        # verify task, profile, templates, history present
def test_empty_history()                        # no prior features → still valid context
def test_reference_sections_included()          # reference knowledge loaded into output
def test_budget_shown()                         # budget_remaining and budget_used in output
```

---

## Step 9: FeatureEngineeringAgent

**Files:**
- Create: `src/agents/feature_engineer.py`

**What:** Internal pipeline: `decision_call` → `leakage_audit_call` → `execute_bounded`. Follow `src/agents/refiner.py` pattern (constructor reads prompt, retry on JSON failure, strips markdown fences). Never raises — returns safe defaults on exhaustion, following `src/agents/preprocessing_agent.py` pattern.

```python
class FeatureEngineeringAgent:
    def __init__(
        self,
        llm: LLMBackend,
        prompt_path: str = "prompts/feature_engineering/feature_engineer_full.md",
        leakage_prompt_path: str = "prompts/feature_engineering/feature_leakage_audit.md",
        registry: Optional[TemplateRegistry] = None,
        max_retries: int = 3,
    ) -> None:
        self._llm = llm
        self._system_prompt = Path(prompt_path).read_text()
        self._leakage_prompt = Path(leakage_prompt_path).read_text()
        self._executor = BoundedExecutor(registry or TemplateRegistry())
        self._validator = FeatureValidator()
        self._context_builder = FeatureContextBuilder()
        self._max_retries = max_retries

    def propose_and_execute(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        df: pd.DataFrame,
        leaderboard: List[dict],
        feature_importances: Dict[str, float],
        history: List[FeatureHistoryEntry],
        incumbent_metric: Optional[float],
        budget_remaining: int,
        budget_used: int,
    ) -> Tuple[FeatureDecision, FeatureAuditVerdict, FeatureExecutionResult]:
        # 1. Build context via FeatureContextBuilder
        # 2. _decision_call(context) → FeatureDecision
        # 3. If blocked/skip → return early
        # 4. If escalate_codegen → return blocked (Phase 1)
        # 5. _leakage_audit_call(decision, data_profile) → FeatureAuditVerdict
        # 6. If verdict=block → return blocked execution result
        # 7. _execute_bounded(decision, df) → FeatureExecutionResult
        # 8. On any exception → return safe defaults (never raises)
```

**`_decision_call(context) -> FeatureDecision`:**
- Build `List[Message]` with system prompt + user context
- LLM call, strip fences, parse JSON into `FeatureDecision`
- Retry up to `max_retries` on parse failure (append assistant + error user message)
- On exhaustion: return `FeatureDecision(status="blocked", action="blocked", reasoning="parse failure")`

**`_leakage_audit_call(decision, data_profile) -> FeatureAuditVerdict`:**
- System prompt from `leakage_prompt_path`
- User message includes serialized `FeatureDecision` + `DataProfile`
- LLM call, parse into `FeatureAuditVerdict`
- On parse failure: return `FeatureAuditVerdict(verdict="block", reasons=["audit parse failure"])`

**Test:** `tests/agents/test_feature_engineer.py`
```python
def test_happy_path_bounded_feature()           # mock LLM returns valid decision + pass audit → success
def test_decision_blocked_returns_early()       # action="blocked" → no audit, no execution
def test_decision_skip_returns_early()          # status="skip" → no audit, no execution
def test_escalate_codegen_blocked_phase1()      # action="escalate_codegen" → blocked execution
def test_leakage_audit_blocks_execution()       # verdict="block" → FeatureExecutionResult(status="blocked")
def test_invalid_json_retries()                 # malformed JSON → retry → eventually succeeds
def test_all_retries_exhausted()                # 3 bad JSONs → blocked decision
def test_exception_returns_safe_defaults()      # LLM raises → never propagates
```

---

## Step 10: Prompt Assets + Reference Folder

**Files:**
- Create: `prompts/feature_engineering/feature_engineer_full.md`
- Create: `prompts/feature_engineering/feature_engineer_router.md`
- Create: `prompts/feature_engineering/feature_leakage_audit.md`
- Create: `prompts/feature_engineering/feature_codegen.md` (placeholder)
- Create: `prompts/feature_engineering/feature_codegen_guardrail.md` (placeholder)
- Create: `references/feature_engineering/ecommerce_features.md`
- Create: `references/feature_engineering/leakage_patterns.md`

**Architecture note:** these reference files are runtime assets checked into the repo. They may be curated by developer skills, but the runtime source of truth stays in `references/`, not `.claude/skills/`.

**Runtime boundary (from design doc):** prompt assets and reference loading are runtime concerns. Template extension, reference curation, leakage-rule authoring, and codegen guardrail authoring are developer workflows — they happen outside the hot path and must not be mixed into runtime code.

**`feature_engineer_full.md` must include:**
- Role: expert ecommerce feature engineer
- Available templates with names and param signatures
- DSL operator surface (14 ops)
- `FeatureDecision` JSON wire format with all fields and allowed values
- Instructions: prefer bounded templates, escalate only when DSL is insufficient
- Examples: one `add` template action, one `composite` action, one `blocked` action

**`feature_leakage_audit.md` must include:**
- Role: leakage auditor reviewing proposed features
- 5 leakage checks: target column usage, future-looking timestamps, post-outcome joins, unbounded aggregations, missing entity/time semantics
- `FeatureAuditVerdict` JSON wire format
- Examples: one pass verdict, one block verdict with reasons

**`ecommerce_features.md`:**
- RFM feature family (recency, frequency, monetary) with typical column mappings
- Temporal features: seasonality, day-of-week, time-since patterns
- Basket features: AOV, basket size, category diversity
- Ratio features: cart-to-purchase rate, discount ratio, return rate
- Common pitfalls per feature family

**`leakage_patterns.md`:**
- Target column appearing in feature computation
- Future-looking timestamps (using data after prediction time)
- Post-outcome joins (joining tables on events that happen after the target event)
- Unbounded aggregations (all-time stats that include future)
- Missing entity boundaries (cross-customer leakage in aggregations)

---

## Step 11: FeatureCampaignOrchestrator

**Files:**
- Extend: `src/models/campaign.py` — add `FeatureCampaignConfig`
- Create: `src/orchestration/feature_campaign.py`

**What:** Sibling to `CampaignOrchestrator`. Runs baseline session, then iterates feature engineering proposals. Follow `src/orchestration/campaign.py` pattern closely.

**`FeatureCampaignConfig` (add to `src/models/campaign.py`):**
```python
class FeatureCampaignConfig(BaseModel):
    max_sessions: int = 10
    max_feature_iterations: int = 10
    plateau_threshold: float = 0.002
    plateau_window: int = 3
    feature_history_path: str = "experiments/feature_history.jsonl"
    max_consecutive_blocks: int = 3
    max_consecutive_codegen_failures: int = 2
```

**`FeatureCampaignOrchestrator`:**
```python
class FeatureCampaignOrchestrator:
    def __init__(
        self,
        task: TaskSpec,
        llm: LLMBackend,
        config: Optional[FeatureCampaignConfig] = None,
        experiments_dir: str = "experiments",
        num_candidates: int = 3,
        max_optimize_iterations: int = 5,
        higher_is_better: bool = True,
    ) -> None:
        # Init: FeatureEngineeringAgent, FeatureHistoryStore, TemplateRegistry
        # No PreprocessingStore, no EmbeddingRetriever, no CaseStore
        # Uses empirical memory + static reference packs only

    def run(self) -> CampaignResult:
        # 1. Baseline ExperimentSession (identity preprocessing)
        # 2. Collect: DataProfile, leaderboard, feature importances
        # 3. Feature iteration loop (up to max_feature_iterations):
        #    a. FeatureContextBuilder.build()
        #    b. FeatureEngineeringAgent.propose_and_execute()
        #    c. If blocked/failed: increment consecutive_blocks, check stop
        #    d. If success: save feature-engineered CSV,
        #       retrain via new ExperimentSession(preprocessed_data_path=...)
        #    e. Store FeatureHistoryEntry with before/after metrics
        #    f. Check plateau, budget, consecutive blocks
        # 4. Return campaign result
```

**Stop conditions:** budget exhausted, plateau (metric moves < threshold across window), `max_consecutive_blocks` reached, `max_consecutive_codegen_failures` reached.

**Test:** `tests/orchestration/test_feature_campaign.py`
```python
def test_baseline_session_runs_first()          # first session uses identity preprocessing
def test_feature_loop_calls_agent()             # mock agent called with correct context
def test_stop_on_budget()                       # budget exhausted → stopped_reason="budget"
def test_stop_on_plateau()                      # metric stagnates → stopped_reason="plateau"
def test_stop_on_consecutive_blocks()           # N blocked decisions → stop
def test_history_store_receives_entries()        # FeatureHistoryEntry persisted per iteration
def test_successful_feature_triggers_retrain()  # success → new ExperimentSession created
```

---

## Step 12: Retire Vector-RAG Modules

**Files:**
- Extend: `src/memory/case_store.py` — add deprecation docstring
- Extend: `src/memory/preprocessing_store.py` — add deprecation docstring
- Extend: `src/memory/embedding_retriever.py` — add deprecation docstring

**What:** Add module-level retirement docstrings. Do NOT delete these files yet — existing `CampaignOrchestrator` still imports them. The new `FeatureCampaignOrchestrator` does not import them. These modules are retired from the active architecture, not immediately removed from the repository.

**Deprecation docstring format:**
```python
"""
RETIRED FROM ACTIVE ARCHITECTURE: This module is retained only for backward compatibility with CampaignOrchestrator.
New feature engineering campaigns use empirical experiment memory plus static reference packs instead.
See docs/plans/2026-03-30-ecommerce-feature-engineering-design.md for rationale.
"""
```

---

## Step 13: Update Docs

**Files:**
- Update: `docs/architecture/current-state.md`
- Append: `docs/changes/implementation-log.md`

**`current-state.md` updates:**
- Add Feature Engineering layer to architecture diagram
- Describe `FeatureEngineeringAgent` pipeline
- Describe `FeatureCampaignOrchestrator` as sibling campaign
- Note retired vector-retrieval modules (CaseStore, PreprocessingStore, EmbeddingRetriever)
- Clarify the knowledge split: empirical experiment memory vs static reference packs
- Update phase to "Phase 5"

**`implementation-log.md` entry:**
- What: Added ecommerce feature engineering subsystem (FeatureEngineeringAgent, bounded template + DSL, FeatureCampaignOrchestrator)
- Why: Replace RAG/vector retrieval with a simpler split between empirical experiment memory and static reference packs; focus on ecommerce lifecycle tasks
- Tradeoff: Less flexible than broad codegen default, but safer leakage guarantees and deterministic bounded features
- Must remain true: leakage audit is mandatory before any execution path; existing CampaignOrchestrator untouched

---

## Step 14: CodegenSandbox

**Files:**
- Create: `src/features/sandbox.py`

**What:** Subprocess-isolated executor for generated code. Follow `src/execution/validation_harness.py` pattern exactly. Never raises — returns `FeatureExecutionResult(status="failed")` on error.

```python
class CodegenSandbox:
    def __init__(self, timeout: int = 60) -> None:
        self._timeout = timeout

    def execute(self, code: str, data_path: str, target_col: str) -> FeatureExecutionResult:
        # 1. Write code to temp file
        # 2. Run in subprocess with timeout
        # 3. Validate output (superset of ValidationHarness 6 checks):
        #    - No exception
        #    - Returns DataFrame
        #    - Row count: len(result) <= 2x len(original) (no explosion)
        #    - Target column preserved and unchanged
        #    - No all-null or constant new columns
        #    - Target reference scan (code should not read/write target)
        # 4. Return FeatureExecutionResult (never raises)
```

**Test:** `tests/features/test_sandbox.py`
```python
def test_valid_code_executes()                  # adds a column → success
def test_code_crashes_returns_failed()          # raises → status="failed"
def test_code_timeout_returns_failed()          # infinite loop → status="failed"
def test_row_explosion_caught()                 # output 3x rows → status="failed"
def test_target_reference_caught()              # code reads target col → status="failed"
def test_all_null_column_caught()               # new col all NaN → warning in result
```

---

## Step 15: Wire Codegen into Agent

**Files:**
- Extend: `src/agents/feature_engineer.py`

**What:** Add codegen private methods and update `propose_and_execute` to handle `escalate_codegen`.

**New private methods:**
```python
def _codegen_call(self, escalation_spec: CodegenEscalationSpec, context: str) -> str:
    # LLM call using feature_codegen.md prompt
    # Returns generated code string

def _codegen_guardrail_call(self, code: str, data_profile: DataProfile) -> FeatureAuditVerdict:
    # LLM review using feature_codegen_guardrail.md prompt
    # Returns FeatureAuditVerdict
```

**Update `propose_and_execute` for `action == "escalate_codegen"`:**
1. `_leakage_audit_call` on the escalation spec
2. `_codegen_call` to generate code
3. `_codegen_guardrail_call` to review code
4. If guardrail verdict=block: return blocked result
5. `CodegenSandbox.execute()` to run code
6. `FeatureValidator.validate_result()` on output
7. Return result

---

## Step 16: Finalize Codegen Prompts

**Files:**
- Finalize: `prompts/feature_engineering/feature_codegen.md`
- Finalize: `prompts/feature_engineering/feature_codegen_guardrail.md`

**`feature_codegen.md` must include:**
- Role: expert code generator for feature engineering
- Function signature requirement: `def engineer_features(df: pd.DataFrame) -> pd.DataFrame`
- Safety rules: no network access, no file system writes, import restrictions (pandas, numpy only)
- Target column avoidance rule
- Defensive coding requirements (check column existence, handle edge cases)

**`feature_codegen_guardrail.md` must include:**
- Role: security reviewer for generated feature code
- Check: target column referenced in computation → block
- Check: future-time patterns → block
- Check: join blow-ups (cross joins, exploding merges) → block
- Check: unsafe imports (os, subprocess, requests) → block
- `FeatureAuditVerdict` JSON wire format

**Test:** `tests/agents/test_feature_engineer_codegen.py`
```python
def test_codegen_happy_path()                   # escalate → generate → guardrail pass → sandbox pass
def test_codegen_guardrail_blocks()             # guardrail verdict=block → no execution
def test_codegen_sandbox_fails()                # code crashes in sandbox → status="failed"
def test_codegen_leakage_audit_blocks()         # leakage audit on escalation spec → block before codegen
```

**Integration test:** `tests/integration/test_codegen_path.py`
```python
def test_forced_codegen_end_to_end()            # mock LLM, force escalate_codegen, verify full pipeline
```

---

## Completion Criteria

- [ ] Bounded path can express useful features on a synthetic ecommerce DataFrame without codegen
- [ ] Bounded features are deterministic and logged cleanly in `FeatureHistoryStore`
- [ ] `escalate_codegen` is exercised and cannot bypass guardrails
- [ ] Leakage audit blocks execution when it detects target usage or future-looking timestamps
- [ ] `FeatureCampaignOrchestrator` runs baseline + feature iterations, stops on plateau/budget/blocks
- [ ] Campaign outputs and feature history are replayable from JSONL
- [ ] Existing `CampaignOrchestrator` and preprocessing pipeline unaffected
- [ ] All new tests pass: `pytest tests/ -x`
- [ ] `docs/architecture/current-state.md` reflects the new subsystem
