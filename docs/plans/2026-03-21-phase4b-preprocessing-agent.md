# Phase 4b: PreprocessingAgent, ValidationHarness, EmbeddingRetriever, Seeds

> **For Claude:** Use obra-executing-plans to implement this plan task-by-task.

**Goal:** Add a `PreprocessingAgent` that generates `preprocess(df)` code via a ReAct tool-use loop, validates it in a subprocess (`ValidationHarness`), stores results in a separate `preprocessing_bank.jsonl`, and replaces the hand-crafted 7-dim `CaseRetriever` with semantic `EmbeddingRetriever` using OpenAI `text-embedding-3-small`.

**Architecture additions:**
- `src/agents/preprocessing_agent.py` — ReAct loop with `inspect_column` tool, JSON wire format
- `src/execution/validation_harness.py` — subprocess isolation, 6-check validation, diff check
- `src/execution/preprocessing_runner.py` — extend stub: execute `generated` strategy
- `src/models/preprocessing.py` — extend `PreprocessingPlan` with `validation_passed`, `turns_used`
- `src/memory/preprocessing_store.py` — append-only JSONL, seeds-on-first-run
- `src/memory/embedding_retriever.py` — OpenAI embeddings, cosine similarity, A/B logging
- `src/llm/providers/openai.py` — add `embed(text) -> List[float]`
- `src/models/nodes.py` — add `description_for_embedding: str` to `CaseEntry`
- `src/memory/distiller.py` — populate embedding + write PreprocessingStore entry
- `src/orchestration/campaign.py` — wire PreprocessingAgent, log outcomes, warn on failures
- `prompts/preprocessing_agent.md` — ReAct system prompt
- `prompts/distiller.md` — add `description_for_embedding` field to output schema
- `data/seeds/preprocessing_seeds.jsonl` — 5 external seed patterns with working code

**Key decisions from CEO + Eng reviews:**
- Wire format: JSON `{"thought": "...", "action": "inspect_column" | "generate_code", "input": "..."}`
- `inspect_column` caches DataFrame in `__init__`, samples max 10K rows, returns error dict on missing col
- Max 3 turns; force `generate_code` message on turn 3 if still inspecting
- `EmbeddingRetriever(backend: OpenAIBackend)` — uses `backend.embed()`, not ABC
- `preprocessing_bank.jsonl` at `experiments/preprocessing_bank.jsonl` (global, symmetric with case_bank)
- `PreprocessingStore.get_similar()` Step 2 = return all entries (no ranking until EmbeddingRetriever wired)
- `EmbeddingRetriever.rank()` logs A/B comparison vs `CaseRetriever` for each query
- `CampaignOrchestrator` logs `plan.validation_passed` + `plan.turns_used` per session; warns on 2+ failures
- `Distiller.distill()` wraps embed in try/except → `case.embedding = None` on failure
- 3 critical try/except gaps: `inspect_column` missing col, `embed()` API failure, JSONL write OSError

**Build order:** Step 1 (Agent + Harness) → smoke test → Step 2 (Store + Seeds) → Step 3 (Embeddings) → Step 4 (Wire Campaign)

**Tech Stack:** Python 3.12, Pydantic v2, openai SDK, pandas, subprocess, pytest

---

## Step 1: Extend PreprocessingPlan model

**Files:**
- Extend: `src/models/preprocessing.py`

**What:** Add `validation_passed: bool = False` and `turns_used: int = 0` to `PreprocessingPlan`. These are populated by `PreprocessingAgent.generate()` and logged by `CampaignOrchestrator`.

```python
class PreprocessingPlan(BaseModel):
    strategy: Literal["identity", "generated"] = "identity"
    code: Optional[str] = None
    rationale: str = ""
    transformations: List[str] = []
    validation_passed: bool = False
    turns_used: int = 0
```

**Test:** Verify new fields have correct defaults; verify existing tests still pass.

---

## Step 2: ValidationHarness

**Files:**
- Create: `src/execution/validation_harness.py`
- Create: `tests/execution/test_validation_harness.py`

**What:** Runs generated `preprocess(df)` in a subprocess with 30s timeout. Returns a `ValidationResult(passed: bool, error: Optional[str])`. Six checks:
1. No exception raised in subprocess
2. Returns a DataFrame
3. `len(result) >= 0.5 * len(original)` — shape check
4. `target_col in result.columns` — target preserved
5. `result[target_col].isnull().sum() == 0` — no NaN in target
6. `set(result.columns) != set(original.columns) OR (result.values != original.values).any()` — diff check (not identity)

**Subprocess approach:** Write a small Python script to a temp file, run it with `sys.executable` via `subprocess.run(timeout=30)`. The script imports pandas, defines and calls `preprocess()`, then prints the result info as JSON.

**Module-level docstring must include the 6-check pipeline as ASCII comment.**

**Tests to write (write failing tests first):**
```python
def test_valid_code_passes()  # code that adds a column
def test_subprocess_timeout()  # infinite loop code, 30s cap
def test_preprocess_not_defined()  # NameError
def test_returns_non_dataframe()  # returns a dict
def test_drops_too_many_rows()  # filters out 60% of rows
def test_drops_target_column()  # del df["Survived"]
def test_identity_transform_fails_diff_check()  # return df unchanged
def test_dtype_only_change_passes()  # df["Age"] = df["Age"].astype(float)
```

---

## Step 3: PreprocessingAgent

**Files:**
- Create: `prompts/preprocessing_agent.md`
- Create: `src/agents/preprocessing_agent.py`
- Create: `tests/agents/test_preprocessing_agent.py`

**What:** ReAct loop. Agent receives task description + column list in initial user message. Calls `inspect_column` (returns stats + 5 sample values for a named column). Calls `generate_code` to emit a `preprocess(df)` function. ValidationHarness validates it. Max 3 turns total (inspect or generate count as turns). Force `generate_code` on turn 3.

**Wire format (JSON, all messages):**
```json
{"thought": "<agent reasoning>", "action": "inspect_column", "input": "Name"}
{"thought": "<agent reasoning>", "action": "generate_code", "input": "def preprocess(df):\n    ..."}
```

**Signature:**
```python
class PreprocessingAgent:
    def __init__(self, llm: LLMBackend, prompt_path: str = "prompts/preprocessing_agent.md") -> None:
        # loads prompt, will cache df once generate() is called

    def generate(
        self,
        task: TaskSpec,
        data_profile: DataProfile,
        data_path: str,
        similar_cases: List[PreprocessingCaseEntry],
    ) -> PreprocessingPlan:
        # returns identity plan on any error; never raises
```

**`inspect_column(col_name)` behavior:**
- Returns `{"col": col_name, "dtype": "...", "n_unique": N, "null_pct": 0.15, "sample_values": ["a", "b", ...]}`
- Returns `{"error": "column 'X' not found. Available: [col1, col2, ...]"}` if col missing

**Fallback on all 3 attempts fail:** return `PreprocessingPlan(strategy="identity", turns_used=3, validation_passed=False)`

**Turn 3 forced generate:** append system message: `"You have used 3 turns. You must now call generate_code with your best preprocess() function. No more inspect_column calls."`

**ASCII diagram in class docstring** (ReAct loop flow as shown in eng review).

**Prompt file (`prompts/preprocessing_agent.md`) must include:**
- Role: expert data scientist
- Available actions: inspect_column, generate_code
- Wire format spec
- Instruction to call inspect_column for interesting columns first
- Instruction to write defensive code (check if column exists before using it)
- Instruction to return df unchanged if no transformation applies

**Tests to write (mock LLM, use small Titanic-like CSV):**
```python
def test_happy_path_generates_valid_plan()     # mock returns inspect then generate → strategy="generated"
def test_all_attempts_fail_returns_identity()  # mock returns bad code 3 times → strategy="identity"
def test_malformed_json_retries()              # mock returns bad JSON first, valid JSON second
def test_missing_column_graceful()             # agent calls inspect_column("NoSuchCol") → gets error dict
def test_turns_used_populated()                # plan.turns_used > 0
def test_validation_passed_populated()         # plan.validation_passed matches harness result
```

---

## Step 4: Extend PreprocessingExecutor

**Files:**
- Extend: `src/execution/preprocessing_runner.py`
- Extend: `tests/execution/test_execution.py` (or create `tests/execution/test_preprocessing_runner.py`)

**What:** Replace `raise NotImplementedError` in the `generated` branch with real execution. Calls `ValidationHarness.validate()` on the plan code, then uses `subprocess` to execute it against the CSV and save the result.

**Note:** ValidationHarness already validates correctness. PreprocessingExecutor just needs to actually run the code and save the CSV.

**Approach:**
```python
if plan.strategy == "generated":
    # exec plan.code in subprocess, pass data_path, save to out_path
    # if exec fails, log warning and fall back to identity copy
    ...
```

---

## Step 5: PreprocessingStore + Seeds

**Files:**
- Create: `src/memory/preprocessing_store.py`
- Create: `data/seeds/preprocessing_seeds.jsonl`
- Create: `tests/memory/test_preprocessing_store.py`

**`PreprocessingCaseEntry` model (define in `src/memory/preprocessing_store.py` or in `src/models/`):**
```python
class PreprocessingCaseEntry(BaseModel):
    case_id: str
    task_type: str
    data_description: str        # "dataset with Name column containing passenger titles"
    transformation_summary: str  # "extracted title from Name, one-hot encoded Pclass"
    code: str                    # the preprocess() function string
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None
    embedding: Optional[List[float]] = None
    source: str = "empirical"    # "empirical" | "external"
```

**`PreprocessingStore`** — mirrors CaseStore pattern:
- `__init__(path: str)` — loads from JSONL; on first run (empty bank), loads seeds from `data/seeds/preprocessing_seeds.jsonl`
- `add(entry: PreprocessingCaseEntry)` — append-only JSONL write (OSError → log warning, skip)
- `get_all() -> List[PreprocessingCaseEntry]`
- `get_similar(task: TaskSpec, top_k: int = 3) -> List[PreprocessingCaseEntry]` — returns all entries in Step 2 (no semantic ranking yet)

**Seeds (`data/seeds/preprocessing_seeds.jsonl`):** 5 entries with `source: "external"`, each with a working `preprocess(df)` function. Patterns:
1. Name column with `"Lastname, Title. Firstname"` → extract title, map to numeric category
2. Cabin-style column (letter prefix + number) → extract deck letter as categorical
3. `SibSp` + `Parch` columns → derive `FamilySize = SibSp + Parch + 1`
4. Column with >70% missing → create binary `was_missing` indicator, impute median
5. Age column with 20% missing → impute with median (grouped by title if available, else global)

**Each seed's `preprocess(df)` must:**
- Check if relevant columns exist before operating (`if "Name" in df.columns:`)
- Return `df` at the end
- Import pandas at the top of the function body (not at module level)
- Handle edge cases gracefully

**Tests:**
```python
def test_add_and_get_all_round_trip()      # add entry, get_all returns it
def test_persists_to_jsonl()               # add entry, new store loads it
def test_seeds_loaded_on_first_run()       # empty bank → seeds present after init
def test_seeds_not_duplicated_on_reload()  # bank already has entries → seeds not re-added
def test_get_similar_returns_all()         # Step 2 behavior, returns all entries
def test_add_oserror_logs_warning()        # mock OSError on write, doesn't raise
```

---

## Step 6: EmbeddingRetriever + OpenAIBackend.embed()

**Files:**
- Extend: `src/llm/providers/openai.py` — add `embed(text: str) -> Optional[List[float]]`
- Create: `src/memory/embedding_retriever.py`
- Create: `tests/memory/test_embedding_retriever.py`
- Extend: `tests/llm/test_backend.py`

**`OpenAIBackend.embed()`:**
```python
def embed(self, text: str) -> Optional[List[float]]:
    """Embed text using text-embedding-3-small. Returns None on API error."""
    try:
        response = self._client.embeddings.create(
            input=text,
            model="text-embedding-3-small",
        )
        return response.data[0].embedding
    except Exception as exc:
        logger.warning("embed() failed: %s", exc)
        return None
```

**`EmbeddingRetriever`:**
```python
class EmbeddingRetriever:
    def __init__(self, backend: OpenAIBackend, ab_log: bool = True) -> None:
        # ab_log: if True, also compute CaseRetriever results for comparison logging

    def rank(self, query_text: str, cases: List[CaseEntry], top_k: int = 3) -> List[CaseEntry]:
        # embed query_text → cosine similarity against case.embedding
        # skip cases with embedding=None (fall back to CaseRetriever for those)
        # if ab_log: also call CaseRetriever.rank() and log both results side-by-side
        # returns top_k most similar cases

    def rank_preprocessing(
        self, query_text: str, cases: List[PreprocessingCaseEntry], top_k: int = 3
    ) -> List[PreprocessingCaseEntry]:
        # same logic for PreprocessingCaseEntry
```

**A/B logging:** `logger.debug("EmbeddingRetriever top-%d: %s | CaseRetriever top-%d: %s", top_k, [...], top_k, [...])`

**Tests (mock `OpenAIBackend.embed()`):**
```python
def test_rank_returns_most_similar()         # 3 cases, 1 most similar → it's first
def test_rank_skips_cases_without_embedding() # cases with embedding=None not ranked
def test_ab_log_written_when_enabled()       # verify logger.debug called with both results
def test_embed_api_failure_returns_none()    # OpenAI raises → returns None
def test_openai_backend_embed_mock()         # mock client, verify embed() returns list
```

---

## Step 7: Extend CaseEntry + Distiller

**Files:**
- Extend: `src/models/nodes.py` — add `description_for_embedding: str = ""`
- Extend: `src/memory/distiller.py` — generate embedding + write to PreprocessingStore
- Extend: `prompts/distiller.md` — add `description_for_embedding` field
- Extend: `tests/memory/test_distiller.py`

**`CaseEntry` change:**
```python
class CaseEntry(BaseModel):
    ...
    embedding: Optional[List[float]] = None          # already exists
    description_for_embedding: str = ""              # NEW: 1-2 sentence natural language summary
```

**`prompts/distiller.md` schema addition:**
```json
{
  ...,
  "description_for_embedding": "Binary classification on tabular data with 891 rows. GBM with medium_quality presets and 120s time limit achieved 0.85 roc_auc."
}
```

**`Distiller.distill()` signature extension:**
```python
def distill(
    self,
    task: TaskSpec,
    data_profile: DataProfile,
    run_history: List[ExperimentRun],
    embedding_retriever: Optional[OpenAIBackend] = None,  # if provided, embed + store
    preprocessing_plan: Optional[PreprocessingPlan] = None,
    preprocessing_store: Optional[PreprocessingStore] = None,
) -> CaseEntry:
    ...
    # After building case entry:
    # 1. case.description_for_embedding = parsed["description_for_embedding"]
    # 2. if embedding_retriever and case.description_for_embedding:
    #        case.embedding = embedding_retriever.embed(case.description_for_embedding)
    # 3. if preprocessing_plan and preprocessing_plan.strategy == "generated" and preprocessing_store:
    #        preprocessing_store.add(PreprocessingCaseEntry(...))
```

**Tests (extend `tests/memory/test_distiller.py`):**
```python
def test_distill_populates_description_for_embedding()
def test_distill_populates_embedding_when_retriever_provided()  # mock embed()
def test_distill_sets_embedding_none_when_retriever_absent()
def test_distill_writes_preprocessing_entry_when_generated()    # mock store.add
def test_distill_skips_preprocessing_entry_when_identity()
```

---

## Step 8: Wire into CampaignOrchestrator

**Files:**
- Extend: `src/orchestration/campaign.py`
- Extend: `campaign.py` (entrypoint)
- Extend: `configs/project.yaml` — add `preprocessing_bank_path`
- Extend: `docs/architecture/current-state.md`
- Append: `docs/changes/implementation-log.md`

**`CampaignOrchestrator.__init__` additions:**
```python
def __init__(
    self,
    ...,
    preprocessing_bank_path: Optional[str] = None,
    openai_backend: Optional[OpenAIBackend] = None,  # for embeddings; can be same as llm if OpenAI
) -> None:
    ...
    self._preprocessing_store = PreprocessingStore(preprocessing_bank_path) if preprocessing_bank_path else None
    self._preprocessing_agent = PreprocessingAgent(llm=llm) if llm else None
    self._embedding_retriever = EmbeddingRetriever(backend=openai_backend) if openai_backend else None
```

**Replace `_preprocessing_plan()` stub:**
```python
def _preprocessing_plan(self, data_profile: DataProfile) -> PreprocessingPlan:
    if self._preprocessing_agent is None or self._preprocessing_store is None:
        return PreprocessingPlan(strategy="identity")
    similar = self._preprocessing_store.get_similar(self._task, top_k=3)
    return self._preprocessing_agent.generate(
        task=self._task,
        data_profile=data_profile,
        data_path=self._task.data_path,
        similar_cases=similar,
    )
```

**Per-session logging (in `run()` loop):**
```python
self._log.info(
    f"Session {i+1} preprocessing: strategy={plan.strategy} "
    f"validation_passed={plan.validation_passed} turns_used={plan.turns_used}"
)
```

**Consecutive failure tracking:**
```python
consecutive_preprocessing_failures = 0
# after getting plan:
if not plan.validation_passed and plan.strategy == "identity":
    consecutive_preprocessing_failures += 1
else:
    consecutive_preprocessing_failures = 0
if consecutive_preprocessing_failures >= 2:
    self._log.warning("PreprocessingAgent has failed %d consecutive sessions — check logs", consecutive_preprocessing_failures)
```

**Pass embedding_retriever and preprocessing info to Distiller at session end.**

**`configs/project.yaml` addition:**
```yaml
campaign:
  max_sessions: 2
  plateau_threshold: 0.002
  plateau_window: 3
  preprocessing_bank_path: "experiments/preprocessing_bank.jsonl"
```

**End-to-end smoke test:** `python3 campaign.py` should:
- Call PreprocessingAgent before session 1
- Log preprocessing outcome
- Write preprocessing_bank.jsonl after session 1 if strategy=generated
- campaign.json contains `preprocessing_strategy` populated from agent

---

## Completion Criteria (from design doc)

- [ ] PreprocessingAgent generates valid `preprocess(df)` on Titanic that passes ValidationHarness AND adds at least one new feature (extracted title from Name)
- [ ] `python3 campaign.py` runs full campaign with preprocessing varying per session
- [ ] EmbeddingRetriever retrieves cases with A/B log visible in session output
- [ ] `experiments/preprocessing_bank.jsonl` populated after a full campaign run
- [ ] External seeds present in bank from first run
- [ ] All new tests pass: `pytest tests/ -x`
