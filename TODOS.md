# TODOS

Deferred work captured during plan reviews.

---

## TODO: SessionConfig dataclass

Bundle the 6 `ExperimentSession` pass-through params (`num_candidates`, `max_optimize_iterations`, `higher_is_better`, `case_store_path`, `preprocessed_data_path`) into a `SessionConfig` dataclass so `CampaignOrchestrator.__init__` has a single config object instead of 6 individual params.

**Why:** Currently `CampaignOrchestrator` has 6 params that are purely passed to `ExperimentSession`. Any new `ExperimentSession` param requires touching both classes.

**Pros:** Single point of change when `ExperimentSession` grows; cleaner orchestrator init signature.

**Cons:** Adds one more dataclass; `ExperimentSession` init would need to accept `Optional[SessionConfig]` OR keep the current flat API alongside it.

**Where to start:**
- `src/models/campaign.py` — add `SessionConfig`
- `src/session.py` — accept `Optional[SessionConfig]`
- `src/orchestration/campaign.py` — pass single object

**Effort:** S (human: ~2h / CC: ~5min) | **Priority:** P3 | **Phase:** 4b

---

## TODO: Cross-session IdeatorAgent history

Pass a summary of prior sessions' best metrics and strategies to `IdeatorAgent` so it can avoid repeating the same hypotheses when the campaign has run multiple sessions.

**Why:** In Phase 4a all sessions use identity preprocessing (same data), so history adds no signal. In Phase 4b, different preprocessing strategies will produce different metrics — history becomes genuinely informative.

**Blocked by:** Phase 4b PreprocessingAgent (strategies must vary before history adds value). **Unblocked when Phase 4b ships.**

**Effort:** S | **Priority:** P2 | **Phase:** 4b (post-ship)

---

## TODO: RefinerAgent CaseStore access

`RefinerAgent` currently receives no `similar_cases`. It could use the `CaseStore` to look up what refinements worked on similar past tasks and avoid repeated dead-ends.

**Why:** Right now refinement is memoryless across sessions. Plugging in the case store gives the refiner the same cross-task memory the ideator already has.

**Blocked by:** Nothing — can be done independently.

**Effort:** S | **Priority:** P2 | **Phase:** 4b

---

## TODO: Extract BaseJSONLStore[T] base class

When Phase 5 adds a store for Graph RAG node data, extract a generic `BaseJSONLStore[T]` base class to remove duplication across `RunStore`, `CaseStore`, and `PreprocessingStore`.

**Why:** Phase 4b ships `PreprocessingStore` as a standalone 25-line mirror of `CaseStore`. When a 3rd store arrives, the pattern is clearly worth abstracting.

**Pros:** Single implementation of JSONL load/append/cache logic; easier to add features (e.g. indexing, filtering) once.

**Cons:** Generic class adds minor complexity; Pydantic models need a `model_validate_json` / `model_dump_json` interface assumption.

**Where to start:**
- `src/memory/base_store.py` — add `BaseJSONLStore[T](Generic[T])` with `__init__`, `add()`, `get_all()`
- `src/memory/case_store.py`, `preprocessing_store.py` — subclass and remove boilerplate

**Effort:** S (human: ~1h / CC: ~10min) | **Priority:** P3 | **Phase:** 5
