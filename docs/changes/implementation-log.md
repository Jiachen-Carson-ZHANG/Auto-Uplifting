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
