# Architecture: Current State

**Last updated:** 2026-03-21
**Phase:** 4b (PreprocessingAgent + EmbeddingRetriever complete)

## Five-Layer Architecture

```
+-----------------------------------------------------------+
|  AGENT LAYER                                              |
|  IdeatorAgent, SelectorAgent, RefinerAgent, ExperimentManager |
|  PreprocessingAgent: ReAct loop (inspect_column/generate_code) |
|  IdeatorAgent: data profile + similar cases → hypotheses  |
|  SelectorAgent: hypothesis → ExperimentPlan (JSON)        |
|  ExperimentManager: warmup/optimize/stop routing          |
+-----------------------------------------------------------+
|  ORCHESTRATION LAYER                                      |
|  CampaignOrchestrator, ExperimentTree, Scheduler, AcceptReject |
|  Campaign: session 0=identity baseline, 1+= PreprocessingAgent |
|  Tree: graph-structured lineage with edge labels          |
|  Scheduler: warmup → optimize transitions, budget         |
|  AcceptReject: direction-aware incumbent gating           |
+-----------------------------------------------------------+
|  EXECUTION LAYER                                          |
|  AutoGluonRunner, ConfigMapper, ResultParser              |
|  PreprocessingExecutor: identity or exec(plan.code)       |
|  ValidationHarness: subprocess-isolated 6-check validator |
+-----------------------------------------------------------+
|  MEMORY LAYER                                             |
|  RunStore: append-only session journal (decisions.jsonl)  |
|  CaseStore: cross-session JSONL knowledge base (CaseEntry)|
|  PreprocessingStore: cross-session preprocessing bank     |
|  CaseRetriever: cosine similarity on TaskTraits vectors   |
|  EmbeddingRetriever: cosine similarity on text embeddings |
|  Distiller: LLM session → CaseEntry; embed() if available |
|  ContextBuilder: assembles SearchContext briefing         |
+-----------------------------------------------------------+
|  DATA LAYER                                               |
|  experiments/case_bank.jsonl          — CaseStore         |
|  experiments/preprocessing_bank.jsonl — PreprocessingStore|
|  data/seeds/preprocessing_seeds.jsonl — 5 bootstrap seeds |
+-----------------------------------------------------------+
```

## Campaign Flow (Phase 4b)

```
Session 0 (baseline):
  identity preprocessing → ExperimentSession → Distiller → CaseStore

Session N (N ≥ 1):
  PreprocessingAgent.generate(similar_cases from PreprocessingStore)
    ├─ inspect_column (up to 2 calls, max 3 turns)
    └─ generate_code → ValidationHarness (subprocess, 6 checks)
  PreprocessingExecutor (identity fallback on exec error)
  ExperimentSession → Distiller → CaseStore
  [if validation_passed] PreprocessingEntry → PreprocessingStore
```

## Session Flow (within each campaign session)

```
1. Profile data → DataProfile
2. Retrieve similar past cases from CaseStore (cosine similarity on TaskTraits)
3. IdeatorAgent generates hypotheses grounded in data profile + similar cases
4. Warm-up: run each hypothesis as root ExperimentNode
5. Optimize: RefinerAgent reads incumbent config + leaderboard + overfitting_gap → targeted ExperimentPlan
6. AcceptReject gates each optimization step
7. Distiller summarises session → CaseEntry → CaseStore (+ embedding if OpenAIBackend provided)
8. Save tree.json (Graph RAG compatible)
```

## Key Data Objects

| Object | Owner | Purpose |
|--------|-------|---------|
| TaskSpec | models/task.py | Problem definition |
| ExperimentPlan | models/task.py | Agent's proposed config (JSON) |
| RunConfig | models/task.py | Concrete AutoGluon kwargs |
| ExperimentRun | models/results.py | Full experiment record (config + result + diagnostics) |
| ExperimentNode | models/nodes.py | Tree node with edge_label for Graph RAG |
| CaseEntry | models/nodes.py | Distilled session knowledge; has `description_for_embedding` + `embedding` |
| SearchContext | models/nodes.py | Briefing assembled for agent before each decision |
| CampaignResult | models/campaign.py | Full record of a multi-session campaign |
| SessionSummary | models/campaign.py | Per-session result; tracks `preprocessing_validation_passed`, `preprocessing_turns_used` |
| PreprocessingPlan | models/preprocessing.py | Agent output: identity or generated code with validation metadata |
| PreprocessingEntry | models/preprocessing.py | Cross-session preprocessing knowledge; has `transformation_summary` + `embedding` |

## Session Outputs

```
experiments/{date}_{HH-MM-SS}_{task}/
  session.log       — timestamped progress log
  decisions.jsonl   — ExperimentRun journal (machine-readable)
  tree.json         — ExperimentNode graph (Graph RAG compatible)
  runs/run_XXXX/
    training.log    — AutoGluon verbose output
experiments/case_bank.jsonl            — cross-session CaseStore
experiments/preprocessing_bank.jsonl   — cross-session PreprocessingStore
experiments/campaigns/{id}_{task}/
  campaign.json     — CampaignResult (written after each session)
  campaign.log      — timestamped campaign log
  preprocessing_N/  — preprocessed_data.csv for session N
```

## What is NOT yet built (Phase 5+)

- ReviewerAgent — post-run quality assessment
- Optuna executor
- Graph RAG over ExperimentNode trees
- BaseJSONLStore[T] base class (when a 3rd store is needed — currently CaseStore, PreprocessingStore are standalone)
