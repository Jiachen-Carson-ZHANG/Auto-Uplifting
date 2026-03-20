# Architecture: Current State

**Last updated:** 2026-03-20
**Phase:** 4a (Campaign Orchestration complete)

## Four-Layer Architecture

```
+-----------------------------------------------------------+
|  AGENT LAYER                                              |
|  IdeatorAgent, SelectorAgent, RefinerAgent, ExperimentManager |
|  IdeatorAgent: data profile + similar cases → hypotheses  |
|  SelectorAgent: hypothesis → ExperimentPlan (JSON)        |
|  ExperimentManager: warmup/optimize/stop routing          |
+-----------------------------------------------------------+
|  ORCHESTRATION LAYER                                      |
|  CampaignOrchestrator, ExperimentTree, Scheduler, AcceptReject |
|  Tree: graph-structured lineage with edge labels          |
|  Scheduler: warmup → optimize transitions, budget         |
|  AcceptReject: direction-aware incumbent gating           |
+-----------------------------------------------------------+
|  EXECUTION LAYER                                          |
|  AutoGluonRunner, ConfigMapper, ResultParser              |
|  Runner: captures training logs to {run_dir}/training.log |
|  ResultParser: leaderboard + diagnostics → RunResult      |
+-----------------------------------------------------------+
|  MEMORY LAYER                                             |
|  RunStore: append-only session journal (decisions.jsonl)  |
|  CaseStore: cross-session JSONL knowledge base            |
|  CaseRetriever: cosine similarity on TaskTraits vectors   |
|  Distiller: LLM session → CaseEntry at session end        |
|  ContextBuilder: assembles SearchContext briefing         |
+-----------------------------------------------------------+
```

## Session Flow

```
1. Profile data → DataProfile
2. Retrieve similar past cases from CaseStore (cosine similarity on TaskTraits)
3. IdeatorAgent generates hypotheses grounded in data profile + similar cases
4. Warm-up: run each hypothesis as root ExperimentNode
5. Optimize: RefinerAgent reads incumbent config + leaderboard + overfitting_gap → targeted ExperimentPlan
6. AcceptReject gates each optimization step
7. Distiller summarises session → CaseEntry → CaseStore
8. Save tree.json (Graph RAG compatible)
```

## Key Data Objects

| Object | Owner | Purpose |
|--------|-------|---------|
| TaskSpec | models/task.py | Problem definition |
| ExperimentPlan | models/task.py | Agent's proposed config (JSON) |
| RunConfig | models/task.py | Concrete AutoGluon kwargs |
| RunEntry | models/results.py | Full experiment record (config + result + diagnostics) |
| ExperimentNode | models/nodes.py | Tree node with edge_label for Graph RAG |
| CaseEntry | models/nodes.py | Distilled session knowledge for cross-session retrieval |
| SearchContext | models/nodes.py | Briefing assembled for agent before each decision |
| CampaignResult | models/campaign.py | Full record of a multi-session campaign |
| SessionSummary | models/campaign.py | Per-session result within a campaign |
| PreprocessingPlan | models/preprocessing.py | Stub for Phase 4b preprocessing strategy |

## Session Outputs

```
experiments/{date}_{HH-MM-SS}_{task}/
  session.log       — timestamped progress log
  decisions.jsonl   — RunEntry journal (machine-readable)
  tree.json         — ExperimentNode graph (Graph RAG compatible)
  runs/run_XXXX/
    training.log    — AutoGluon verbose output
experiments/case_bank.jsonl  — cross-session CaseStore
```

## What is NOT yet built (Phase 4+)

- ReviewerAgent — post-run quality assessment (Phase 4)
- Optuna executor (Phase 4)
- **Phase 4b:** PreprocessingAgent (ReAct code gen), ValidationHarness, preprocessing_bank, EmbeddingRetriever, external knowledge seeds
- Graph RAG over ExperimentNode trees (Phase 5)
