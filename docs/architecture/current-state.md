# Architecture: Current State

**Last updated:** 2026-03-31
**Phase:** 5 (Feature Engineering Subsystem)

## Architecture Overview

The system has two campaign paths:

1. **Preprocessing Campaign** (Phase 4b) — `CampaignOrchestrator` with `PreprocessingAgent`
2. **Feature Engineering Campaign** (Phase 5) — `FeatureCampaignOrchestrator` with `FeatureEngineeringAgent`

Both share the core session/training spine (`ExperimentSession`, `AutoGluonRunner`, etc.) but are independent sibling orchestrators.

## Six-Layer Architecture

```
+-----------------------------------------------------------+
|  AGENT LAYER                                              |
|  IdeatorAgent, SelectorAgent, RefinerAgent, ExperimentManager |
|  PreprocessingAgent: ReAct loop (inspect_column/generate_code) |
|  FeatureEngineeringAgent: decision → leakage audit → execute |
|    Internal pipeline: _decision_call → _leakage_audit_call |
|    → _execute_bounded (Phase 1) or _codegen (Phase 2)     |
+-----------------------------------------------------------+
|  ORCHESTRATION LAYER                                      |
|  CampaignOrchestrator: preprocessing campaign (Phase 4b)  |
|  FeatureCampaignOrchestrator: feature engineering campaign |
|  ExperimentTree, Scheduler, AcceptReject                  |
+-----------------------------------------------------------+
|  FEATURE ENGINEERING LAYER (Phase 5)                      |
|  TemplateRegistry: named feature template functions       |
|  BoundedExecutor: dispatches DSL configs to templates     |
|  FeatureValidator: row count, target, null checks         |
|  DSL: 14-operator surface with time-op leakage guards     |
|  Templates: customer, order, temporal, transforms, composites |
+-----------------------------------------------------------+
|  EXECUTION LAYER                                          |
|  AutoGluonRunner, ConfigMapper, ResultParser              |
|  PreprocessingExecutor: identity or exec(plan.code)       |
|  ValidationHarness: subprocess-isolated 6-check validator |
+-----------------------------------------------------------+
|  MEMORY LAYER                                             |
|  RunStore: append-only session journal (decisions.jsonl)  |
|  FeatureHistoryStore: empirical experiment memory (JSONL) |
|  ContextBuilder: assembles SearchContext briefing         |
|  FeatureContextBuilder: assembles feature eng context     |
|  [RETIRED] CaseStore, PreprocessingStore, EmbeddingRetriever |
+-----------------------------------------------------------+
|  DATA LAYER                                               |
|  experiments/feature_history.jsonl — FeatureHistoryStore   |
|  references/feature_engineering/  — static reference packs |
|  prompts/feature_engineering/     — agent prompt assets    |
+-----------------------------------------------------------+
```

## Feature Engineering Campaign Flow (Phase 5)

```
Baseline session:
  identity preprocessing → ExperimentSession → collect DataProfile + leaderboard

Feature iteration loop:
  FeatureContextBuilder.build(profile, leaderboard, importances, history)
  → FeatureEngineeringAgent.propose_and_execute()
    ├─ _decision_call → FeatureDecision (JSON)
    ├─ _leakage_audit_call → FeatureAuditVerdict (mandatory, no bypass)
    └─ _execute_bounded → FeatureExecutionResult (via BoundedExecutor)
  Save featured CSV → new ExperimentSession (retrain)
  Store FeatureHistoryEntry (empirical memory)
  Check stop: plateau / budget / consecutive blocks
```

## Knowledge Architecture

The feature engineering system uses two knowledge sources:

### Empirical Experiment Memory
- `FeatureHistoryStore` — what we tried, what happened, distilled takeaways
- Append-only JSONL, scoped to campaigns
- Available to the agent as context in each iteration

### Static Reference Packs
- `references/feature_engineering/ecommerce_features.md` — curated feature knowledge
- `references/feature_engineering/leakage_patterns.md` — known leakage patterns
- Loaded into prompts as static context, not retrieved semantically

### Retired: Vector RAG
- `CaseStore`, `PreprocessingStore`, `EmbeddingRetriever` — retained for backward compatibility with `CampaignOrchestrator` only
- Not used by `FeatureCampaignOrchestrator`

## Preprocessing Campaign Flow (Phase 4b, unchanged)

```
Session 0 (baseline):
  identity preprocessing → ExperimentSession → Distiller → CaseStore

Session N (N ≥ 1):
  PreprocessingAgent.generate(similar_cases from PreprocessingStore)
    ├─ inspect_column (up to 2 calls, max 3 turns)
    └─ generate_code → ValidationHarness (subprocess, 6 checks)
  PreprocessingExecutor (identity fallback on exec error)
  ExperimentSession → Distiller → CaseStore
```

## Session Flow (shared by both campaign types)

```
1. Profile data → DataProfile
2. IdeatorAgent generates hypotheses
3. Warm-up: run each hypothesis as root ExperimentNode
4. Optimize: RefinerAgent → targeted ExperimentPlan
5. AcceptReject gates each optimization step
6. Save tree.json
```

## Key Data Objects

| Object | Owner | Purpose |
|--------|-------|---------|
| TaskSpec | models/task.py | Problem definition |
| ExperimentPlan | models/task.py | Agent's proposed config (JSON) |
| RunConfig | models/task.py | Concrete AutoGluon kwargs |
| ExperimentRun | models/results.py | Full experiment record |
| FeatureDecision | models/feature_engineering.py | Agent's proposed feature action |
| FeatureSpec | models/feature_engineering.py | Discriminated union: template/transform/composite/codegen |
| FeatureAuditVerdict | models/feature_engineering.py | Leakage audit result |
| FeatureExecutionResult | models/feature_engineering.py | Execution outcome |
| FeatureHistoryEntry | models/feature_engineering.py | Empirical experiment memory entry |
| FeatureCampaignConfig | models/campaign.py | Feature campaign loop config |
| CampaignResult | models/campaign.py | Full record of a multi-session campaign |

## What is NOT yet built

- Phase 2 codegen escape hatch (CodegenSandbox, codegen guardrail)
- ReviewerAgent — post-run quality assessment
- Graph RAG over ExperimentNode trees
- Dataset benchmark validation (UCI Online Retail, Olist, RetailRocket)
