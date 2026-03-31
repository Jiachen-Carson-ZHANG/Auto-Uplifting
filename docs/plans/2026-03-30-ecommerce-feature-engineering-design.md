# Ecommerce Feature Engineering Design

**Date:** 2026-03-30
**Status:** Approved for audit before implementation
**Scope:** New ecommerce-focused feature-engineering subsystem added alongside the existing preprocessing campaign

## Goal

Add a new feature-engineering path for ecommerce lifecycle tasks that sits on top of the current session/training spine without refactoring the existing preprocessing campaign first. The new runtime path should:

- stay local-first and easy to debug
- keep one top-level `FeatureEngineeringAgent` node in the campaign
- use bounded `template + DSL` execution by default
- support richer code generation later through an explicit escape hatch
- store reusable feature history and distilled facts for future sessions
- remove vector-retrieval/RAG behavior from the new feature system

This design intentionally targets ecommerce lifecycle problems first:

- churn
- repurchase
- lifetime value

Recommendation and event-sequence problems are treated as stress cases, not the first success criterion.

## Why A Separate Subsystem

The current repo already has a coherent preprocessing flow:

- `PreprocessingAgent`
- `ValidationHarness`
- `PreprocessingExecutor`
- `PreprocessingStore`
- `CampaignOrchestrator`

That spine is useful, but feature engineering is a different concern from preprocessing. Preprocessing is about making the data usable and safe. Feature engineering is about proposing, validating, executing, and learning from candidate predictive signals. Overloading preprocessing to carry both responsibilities would blur boundaries and make the system harder to reason about.

The new feature-engineering path should therefore be implemented as a sibling subsystem, not as a rewrite of preprocessing.

## Architectural Decision

Build one runtime `FeatureEngineeringAgent` node with a small internal pipeline rather than several peer agents in the outer orchestration loop.

Externally, the campaign sees one decision point:

1. baseline session runs on raw or identity-preprocessed data
2. feature-engineering node proposes one action
3. feature path executes
4. model retrains
5. result and feature facts are persisted

Internally, the feature node may make several LLM calls, but they remain an implementation detail of that single node.

This keeps the campaign understandable while still allowing structured reasoning inside the feature loop.

## Internal Feature Node Pipeline

The internal flow of `FeatureEngineeringAgent` is:

1. `decision_call`
2. `leakage_audit_call`
3. branch into:
   - bounded `template + DSL` execution
   - or `escalate_codegen`
4. if codegen path:
   - `codegen_call`
   - `codegen_guardrail_call`
   - sandbox execution and validation
5. return final feature result plus facts to save

Two rules are important:

- `leakage_audit_call` runs on both bounded and codegen paths
- `codegen_guardrail_call` runs only on the codegen path

This gives the node a disciplined structure without turning the whole runtime into a multi-agent swarm.

## Runtime Boundaries

### In Runtime

These belong inside the product runtime:

- feature decision
- leakage audit
- bounded template execution
- guarded code generation
- feature validation
- feature history persistence
- prompt assets and context loading

### Outside Runtime

These are developer workflows, not part of the hot path:

- template extension
- reference curation and research maintenance
- leakage-rule authoring workflow
- codegen guardrail authoring workflow

This split avoids mixing product behavior with developer ergonomics.

## Experiment Memory vs Reference Knowledge

The new feature system should use two different knowledge sources:

### Experiment Memory

This is empirical knowledge learned from our own runs.

Examples:

- prior feature attempts
- metric deltas
- blocked leakage cases
- failed codegen attempts
- distilled takeaways from previous campaigns

This memory is part of the runtime. It is the right place for what we earlier called "refiner memory" or "historical experiment memory." It should be available to the feature-engineering loop and can later be generalized for other agents such as the current model refiner.

### Reference Knowledge

This is curated external knowledge.

Examples:

- ecommerce feature families
- leakage patterns
- domain playbooks
- task-specific heuristics

Reference knowledge is not memory. It is loaded into prompts as static context. It should live in the repo as versioned runtime assets under `references/`, not inside developer skill folders.

### No Vector RAG In The New Path

The new feature-engineering subsystem should not use vector retrieval or embedding-based ranking. The old `CaseStore`, `PreprocessingStore`, and `EmbeddingRetriever` are not part of the target architecture for this subsystem.

This does not mean "no memory." It means:

- no semantic retrieval layer
- yes to experiment logs and distilled empirical facts
- yes to static reference packs

That separation is the intended simplification.

## Two-Phase Strategy

### Phase 1: Bounded Default Path

Phase 1 uses approved templates plus a config-shaped DSL. It is the normal path.

Supported action families:

- `add`
- `drop`
- `transform`
- `composite`

Initial ecommerce feature families:

- recency
- frequency
- monetary or spend
- average order value or basket value
- category or product diversity
- discount or payment ratio
- temporal seasonality
- safe composite ratios

Phase 1 is intentionally thin. It is not meant to encode every ecommerce feature up front. It is meant to provide a safe, reusable baseline that covers common lifecycle tasks on transactional ecommerce data.

### Phase 2: Codegen Escape Hatch

Phase 2 is not the default. It is an explicit escalation path reached only when the agent decides the bounded surface cannot express the needed feature.

The output must say why the registry and DSL are insufficient. Example pressure cases:

- sequence-aware event logic
- unusual multi-table joins
- company-specific derived signals
- nested session or cart logic

This is an escape hatch, not broad free code generation. The system should prefer reusable abstractions first, then escalate when the dataset truly requires richer logic.

## Why Not Broad Free Code Generation By Default

Broad code generation is appealing because it looks faster early on, but it moves complexity into validation and debugging:

- execution safety is easy to check
- ecommerce leakage correctness is not
- one-off feature code is harder to compare and reuse
- subtle future-looking joins or target leakage are easy to miss

Bounded `template + DSL` gives a controlled baseline. Phase 2 codegen stays available for schemas the bounded path cannot handle well.

## Decision Contracts

The feature node should use strict structured contracts rather than free-form text.

### `FeatureDecision`

Core fields:

- `status`
- `action`
- `reasoning`
- `feature_spec`
- `expected_impact`
- `risk_flags`
- `observations`
- `facts_to_save`

Allowed `action` values:

- `add`
- `drop`
- `transform`
- `composite`
- `request_context`
- `blocked`
- `escalate_codegen`

### `FeatureSpec`

The spec should be discriminated by execution path:

- `TemplateFeatureSpec`
- `TransformFeatureSpec`
- `CompositeFeatureSpec`
- `CodegenEscalationSpec`

### `FeatureAuditVerdict`

Used by leakage audit and codegen guardrail:

- `verdict`
- `reasons`
- `required_fixes`

### `FeatureExecutionResult`

Used to capture execution outcome:

- output path or produced columns
- validation outcome
- warnings
- failure reason if any

### `FeatureHistoryEntry`

Append-only telemetry for later retrieval:

- feature action
- dataset context
- task type
- metric before and after
- observed outcome
- distilled takeaway

## DSL Shape

The DSL should be config-shaped, not string-formula-based.

Example:

```json
{
  "action": "composite",
  "feature": {
    "name": "cart_to_purchase_rate_L30D",
    "op": "safe_divide",
    "inputs": [
      {"ref": "purchase_count_L30D"},
      {"ref": "cart_add_count_L30D"}
    ],
    "post": ["clip_0_1"]
  }
}
```

Recommended initial operator surface:

- `safe_divide`
- `subtract`
- `add`
- `multiply`
- `ratio_to_baseline`
- `log1p`
- `clip`
- `bucketize`
- `is_missing`
- `days_since`
- `count_in_window`
- `sum_in_window`
- `mean_in_window`
- `nunique_in_window`

For time-based features, the system must require:

- `entity_key`
- `time_col`
- explicit window definition
- cutoff mode or prediction-time semantics

That requirement is one of the main leakage defenses.

## Leakage And Guardrails

Leakage audit is part of the feature node, not an optional afterthought.

The audit should check at least:

- target column usage
- future-looking timestamps beyond cutoff
- post-outcome joins
- unbounded aggregations
- missing entity or time semantics for windowed features

Phase 2 codegen adds stricter checks before execution:

- target reference scan
- future-time pattern scan
- row explosion or join blow-up checks
- all-null or constant feature checks
- replayability and artifact logging

The existing `ValidationHarness` is still useful, but mainly as a Phase 2 building block. It should not become the center of Phase 1 bounded validation.

## Campaign Integration

This should be a new sibling campaign or orchestrator path rather than a rewrite of the preprocessing campaign.

High-level flow:

1. baseline session on raw or identity-preprocessed data
2. collect profile, leaderboard, importances, overfitting gap, and recent history
3. call `FeatureEngineeringAgent`
4. validate feature proposal
5. execute bounded path or codegen path
6. retrain
7. store feature result, telemetry, and distilled facts
8. repeat until stop condition

Stop conditions should mirror the current campaign style:

- budget exhausted
- plateau in metric improvement
- repeated blocked outcomes
- repeated escalation failures

For v1, preprocessing remains identity inside the feature-engineering campaign. The new system should not combine preprocessing generation and feature generation in the same outer loop yet.

## Proposed Runtime Modules

New modules expected from this design:

- `src/models/feature_engineering.py`
- `src/agents/feature_engineer.py`
- `src/features/context_builder.py`
- `src/features/history.py`
- `src/features/validator.py`
- `src/features/executor.py`
- `src/features/registry.py`
- `src/features/dsl.py`
- `src/features/sandbox.py`
- `src/features/templates/customer.py`
- `src/features/templates/order.py`
- `src/features/templates/temporal.py`
- `src/features/templates/transforms.py`
- `src/features/templates/composites.py`
- `src/memory/feature_store.py`
- `src/orchestration/feature_campaign.py`

Prompt assets:

- `prompts/feature_engineering/feature_engineer_router.md`
- `prompts/feature_engineering/feature_engineer_full.md`
- `prompts/feature_engineering/feature_leakage_audit.md`
- `prompts/feature_engineering/feature_codegen.md`
- `prompts/feature_engineering/feature_codegen_guardrail.md`

Reference assets:

- `references/feature_engineering/ecommerce_features.md`
- `references/feature_engineering/leakage_patterns.md`

## Dataset Feasibility Benchmark

Before implementation is considered complete, the system should be validated conceptually and then in tests against three ecommerce dataset shapes:

- UCI Online Retail for transactional coverage
- Olist for richer multi-table ecommerce coverage
- RetailRocket as the early pressure case for codegen

Expected outcome:

- UCI and Olist should fit the bounded path reasonably well
- RetailRocket-like event streams are the likely earliest trigger for `escalate_codegen`

This benchmark is not a production eval. It is a coverage test for the architecture.

## Error Handling

The system should fail closed where leakage or schema safety is involved and fail soft where LLM output quality is involved.

Examples:

- malformed feature decision JSON should retry and then return a blocked result
- leakage audit `block` should prevent execution
- bounded executor failures should record a structured failure, not crash the campaign
- codegen guardrail failures should stop code execution and persist the reason
- store write failures should log clearly and preserve campaign execution where safe

Important branches should emit enough logs and artifacts to reconstruct what happened after the fact.

## Testing Strategy

The first implementation should be judged by contracts and reproducibility, not just by whether it runs once.

Test layers:

- unit tests for models, registry lookup, DSL validation, leakage checks, executor behavior, and feature history
- prompt or contract tests for internal LLM subcalls and invalid JSON recovery
- integration tests for:
  - baseline plus bounded feature loop
  - forced `escalate_codegen` path
  - identity preprocessing inside feature campaigns

Acceptance criteria:

- bounded path can express useful features on UCI and Olist without codegen
- bounded features are deterministic and logged cleanly
- `escalate_codegen` is exercised and cannot bypass guardrails
- campaign outputs and feature history are replayable

## Non-Goals For First Pass

The first implementation should explicitly not attempt:

- a runtime template-extension loop
- automatic reference research on every iteration
- broad free code generation as default behavior
- merging preprocessing generation and feature generation into one node
- recommendation-first architecture
- a large template universe before coverage is proven

## Implementation Notes For Later

This document captures the approved design, not the task-by-task execution sequence. The implementation plan should come next and should be written separately before code changes begin.

The intended execution posture is:

- design saved and audited first
- implementation plan written second
- runtime code implemented in an isolated worktree
- docs updated again after architecture changes land
