# Holdout Leakage Remediation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Prevent adaptive trial generation, judging, tuning, and live champion selection from using held-out/test feedback before final audit.

**Architecture:** Validation is the only metric surface inside adaptive loops. Held-out/test scoring becomes an explicit post-selection audit mode. Existing leaked artifacts remain available for transparency, but are labeled as quarantined and cannot be used as final champion evidence.

**Tech Stack:** Python, pandas, pytest, AutoLift uplift runner, markdown result artifacts.

---

### Task 1: Add Guardrail Tests

**Files:**
- Modify: `tests/uplift/test_ledger_loop.py`
- Modify: `tests/uplift/test_pr2_guardrails.py`
- Modify: `tests/uplift/test_pr2_phase_flow.py`
- Modify: `tests/uplift/test_agentic_tuning_execution.py`
- Modify: `tests/uplift/test_pr1_orchestration_names.py`
- Modify: `tests/integration/test_autonomous_pipeline_demo.py`

**Steps:**
1. Assert `run_uplift_trials` hides held-out metrics by default even when a test partition exists.
2. Assert held-out scoring appears only when `score_held_out=True`.
3. Assert `run_evaluation_phase` ignores `held_out_scores_df` unless `allow_held_out_metrics=True`.
4. Assert planning prompts do not serialize `held_out_*` metrics or artifact paths.
5. Assert tuning execution summaries do not include held-out metrics.
6. Assert live champion selection uses validation-only evidence.

### Task 2: Make Held-Out Scoring Opt-In

**Files:**
- Modify: `src/uplift/loop.py`
- Modify: `src/uplift/agentic_tuning_execution.py`

**Steps:**
1. Add `score_held_out: bool = False` to `run_uplift_trials`.
2. Pass the test partition to `run_uplift_template` only when `score_held_out=True`.
3. Leave tuning execution on the default validation-only path.
4. Remove held-out metric serialization from tuning summaries and champion summaries.

### Task 3: Seal Planning and Judge Feedback

**Files:**
- Modify: `src/uplift/planning_agents.py`
- Modify: `src/uplift/evaluation_agents.py`

**Steps:**
1. Remove held-out fields from planning record summaries.
2. Remove prompt wording that asks the LLM to prefer held-out stability.
3. Add `allow_held_out_metrics: bool = False` to judge/evaluation APIs.
4. Select prior champions for judge comparison using validation metrics only.

### Task 4: Fix Live Champion and Report Semantics

**Files:**
- Modify: `src/uplift/orchestrator.py`
- Modify: `demos/uplift_run_autonomous_pipeline.py`
- Modify: `results/run_20260430_best/README.md`
- Modify: `results/run_20260430_best/final_report.md`
- Modify: `results/run_20260430_best/explainability/EXPLAINABILITY_REPORT.md`

**Steps:**
1. Select live agent champions by validation-only score.
2. Keep held-out language only for explicit post-selection audit sections.
3. Label earlier adaptive held-out artifacts as quarantined exploratory evidence.
4. Keep `RUN-f1c30175` as the reportable validation+CV-selected candidate.

### Task 5: Verify and Push

**Commands:**
- `python -m py_compile ...`
- `.venv/bin/python -m pytest <focused guardrail tests> -q`
- `.venv/bin/python -m pytest tests/uplift/test_ledger_loop.py tests/uplift/test_pr2_guardrails.py tests/uplift/test_pr2_phase_flow.py tests/uplift/test_pre_run_tuning.py tests/uplift/test_agentic_tuning_execution.py tests/uplift/test_pr1_orchestration_names.py tests/integration/test_autonomous_pipeline_demo.py -q`
- `git diff --check`
- `git commit -m "Seal holdout metrics from adaptive uplift loop"`
- `git push origin main`
