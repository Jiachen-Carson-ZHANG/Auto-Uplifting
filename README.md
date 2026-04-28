# BT5153 Agentic Uplift Modeling

This repository contains the BT5153 project code for the X5 RetailHero uplift-modeling task.

The current codebase is intentionally narrow. It focuses on a reliable uplift-modeling kernel first, then adds an agentic supervisor only after the measurement and experiment loop are trustworthy.

## Current Scope

The project answers one business question:

> Which customers should receive a campaign treatment, and how robust is that targeting decision?

Implemented so far:

- Contract-driven RetailHero table and column semantics.
- Dataset validation for labeled training data and unlabeled scoring data.
- Treatment/control balance diagnostics.
- Cached customer-level feature construction from clients and purchases.
- Uplift metrics: Qini AUC, uplift AUC, uplift@k, decile tables, and policy gain.
- Deterministic baseline templates: random, response model, two-model, and solo-model.
- JSONL experiment ledger with prediction, metric, curve, report, and submission artifacts.
- Guarded advisory planning/reporting path that cannot mutate target, treatment, split, metric, or submission semantics.
- Pointer-only hypothesis records for the next supervisor layer.

Not included in this cleaned BT5153 repo:

- Previous generic Agentic ML framework experiments.
- Internal planning notes, implementation logs, Graphify/GSD artifacts, and agent scratch files.
- Raw RetailHero data or generated experiment artifacts.
- Free-form code generation or broad AutoML.

## Architecture

```text
RetailHero CSV files
        |
        v
UpliftProjectContract
        |
        +--> validate tables, treatment, target, scoring split
        +--> compute treatment/control balance diagnostics
        |
        v
Cached feature artifacts
        |
        v
Registered uplift templates
        |
        +--> random baseline
        +--> response model
        +--> two-model
        +--> solo-model
        |
        v
Metrics + ledger + reports + submission artifact
        |
        v
UpliftHypothesisStore
        |
        v
Next: deterministic wave runner for hypothesis-driven experiments
```

The next implementation step is **M2a: wave contracts, deterministic wave runner, and `recipe_comparison` only**. LLM advisory calls come later, after manual waves are safe.

### Teammate-Aligned Supervisor Roadmap

From `Completely change our assumption and philosiphy.md`:

```text
Teammate-aligned direction
        |
        v
Data ingestion + preprocessing
        |
        v
Hypothesis reasoning + strategy selection
        |
        v
Trial spec + feature recipe planning
        |
        v
Deterministic uplift kernel
        |
        +--> registered templates only
        +--> Qini / AUUC / uplift@k / policy gain
        +--> ledger + artifacts
        |
        v
Experiment memory + hypothesis verdicts
        |
        v
Next: conservative UpliftResearchLoop
```

## Repository Layout

```text
src/
  models/uplift.py        # Uplift contracts and Pydantic models
  uplift/                 # Validation, features, metrics, baselines, ledger, reports

demos/
  uplift_validate_dataset.py
  uplift_build_features.py
  uplift_run_baselines.py

tests/
  fixtures/uplift/        # Tiny fixture data for deterministic tests
  models/test_uplift_*.py
  uplift/
  integration/test_uplift_*.py
```

## What Is In This Repo So Far

The repo currently contains the deterministic uplift kernel and the first hypothesis-memory layer. The files under `src/` are the reusable project code:

- `src/models/uplift.py`: Pydantic contracts for RetailHero table paths, split policy, evaluation policy, feature recipes, feature artifacts, trial specs, result cards, ledger records, submission artifacts, and pointer-only hypotheses.
- `src/uplift/validation.py`: dataset validation, treatment/control balance diagnostics, and stratification feasibility checks.
- `src/uplift/features.py`: deterministic customer-level feature builder from `clients.csv` and `purchases.csv`, including cached feature artifact metadata.
- `src/uplift/splitting.py`: labeled train/validation/test splitting from `uplift_train.csv` only.
- `src/uplift/metrics.py`: Qini curve/AUC, uplift curve/AUC, uplift@k, decile tables, and policy-gain helpers.
- `src/uplift/templates.py`: registered baseline learners: random, response model, two-model uplift, and solo-model uplift.
- `src/uplift/loop.py`: deterministic trial runner that executes registered templates, writes prediction/curve/decile/result artifacts, and appends ledger records.
- `src/uplift/ledger.py`: JSONL ledger for trial-level evidence.
- `src/uplift/reporting.py`: markdown report generation plus final `client_id,uplift` submission generation and validation.
- `src/uplift/planner.py`: guarded advisory planner/reporting path; it can suggest from allowed structures but cannot change contract semantics.
- `src/uplift/hypotheses.py`: hypothesis lifecycle helpers and a JSONL-backed `UpliftHypothesisStore`.

The `demos/` folder contains teammate-facing commands that exercise the kernel without needing to read the internals:

- `demos/uplift_validate_dataset.py`: validates the RetailHero-style files and prints schema/balance diagnostics.
- `demos/uplift_build_features.py`: builds cached customer-level feature artifacts for train, scoring, or all cohorts.
- `demos/uplift_run_baselines.py`: builds features, runs the baseline ladder, writes ledger/artifacts/report/submission outputs, and prints a summary.

The `tests/` folder keeps a tiny RetailHero-like fixture dataset and regression coverage for contracts, validation, feature building, metrics, baselines, ledger records, reports, submissions, hypotheses, and demo scripts.

## Setup

Use Python 3.12+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

If you prefer requirements files:

```bash
pip install -r requirements.txt
```

## Test

Run the teammate-facing uplift suite:

```bash
python3 -m pytest \
  tests/models/test_uplift_contracts.py \
  tests/models/test_uplift_hypotheses.py \
  tests/uplift \
  tests/integration/test_uplift_validate_demo.py \
  tests/integration/test_uplift_build_features_demo.py \
  tests/integration/test_uplift_run_baselines_demo.py \
  -q
```

Expected current result:

```text
65 passed
```

## Demo With Tiny Fixtures

Validate the fixture dataset:

```bash
python3 demos/uplift_validate_dataset.py --data-dir tests/fixtures/uplift
```

Build cached features:

```bash
python3 demos/uplift_build_features.py \
  --data-dir tests/fixtures/uplift \
  --output-dir artifacts/uplift/features \
  --cohort train \
  --chunksize 10 \
  --force
```

Run the baseline ladder:

```bash
python3 demos/uplift_run_baselines.py \
  --data-dir tests/fixtures/uplift \
  --output-dir artifacts/uplift/baseline_runs \
  --chunksize 10 \
  --small-fixture-mode
```

Generated artifacts are intentionally ignored by Git.

## Running On RetailHero Data

Place the real RetailHero CSV files locally under:

```text
retailhero-uplift/data/
  clients.csv
  purchases.csv
  products.csv
  uplift_train.csv
  uplift_test.csv
```

The raw dataset is not committed to this repo.

Then run the same demo commands without `--data-dir tests/fixtures/uplift`.

## Design Philosophy

The project uses a conservative execution boundary:

- Contracts own target, treatment, metric, split, and submission semantics.
- The experiment kernel owns fitting, scoring, and metric artifacts.
- Advisory LLM calls may propose hypotheses later, but cannot rewrite contracts or bypass validation.
- Raw purchase data is scanned only when building cached feature artifacts, not inside every experiment loop.

This keeps the project auditable enough for a team setting while leaving room to add a supervisor loop next.
