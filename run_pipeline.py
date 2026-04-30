#!/usr/bin/env python3
"""
AutoLift Pipeline Runner — X5 RetailHero End-to-End
Phases I–VII with stub LLM (no API key required).
"""
from __future__ import annotations

import gzip
import io
import json
import sys
import shutil
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── output directory layout ───────────────────────────────────────────────────
ARTIFACTS_DIR = ROOT / "artifacts" / "pipeline_run"
DATA_DIR       = ARTIFACTS_DIR / "data"
FEATURE_DIR    = ARTIFACTS_DIR / "features"
TRIALS_DIR     = ARTIFACTS_DIR / "trials"
LOGS_DIR       = ARTIFACTS_DIR / "logs"
for d in [DATA_DIR, FEATURE_DIR, TRIALS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LEDGER_PATH  = TRIALS_DIR / "uplift_ledger.jsonl"
REPORT_PATH  = LOGS_DIR  / "final_report.md"


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE I — Data ingestion
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE I — DATA INGESTION")
print("=" * 70)

import pandas as pd
import numpy as np

CACHE_DIR = Path.home() / "scikit-uplift-data"

def _load_gz_csv(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rb") as f:
        return pd.read_csv(io.BytesIO(f.read()))

try:
    clients_df   = _load_gz_csv(CACHE_DIR / "clients.csv.gz")
    train_raw_df = _load_gz_csv(CACHE_DIR / "uplift_train.csv.gz")
    print(f"  clients   : {clients_df.shape}  cols={list(clients_df.columns)}")
    print(f"  train_raw : {train_raw_df.shape}  cols={list(train_raw_df.columns)}")
except Exception as e:
    print(f"ERROR loading cached data: {e}")
    sys.exit(1)

# Build train table (client_id, treatment_flg, target)
train_df = train_raw_df[["client_id", "treatment_flg", "target"]].copy()
print(f"  train_df  : {train_df.shape}")
print(f"  treatment split: {train_df['treatment_flg'].value_counts().to_dict()}")
print(f"  target    split: {train_df['target'].value_counts().to_dict()}")

# Save canonical CSV files for the contract
clients_csv_path = DATA_DIR / "clients.csv"
train_csv_path   = DATA_DIR / "uplift_train.csv"
test_csv_path    = DATA_DIR / "uplift_test.csv"
purch_csv_path   = DATA_DIR / "purchases.csv"

clients_df.to_csv(clients_csv_path, index=False)

# Rename client_id column: train uses 'client_id'
train_df.to_csv(train_csv_path, index=False)

# Scoring table: all clients (just client_id, no labels)
clients_df[["client_id"]].to_csv(test_csv_path, index=False)

# Purchases stub: header-only CSV (not used in demographic recipe)
# Check if purchases is already downloaded and valid
import hashlib

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

purchases_gz = CACHE_DIR / "purchases.csv.gz"
EXPECTED_PURCHASES_HASH = "48d2de13428e24e8b61d66fef02957a8"
purchases_ready = (
    purchases_gz.exists()
    and _md5(purchases_gz) == EXPECTED_PURCHASES_HASH
)

if purchases_ready:
    print("  purchases : full file available — decompressing…")
    purch_df = _load_gz_csv(purchases_gz)
    purch_df.to_csv(purch_csv_path, index=False)
    USE_RFM = True
    print(f"  purchases : {purch_df.shape}")
else:
    print("  purchases : download in progress — using stub header for fingerprint")
    # Write stub CSV with correct columns
    stub_cols = [
        "client_id","transaction_id","transaction_datetime",
        "regular_points_received","express_points_received",
        "regular_points_spent","express_points_spent",
        "purchase_sum","store_id","product_id","product_quantity",
        "trn_sum_from_iss","trn_sum_from_red",
    ]
    pd.DataFrame(columns=stub_cols).to_csv(purch_csv_path, index=False)
    USE_RFM = False
    print("  purchases : stub created (demographic features only this run)")

print()
print("PHASE I COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE II — Experiment Planning (stub LLM)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE II — EXPERIMENT PLANNING")
print("=" * 70)

from src.models.uplift import (
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftFeatureRecipeSpec,
    UpliftTrialSpec,
)
from src.uplift.features import build_feature_table
from src.uplift.hypotheses import UpliftHypothesisStore
from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import make_chat_llm
from src.uplift.loop import run_uplift_trials
from src.uplift.planning_agents import ExperimentPlanningPhase
from src.uplift.orchestrator import (
    ManualBenchmarkAgent,
    ReportingAgent,
    RetryControllerAgent,
)

# ── Contract ──────────────────────────────────────────────────────────────────
contract = UpliftProjectContract(
    task_name="retailhero-uplift",
    description="X5 RetailHero campaign uplift modeling — AutoLift pipeline run.",
    table_schema=UpliftTableSchema(
        clients_table=str(clients_csv_path),
        purchases_table=str(purch_csv_path),
        train_table=str(train_csv_path),
        scoring_table=str(test_csv_path),
    ),
    split_contract=UpliftSplitContract(
        train_fraction=0.70,
        val_fraction=0.15,
        test_fraction=0.15,
        random_seed=42,
    ),
)

# ── LLM + planning infra ───────────────────────────────────────────────────────
llm            = make_chat_llm(provider="stub")
ledger         = UpliftLedger(LEDGER_PATH)
hypothesis_store = UpliftHypothesisStore(LOGS_DIR / "hypotheses.jsonl")

planner = ExperimentPlanningPhase(
    ledger=ledger,
    hypothesis_store=hypothesis_store,
    llm=llm,
)

print("[ExperimentPlanning] Running Phase II planning agents…")
planning_spec = planner.run()
print(f"  Trial ID       : {planning_spec.trial_id}")
print(f"  Hypothesis     : {planning_spec.hypothesis}")
print(f"  Learner family : {planning_spec.learner_family}")
print(f"  Base estimator : {planning_spec.base_estimator}")
print(f"  Feature recipe : {planning_spec.feature_recipe}")
print(f"  Model          : {planning_spec.model}")
print(f"  Stop criteria  : {planning_spec.stop_criteria}")

# Save trial spec for reference
trial_spec_path = LOGS_DIR / "trial_spec.json"
trial_spec_path.write_text(json.dumps({
    "trial_id": planning_spec.trial_id,
    "hypothesis": planning_spec.hypothesis,
    "learner_family": planning_spec.learner_family,
    "base_estimator": planning_spec.base_estimator,
    "feature_recipe": planning_spec.feature_recipe,
    "params": planning_spec.params,
    "model": planning_spec.model,
    "split_seed": planning_spec.split_seed,
    "stop_criteria": planning_spec.stop_criteria,
}, indent=2))
print(f"  → trial_spec saved to {trial_spec_path}")
print()
print("PHASE II COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  Feature Engineering — build one artifact per recipe
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("FEATURE ENGINEERING")
print("=" * 70)

# Choose recipe based on data availability
if USE_RFM:
    recipe = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=[30, 90],
        builder_version="v1",
    )
    recipe_label = "rfm+basket+points (30d,90d windows)"
else:
    recipe = UpliftFeatureRecipeSpec(
        source_tables=["clients"],
        feature_groups=["demographic"],
        builder_version="v1",
    )
    recipe_label = "demographic-only (purchases not yet downloaded)"

print(f"  Recipe: {recipe_label}")
print("  Building feature table…")

try:
    feature_artifact = build_feature_table(
        contract,
        recipe=recipe,
        output_dir=FEATURE_DIR,
        cohort="train",
        chunksize=100_000,
    )
except Exception as e:
    print(f"  ERROR building features: {e}")
    sys.exit(1)

print(f"  Artifact ID    : {feature_artifact.feature_artifact_id}")
print(f"  Recipe ID      : {feature_artifact.feature_recipe_id}")
print(f"  Rows           : {feature_artifact.row_count}")
print(f"  Features       : {feature_artifact.generated_columns}")

# Save feature table path for Phase III handoff
feature_table_path = LOGS_DIR / "feature_table.parquet"
pd.read_csv(feature_artifact.artifact_path).to_parquet(feature_table_path, index=False)
print(f"  → feature_table.parquet saved to {feature_table_path}")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE III — Training Execution (warm-up: all 4 learner families)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE III — TRAINING EXECUTION (warm-up: all 4 learner families)")
print("=" * 70)

from src.uplift.orchestrator import _trial_from_planning_spec, _template_name

def _make_warmup_trials(feature_artifact, split_seed: int = 42) -> list[UpliftTrialSpec]:
    """One trial per learner family with logistic_regression."""
    families = [
        ("response_model", "logistic_regression", "response_model_sklearn"),
        ("solo_model",     "logistic_regression", "solo_model_sklearn"),
        ("two_model",      "logistic_regression", "two_model_sklearn"),
        ("class_transformation", "logistic_regression", "class_transformation_sklearn"),
    ]
    specs = []
    for learner_family, base_estimator, template_name in families:
        specs.append(UpliftTrialSpec(
            spec_id=f"warmup_{learner_family}",
            hypothesis_id=planning_spec.trial_id,
            template_name=template_name,
            learner_family=learner_family,
            base_estimator=base_estimator,
            feature_recipe_id=feature_artifact.feature_recipe_id,
            params={"C": 1.0, "max_iter": 1000},
            split_seed=split_seed,
        ))
    return specs

warmup_trials = _make_warmup_trials(feature_artifact)
print(f"  Executing {len(warmup_trials)} warm-up trials…")

loop_result = run_uplift_trials(
    contract,
    feature_artifact=feature_artifact,
    trial_specs=warmup_trials,
    output_dir=TRIALS_DIR,
)

print(f"\n  Warm-up Results:")
print(f"  {'Trial ID':<30} {'Family':<22} {'Status':<10} {'Qini AUC':<12} {'Uplift AUC'}")
print(f"  {'-'*30} {'-'*22} {'-'*10} {'-'*12} {'-'*12}")

warmup_records = loop_result.records
for r in warmup_records:
    qini = f"{r.qini_auc:.4f}" if r.qini_auc is not None else "N/A"
    uauc = f"{r.uplift_auc:.4f}" if r.uplift_auc is not None else "N/A"
    if r.status == "failed":
        print(f"  {r.run_id:<30} {r.uplift_learner_family:<22} FAILED     {r.error[:40]}")
    else:
        print(f"  {r.run_id:<30} {r.uplift_learner_family:<22} {r.status:<10} {qini:<12} {uauc}")

successful = [r for r in warmup_records if r.status == "success"]
if not successful:
    print("\n  ERROR: No successful warm-up trials. Check error messages above.")
else:
    best = max(successful, key=lambda r: r.qini_auc or float("-inf"))
    print(f"\n  Best warm-up trial: {best.uplift_learner_family} | Qini AUC = {best.qini_auc:.4f}")

print()
print("PHASE III COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE IV — Memory / Ledger check (ExperimentMemoryAgent equivalent)
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE IV — EXPERIMENT MEMORY (Ledger)")
print("=" * 70)

records_on_disk = ledger.load()
print(f"  Records in ledger: {len(records_on_disk)}")
params_hashes = []
seen = set()
for r in records_on_disk:
    collision = r.params_hash in seen
    flag = " ← COLLISION" if collision else ""
    print(f"  run_id={r.run_id} | family={r.uplift_learner_family:<22} | params_hash={r.params_hash}{flag}")
    params_hashes.append(r.params_hash)
    seen.add(r.params_hash)

if len(params_hashes) == len(set(params_hashes)):
    print("  ✓ No collision warnings — all params_hashes are unique")
else:
    print("  ⚠ Collision detected: duplicate params_hashes found")

print()
print("PHASE IV COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE V — Manual Benchmark
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE V — MANUAL BENCHMARK")
print("=" * 70)

benchmark_record = ManualBenchmarkAgent(
    contract,
    feature_artifact=feature_artifact,
    output_dir=TRIALS_DIR,
).run()

if benchmark_record.status == "success":
    print(f"  Benchmark run_id  : {benchmark_record.run_id}")
    print(f"  Template          : {benchmark_record.template_name}")
    print(f"  Learner family    : {benchmark_record.uplift_learner_family}")
    print(f"  Qini AUC          : {benchmark_record.qini_auc}")
    print(f"  Uplift AUC        : {benchmark_record.uplift_auc}")
    print(f"  Uplift@k          : {benchmark_record.uplift_at_k}")
    print(f"  Verdict           : {benchmark_record.verdict}")
else:
    print(f"  Benchmark FAILED: {benchmark_record.error}")

print()
print("PHASE V COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE VI — Retry Decision
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE VI — RETRY DECISION")
print("=" * 70)

retry = RetryControllerAgent(ledger).run()
print(f"  should_continue       : {retry.should_continue}")
print(f"  reason                : {retry.reason}")
print(f"  suggested_next_action : {retry.suggested_next_action}")

print()
print("PHASE VI COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE VII — Additional iteration if should_continue is True
# ─────────────────────────────────────────────────────────────────────────────

all_records = list(warmup_records)
evaluation_results = []

if retry.should_continue:
    print("=" * 70)
    print("PHASE VII — ADDITIONAL ITERATION (retry requested)")
    print("=" * 70)
    for iteration in range(1, 3):  # at most 2 additional iterations
        print(f"\n  --- Iteration {iteration} ---")
        next_spec = planner.run()
        print(f"  Trial ID   : {next_spec.trial_id}")
        print(f"  Family     : {next_spec.learner_family}")
        print(f"  Estimator  : {next_spec.base_estimator}")

        template = _template_name(next_spec.learner_family, next_spec.base_estimator)
        trial_spec = UpliftTrialSpec(
            spec_id=next_spec.trial_id,
            hypothesis_id=next_spec.trial_id,
            template_name=template,
            learner_family=next_spec.learner_family,
            base_estimator=next_spec.base_estimator,
            feature_recipe_id=feature_artifact.feature_recipe_id,
            params=next_spec.params,
            split_seed=next_spec.split_seed,
        )
        iter_result = run_uplift_trials(
            contract,
            feature_artifact=feature_artifact,
            trial_specs=[trial_spec],
            output_dir=TRIALS_DIR,
        )
        r = iter_result.records[0]
        all_records.append(r)
        if r.status == "success":
            print(f"  Qini AUC   : {r.qini_auc:.4f}")
            print(f"  Uplift AUC : {r.uplift_auc:.4f}")
        else:
            print(f"  FAILED: {r.error}")

        retry2 = RetryControllerAgent(ledger).run()
        if not retry2.should_continue:
            print(f"  RetryController: stopping — {retry2.reason}")
            retry = retry2
            break
    print()
    print("PHASE VII COMPLETE")
    print()
else:
    print("PHASE VII — SKIPPED (retry not requested)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  PHASE VIII — Final Report
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("PHASE VIII — FINAL REPORT")
print("=" * 70)

reporter = ReportingAgent(
    contract,
    ledger,
    output_path=REPORT_PATH,
    retry_decision=retry,
    evaluation_results=evaluation_results,
)
report_path = reporter.run()
print(f"  Report saved: {report_path}")
print()
report_text = Path(report_path).read_text()
print("─" * 70)
print(report_text)
print("─" * 70)
print()
print("PHASE VIII COMPLETE")
print()


# ─────────────────────────────────────────────────────────────────────────────
#  FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 70)
print("FINAL SUMMARY TABLE")
print("=" * 70)

all_ledger = ledger.load()

def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)

header = f"{'trial_id':<32} {'family':<22} {'estimator':<22} {'Qini AUC':<10} {'Uplift AUC':<12} {'U@10%':<10} {'U@20%':<10} {'verdict'}"
print(header)
print("-" * len(header))

for r in all_ledger:
    u10 = _fmt(r.uplift_at_k.get("0.1") or r.uplift_at_k.get("0.10"))
    u20 = _fmt(r.uplift_at_k.get("0.2") or r.uplift_at_k.get("0.20"))
    label = "BENCHMARK" if r.hypothesis_id == "manual_baseline" else r.run_id[:28]
    row = (
        f"{label:<32} "
        f"{r.uplift_learner_family:<22} "
        f"{r.base_estimator:<22} "
        f"{_fmt(r.qini_auc):<10} "
        f"{_fmt(r.uplift_auc):<12} "
        f"{u10:<10} "
        f"{u20:<10} "
        f"{r.verdict}"
    )
    print(row)

print()
print("Pipeline run complete.")
