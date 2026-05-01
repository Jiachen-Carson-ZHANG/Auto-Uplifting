#!/usr/bin/env python3
"""Run the BT5153 autonomous uplift pipeline in reproducible stages."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.uplift import (  # noqa: E402
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
)
from src.uplift.features import build_feature_tables_multi_recipe  # noqa: E402
from src.uplift.hypotheses import UpliftHypothesisStore  # noqa: E402
from src.uplift.ledger import UpliftLedger  # noqa: E402
from src.uplift.llm_client import make_chat_llm  # noqa: E402
from src.uplift.metrics import normalized_qini_auc_score  # noqa: E402
from src.uplift.orchestrator import AutoLiftOrchestrator  # noqa: E402
from src.uplift.planning_agents import ExperimentPlanningPhase  # noqa: E402
from src.uplift.recipe_registry import UpliftFeatureRecipeRegistry  # noqa: E402
from src.uplift.reporting import (  # noqa: E402
    generate_submission_artifact,
    validate_submission_artifact,
)
from src.uplift.tuning import (  # noqa: E402
    build_agentic_tuning_plan,
    write_agentic_tuning_plan,
)
from src.uplift.validation import (  # noqa: E402
    compute_treatment_control_balance,
    validate_uplift_dataset,
)

DEFAULT_SEMANTIC_RECIPES = [
    "rfm_baseline",
    "human_semantic_v1",
    "hybrid_safe_semantic_v1",
]


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y"}


def _parse_int_csv(value: str) -> list[int]:
    seeds: list[int] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        seeds.append(int(item))
    return seeds or [42]


def _parse_str_csv(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _api_key_for_provider(provider: str) -> str | None:
    key_by_provider = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
    }
    env_name = key_by_provider.get(provider)
    if env_name is None:
        return None
    value = os.getenv(env_name)
    if not value or value.startswith("PASTE_"):
        raise RuntimeError(f"Set {env_name} in .env before using provider={provider!r}.")
    return value


def _build_contract(data_dir: Path, *, small_fixture_mode: bool) -> UpliftProjectContract:
    split_contract = (
        UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.5,
            test_fraction=0.0,
            min_rows_per_partition=1,
            random_seed=7,
        )
        if small_fixture_mode
        else UpliftSplitContract()
    )
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        description="BT5153 autonomous uplift experiment.",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            products_table=str(data_dir / "products.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
        ),
        split_contract=split_contract,
        feature_sources=["clients", "purchases", "products"],
    )


def _build_default_recipe() -> UpliftFeatureRecipeSpec:
    return UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=[30, 60, 90],
        builder_version="v1",
    )


def _build_feature_artifact_map(
    contract: UpliftProjectContract,
    *,
    recipe_names: list[str],
    output_dir: Path,
    cohort: str,
    chunksize: int,
) -> dict[str, object]:
    registry = UpliftFeatureRecipeRegistry.default()
    recipes = [registry.recipe_for_family(name) for name in recipe_names]
    built = build_feature_tables_multi_recipe(
        contract,
        recipes=recipes,
        output_dir=output_dir,
        cohort=cohort,  # type: ignore[arg-type]
        chunksize=chunksize,
        progress_logger=print,
    )
    return dict(zip(recipe_names, built))


def _scoring_artifact_for_champion(champion, scoring_artifacts_by_name):
    for artifact in scoring_artifacts_by_name.values():
        if artifact.feature_recipe_id == champion.feature_recipe_id:
            return artifact
    raise RuntimeError(
        "Selected agent champion feature recipe has no matching scoring artifact: "
        f"{champion.feature_recipe_id}"
    )


def _champion_trial_from_record(record) -> UpliftTrialSpec:
    return UpliftTrialSpec(
        spec_id=record.hypothesis_id,
        hypothesis_id=record.hypothesis_id,
        template_name=record.template_name,
        learner_family=record.uplift_learner_family,
        base_estimator=record.base_estimator,
        feature_recipe_id=record.feature_recipe_id,
        params={},
        split_seed=record.split_seed,
    )


def _load_model(path: str):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def _champion_model_path(record) -> str:
    model_path = record.artifact_paths.get("model")
    if not model_path:
        raise RuntimeError(
            "Selected agent champion is missing a saved model artifact: "
            f"run_id={record.run_id}, hypothesis_id={record.hypothesis_id}"
        )
    if not Path(model_path).exists():
        raise RuntimeError(
            "Selected agent champion model artifact does not exist: "
            f"{model_path}"
        )
    return model_path


def _successful_agent_champion(records):
    successful = [
        record
        for record in records
        if record.status == "success" and record.hypothesis_id != "manual_baseline"
    ]
    if not successful:
        raise RuntimeError("No successful autonomous trial is available for submission scoring.")
    return max(
        successful,
        key=_validation_champion_score,
    )


def _validation_champion_score(record) -> float:
    validation = _normalized_qini_from_artifact(record, "uplift_scores")
    if validation is not None:
        return validation
    return record.qini_auc if record.qini_auc is not None else float("-inf")


def _normalized_qini_from_artifact(record, artifact_key: str) -> float | None:
    path = getattr(record, "artifact_paths", {}).get(artifact_key)
    if not path:
        return None
    try:
        scores = pd.read_csv(path)
        return normalized_qini_auc_score(
            scores["target"].to_numpy(),
            scores["treatment_flg"].to_numpy(),
            scores["uplift"].to_numpy(),
        )
    except Exception:
        return None


def _format_summary_metric(value) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "n/a"


def _uplift_at_text(record, key: str) -> str:
    value = getattr(record, "uplift_at_k", {}).get(key)
    if value is None:
        return "n/a"
    return _format_summary_metric(value)


def _recipe_display_name(record) -> str:
    """Return semantic_name from artifact metadata if available, else short hash."""
    semantic = getattr(record, "semantic_name", None) or getattr(record, "temporal_policy", None)
    if semantic:
        return semantic
    rid = getattr(record, "feature_recipe_id", "") or ""
    # Try to match via registry
    try:
        registry = UpliftFeatureRecipeRegistry.default()
        for name in ["rfm_baseline", "human_semantic_v1", "hybrid_safe_semantic_v1"]:
            recipe = registry.recipe_for_family(name)
            if recipe.feature_recipe_id == rid:
                return name
    except Exception:
        pass
    return rid[:10]


def _print_final_summary_table(records) -> None:
    # Exclude pre-run tuning candidates — only show main ledger trials.
    main_records = [r for r in records if "__tune_" not in (r.hypothesis_id or "")]
    print("FINAL SUMMARY TABLE")
    print(
        "| role | run_id | learner | estimator | recipe | "
        "val_norm_qini | uplift@10 | verdict |"
    )
    print("|---|---|---|---|---|---:|---:|---|")
    for record in main_records:
        role = "manual" if record.hypothesis_id == "manual_baseline" else "agent"
        val_norm = _normalized_qini_from_artifact(record, "uplift_scores")
        print(
            "| "
            f"{role} | "
            f"{record.run_id} | "
            f"{record.uplift_learner_family} | "
            f"{record.base_estimator} | "
            f"{_recipe_display_name(record)} | "
            f"{_format_summary_metric(val_norm)} | "
            f"{_uplift_at_text(record, 'top_10pct')} | "
            f"{record.verdict} |"
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--planning-model", default=None)
    parser.add_argument("--evaluation-model", default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--retry-max-trials", type=int, default=None)
    parser.add_argument(
        "--enable-pre-run-tuning",
        action="store_true",
        help="Tune planned model params/seeds in a separate ledger before the final trial run.",
    )
    parser.add_argument(
        "--tuning-seeds",
        default=None,
        help="Comma-separated split seeds for pre-run tuning, e.g. 42,7,99,123.",
    )
    parser.add_argument("--tuning-max-param-sets", type=int, default=None)
    parser.add_argument(
        "--write-agentic-tuning-plan",
        action="store_true",
        help=(
            "After the main run, write a deterministic dry-run tuning plan for "
            "the top internal AutoLift candidates without executing it."
        ),
    )
    parser.add_argument("--agentic-tuning-seed", type=int, default=None)
    parser.add_argument("--agentic-tuning-top-k", type=int, default=None)
    parser.add_argument(
        "--agentic-tuning-max-trials-per-candidate",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--semantic-features",
        action="store_true",
        help="Build semantic feature recipes and let the planner choose among them.",
    )
    parser.add_argument(
        "--baseline-features-only",
        action="store_true",
        help="Only build the conservative rfm_baseline feature recipe.",
    )
    parser.add_argument(
        "--semantic-recipes",
        default=None,
        help=(
            "Comma-separated semantic recipes to build. Defaults to "
            "rfm_baseline,human_semantic_v1,hybrid_safe_semantic_v1."
        ),
    )
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--small-fixture-mode", action="store_true")
    benchmark_group = parser.add_mutually_exclusive_group()
    benchmark_group.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Force the manual benchmark to run, overriding BT5153_RUN_BENCHMARK=false.",
    )
    benchmark_group.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the ledger, hypotheses store, and final report before starting. "
             "Use this for a clean run on an existing output directory.",
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path for the pipeline transcript. Defaults to OUTPUT_DIR/pipeline.log.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_local_env(ROOT / args.env_file)

    provider = (args.provider or os.getenv("LLM_PROVIDER", "stub")).strip().lower()
    base_model = args.model or os.getenv("LLM_MODEL") or None
    planning_model = args.planning_model or os.getenv("LLM_PLANNING_MODEL") or base_model
    evaluation_model = (
        args.evaluation_model or os.getenv("LLM_EVALUATION_MODEL") or base_model
    )
    max_iterations = args.max_iterations or int(os.getenv("BT5153_MAX_ITERATIONS", "3"))
    retry_max_trials = args.retry_max_trials or int(
        os.getenv("BT5153_RETRY_MAX_TRIALS", "5")
    )
    enable_pre_run_tuning = args.enable_pre_run_tuning or _bool_env(
        "BT5153_ENABLE_PRE_RUN_TUNING",
        False,
    )
    tuning_seeds = _parse_int_csv(
        args.tuning_seeds or os.getenv("BT5153_TUNING_SEEDS", "42,7,99,123")
    )
    tuning_max_param_sets = args.tuning_max_param_sets or int(
        os.getenv("BT5153_TUNING_MAX_PARAM_SETS", "2")
    )
    write_agentic_tuning = args.write_agentic_tuning_plan or _bool_env(
        "BT5153_WRITE_AGENTIC_TUNING_PLAN",
        False,
    )
    agentic_tuning_seed = args.agentic_tuning_seed or int(
        os.getenv("BT5153_AGENTIC_TUNING_SEED", "20260501")
    )
    agentic_tuning_top_k = args.agentic_tuning_top_k or int(
        os.getenv("BT5153_AGENTIC_TUNING_TOP_K", "2")
    )
    agentic_tuning_max_trials = args.agentic_tuning_max_trials_per_candidate or int(
        os.getenv("BT5153_AGENTIC_TUNING_MAX_TRIALS_PER_CANDIDATE", "16")
    )
    semantic_features = args.semantic_features or (
        not args.baseline_features_only
        and _bool_env("BT5153_SEMANTIC_FEATURES", True)
    )
    recipe_names = (
        _parse_str_csv(args.semantic_recipes)
        if args.semantic_recipes
        else list(DEFAULT_SEMANTIC_RECIPES if semantic_features else ["rfm_baseline"])
    )
    run_benchmark = args.run_benchmark or (
        (not args.skip_benchmark)
        and _bool_env(
            "BT5153_RUN_BENCHMARK",
            True,
        )
    )
    data_dir = Path(args.data_dir or os.getenv("RETAILHERO_DATA_DIR", "tests/fixtures/uplift"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    output_dir = Path(
        args.output_dir
        or os.getenv("BT5153_OUTPUT_DIR")
        or f"artifacts/uplift/run_{datetime.now():%Y%m%d_%H%M%S}"
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    if args.reset and output_dir.exists():
        import shutil
        for stale in [output_dir / "hypotheses.jsonl", output_dir / "pipeline.log"]:
            if stale.exists():
                stale.unlink()
        for stale_dir in [output_dir / "runs"]:
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        submission = output_dir / "uplift_submission.csv"
        if submission.exists():
            submission.unlink()
        print(f"[reset] cleared ledger and state in {output_dir}")

    log_path = Path(args.log_file) if args.log_file else output_dir / "pipeline.log"
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    log_handle = log_path.open("a", encoding="utf-8")
    sys.stdout = _Tee(original_stdout, log_handle)  # type: ignore[assignment]
    sys.stderr = _Tee(original_stderr, log_handle)  # type: ignore[assignment]

    try:
        api_key = _api_key_for_provider(provider)
        print(f"Log file: {log_path}")
        print(
            "Stage 1/7: configuration "
            f"provider={provider}, planning_model={planning_model}, "
            f"evaluation_model={evaluation_model}, max_iterations={max_iterations}, "
            f"retry_max_trials={retry_max_trials}, "
            f"pre_run_tuning={enable_pre_run_tuning}, "
            f"write_agentic_tuning={write_agentic_tuning}, "
            f"semantic_features={semantic_features}, recipes={recipe_names}"
        )

        print("Stage 2/7: contract and dataset validation")
        contract = _build_contract(data_dir, small_fixture_mode=args.small_fixture_mode)
        validation = validate_uplift_dataset(contract)
        train_df = pd.read_csv(contract.table_schema.train_table)
        balance = compute_treatment_control_balance(
            train_df,
            entity_key=contract.entity_key,
            treatment_col=contract.treatment_column,
            target_col=contract.target_column,
        )
        if not validation.valid:
            raise RuntimeError(f"Dataset validation failed: {validation.errors}")

        print("Stage 3/7: feature artifacts")
        feature_dir = output_dir / "features"
        train_artifacts_by_name = _build_feature_artifact_map(
            contract,
            recipe_names=recipe_names,
            output_dir=feature_dir,
            cohort="train",
            chunksize=args.chunksize,
        )
        scoring_artifacts_by_name = _build_feature_artifact_map(
            contract,
            recipe_names=recipe_names,
            output_dir=feature_dir,
            cohort="scoring",
            chunksize=args.chunksize,
        )
        train_artifact = train_artifacts_by_name[recipe_names[0]]

        print("Stage 4/7: autonomous planning, execution, evaluation, and retry")
        run_dir = output_dir / "runs"
        ledger = UpliftLedger(run_dir / "uplift_ledger.jsonl")
        hypothesis_store = UpliftHypothesisStore(output_dir / "hypotheses.jsonl")
        planner_llm = make_chat_llm(provider, planning_model, api_key)
        evaluation_llm = make_chat_llm(provider, evaluation_model, api_key)
        planner = ExperimentPlanningPhase(
            ledger,
            hypothesis_store,
            planner_llm,
            available_feature_recipes=list(train_artifacts_by_name),
        )
        result = AutoLiftOrchestrator(
            contract=contract,
            planner=planner,
            feature_artifacts_by_name=train_artifacts_by_name,
            output_dir=run_dir,
            llm=evaluation_llm,
            run_benchmark=run_benchmark,
            retry_max_trials=retry_max_trials,
            enable_pre_run_tuning=enable_pre_run_tuning,
            tuning_split_seeds=tuple(tuning_seeds),
            tuning_max_param_sets=tuning_max_param_sets,
        ).run(max_iterations=max_iterations)
        retry_snapshot = result.retry_decision

        print("Stage 5/7: submission preview")
        records = ledger.load()
        agentic_tuning_plan_path = None
        agentic_tuning_trial_specs = 0
        agentic_tuning_candidate_templates: list[str] = []
        champion = _successful_agent_champion(records)
        scoring_artifact = _scoring_artifact_for_champion(
            champion,
            scoring_artifacts_by_name,
        )
        submission = generate_submission_artifact(
            contract,
            model=_load_model(_champion_model_path(champion)),
            scoring_feature_artifact=scoring_artifact,
            champion_trial=_champion_trial_from_record(champion),
            output_path=output_dir / "uplift_submission.csv",
        )
        validate_submission_artifact(contract, submission)

        print("Stage 6/7: ledger summary")
        for record in records:
            print(
                f"- {record.run_id}: {record.template_name} "
                f"status={record.status} qini_auc={record.qini_auc}"
            )
        _print_final_summary_table(records)
        if write_agentic_tuning:
            tuning_plan = build_agentic_tuning_plan(
                records,
                llm=planner_llm,
                tuning_seed=agentic_tuning_seed,
                top_k=agentic_tuning_top_k,
                max_trials_per_candidate=agentic_tuning_max_trials,
            )
            agentic_tuning_plan_path = write_agentic_tuning_plan(
                output_dir / "agentic_tuning_plan.json",
                tuning_plan,
            )
            agentic_tuning_trial_specs = len(tuning_plan.trial_specs)
            agentic_tuning_candidate_templates = [
                candidate.template_name for candidate in tuning_plan.candidates
            ]
            print(
                "[agentic_tuning] dry-run plan "
                f"path={agentic_tuning_plan_path} "
                f"candidates={agentic_tuning_candidate_templates} "
                f"trial_specs={agentic_tuning_trial_specs}"
            )

        print("Stage 7/7: final artifact index")
        summary = {
            "provider": provider,
            "planning_model": planning_model,
            "evaluation_model": evaluation_model,
            "max_iterations": max_iterations,
            "retry_max_trials": retry_max_trials,
            "run_benchmark": run_benchmark,
            "enable_pre_run_tuning": enable_pre_run_tuning,
            "tuning_seeds": tuning_seeds,
            "tuning_max_param_sets": tuning_max_param_sets,
            "write_agentic_tuning_plan": write_agentic_tuning,
            "agentic_tuning_seed": agentic_tuning_seed,
            "agentic_tuning_top_k": agentic_tuning_top_k,
            "agentic_tuning_max_trials_per_candidate": agentic_tuning_max_trials,
            "agentic_tuning_plan_path": agentic_tuning_plan_path,
            "agentic_tuning_trial_specs": agentic_tuning_trial_specs,
            "agentic_tuning_candidate_templates": agentic_tuning_candidate_templates,
            "dataset_rows": validation.table_rows,
            "balance_warnings": balance.warnings,
            "n_agent_trials": len(result.trial_records),
            "n_evaluation_results": len(result.evaluation_results),
            "n_ledger_records": len(records),
            "retry_should_continue": result.retry_decision.should_continue,
            "retry_reason": result.retry_decision.reason,
            "retry_snapshot_reason": retry_snapshot.reason,
            "semantic_features": semantic_features,
            "feature_recipes": recipe_names,
            "train_feature_artifact": train_artifact.artifact_path,
            "train_feature_artifacts": {
                name: artifact.artifact_path
                for name, artifact in train_artifacts_by_name.items()
            },
            "scoring_feature_artifact": scoring_artifact.artifact_path,
            "ledger_path": str(ledger.path),
            "report_path": result.report_path,
            "submission_path": submission.artifact_path,
            "hypothesis_store_path": str(hypothesis_store.path),
            "log_path": str(log_path),
        }
        print("SUMMARY_JSON=" + json.dumps(summary, sort_keys=True))
        return 0
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
