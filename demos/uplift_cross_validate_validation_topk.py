#!/usr/bin/env python3
"""Cross-validate the top validation-selected uplift candidates.

Selection is deliberately two-stage and leakage-safe:

1. Rank candidates using validation predictions only.
2. Refit and cross-validate only the top-k candidates on the original
   train+validation pool, leaving the internal test partition sealed for the
   final audit.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demos.uplift_cross_validate_champion import (  # noqa: E402
    CrossValidationCandidate,
    _build_contract,
    _labeled_feature_frame,
    run_cross_validation,
)
from src.models.uplift import UpliftFeatureArtifact, UpliftProjectContract  # noqa: E402
from src.uplift.metrics import normalized_qini_auc_score  # noqa: E402
from src.uplift.splitting import split_labeled_uplift_frame  # noqa: E402


DEFAULT_LEDGER_PATHS = [
    ROOT / "results" / "run_20260430_best" / "uplift_ledger.jsonl",
    ROOT
    / "results"
    / "run_20260430_best"
    / "agentic_tuning_validation_only_ledger.jsonl",
]
DEFAULT_PLAN_PATHS = [
    ROOT
    / "results"
    / "run_20260430_best"
    / "agentic_tuning_plan_validation_only.json",
]
DEFAULT_FEATURE_METADATA_GLOBS = [
    str(
        ROOT
        / "artifacts"
        / "uplift"
        / "run_20260430_221602"
        / "features"
        / "uplift_features_train_*.metadata.json"
    ),
    str(
        ROOT
        / "artifacts"
        / "uplift"
        / "scratch_full_rerun_20260501_034315"
        / "features"
        / "uplift_features_train_*.metadata.json"
    ),
]


@dataclass(frozen=True)
class TopKCrossValidationResult:
    output_dir: str
    summary_path: str
    leaderboard_path: str
    selected_candidate_run_id: str
    selected_candidate_hypothesis_id: str


def load_validation_candidates(
    ledger_paths: Iterable[str | Path],
    *,
    plan_paths: Iterable[str | Path] = (),
    top_k: int = 3,
) -> list[CrossValidationCandidate]:
    """Return top candidates ranked only by validation predictions."""
    trial_params = _load_trial_params(plan_paths)
    rows: list[dict[str, Any]] = []
    seen_run_ids: set[str] = set()
    for ledger_path in ledger_paths:
        path = _resolve(ledger_path)
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            run_id = str(record.get("run_id") or "")
            if not run_id or run_id in seen_run_ids:
                continue
            seen_run_ids.add(run_id)
            row = _validation_candidate_row(record, source_ledger=path)
            if row is not None:
                row["params"] = trial_params.get(
                    str(record.get("hypothesis_id") or ""),
                    record.get("params") if isinstance(record.get("params"), dict) else {},
                )
                rows.append(row)

    rows.sort(
        key=lambda row: (
            float(row["validation_normalized_qini_auc"]),
            float(row["validation_qini_auc"]),
        ),
        reverse=True,
    )
    candidates: list[CrossValidationCandidate] = []
    for rank, row in enumerate(rows[:top_k], start=1):
        candidates.append(
            CrossValidationCandidate(
                run_id=row["run_id"],
                hypothesis_id=row["hypothesis_id"],
                template_name=row["template_name"],
                learner_family=row["learner_family"],
                base_estimator=row["base_estimator"],
                feature_recipe_id=row["feature_recipe_id"],
                feature_artifact_id=row["feature_artifact_id"],
                params=row["params"],
                params_hash=row["params_hash"],
                split_seed=row["split_seed"],
                validation_qini_auc=row["validation_qini_auc"],
                validation_normalized_qini_auc=row[
                    "validation_normalized_qini_auc"
                ],
                validation_uplift_auc=row["validation_uplift_auc"],
                validation_rank=rank,
                source_ledger=row["source_ledger"],
            )
        )
    return candidates


def run_validation_topk_cross_validation(
    contract: UpliftProjectContract,
    *,
    ledger_paths: Iterable[str | Path],
    plan_paths: Iterable[str | Path],
    feature_metadata_globs: Iterable[str],
    output_dir: str | Path,
    top_k: int = 3,
    n_folds: int = 5,
    seed: int = 20260501,
) -> TopKCrossValidationResult:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    ledger_paths = list(ledger_paths)
    plan_paths = list(plan_paths)
    feature_metadata_globs = list(feature_metadata_globs)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidates = load_validation_candidates(
        ledger_paths,
        plan_paths=plan_paths,
        top_k=top_k,
    )
    if not candidates:
        raise RuntimeError("No validation-ranked candidates were available.")

    feature_artifacts = _load_feature_artifacts(feature_metadata_globs)
    candidate_rows = [candidate.to_summary_dict() for candidate in candidates]
    pd.DataFrame(candidate_rows).to_csv(
        output / "validation_topk_candidates.csv",
        index=False,
    )

    cv_rows: list[dict[str, Any]] = []
    cv_summaries: list[dict[str, Any]] = []
    split_summary: dict[str, Any] | None = None
    for candidate in candidates:
        feature_artifact = _feature_artifact_for_candidate(candidate, feature_artifacts)
        labeled = _labeled_feature_frame(contract, feature_artifact)
        split = split_labeled_uplift_frame(labeled, contract)
        cv_pool = pd.concat([split.train, split.validation], ignore_index=True)
        split_summary = {
            "full_labeled_rows": int(len(labeled)),
            "cv_pool_rows": int(len(cv_pool)),
            "internal_train_rows": int(len(split.train)),
            "internal_validation_rows": int(len(split.validation)),
            "sealed_internal_test_rows": int(len(split.test)),
            "internal_split_strategy": split.strategy,
        }
        candidate_output = output / (
            f"rank_{candidate.validation_rank:02d}_{_slug(candidate.run_id)}"
        )
        result = run_cross_validation(
            contract,
            feature_artifact=feature_artifact,
            output_dir=candidate_output,
            candidate=candidate,
            labeled_frame=cv_pool,
            pool_name="internal_train_plus_validation_only",
            selection_policy=(
                "top-k selected by validation normalized Qini only; "
                "internal test partition excluded from CV"
            ),
            n_folds=n_folds,
            seed=seed,
        )
        summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
        cv_summaries.append(summary)
        metrics = summary["metrics"]
        cv_rows.append(
            {
                "validation_rank": candidate.validation_rank,
                "run_id": candidate.run_id,
                "hypothesis_id": candidate.hypothesis_id,
                "template_name": candidate.template_name,
                "base_estimator": candidate.base_estimator,
                "validation_normalized_qini_auc": (
                    candidate.validation_normalized_qini_auc
                ),
                "validation_qini_auc": candidate.validation_qini_auc,
                "cv_mean_normalized_qini_auc": metrics[
                    "normalized_qini_auc"
                ]["mean"],
                "cv_std_normalized_qini_auc": metrics["normalized_qini_auc"][
                    "std"
                ],
                "cv_mean_qini_auc": metrics["qini_auc"]["mean"],
                "cv_mean_uplift_auc": metrics["uplift_auc"]["mean"],
                "cv_mean_uplift_top_5pct": metrics.get(
                    "uplift_top_5pct", {}
                ).get("mean"),
                "cv_mean_uplift_top_10pct": metrics.get(
                    "uplift_top_10pct", {}
                ).get("mean"),
                "cv_summary_path": result.summary_path,
            }
        )

    leaderboard = pd.DataFrame(cv_rows).sort_values(
        ["cv_mean_normalized_qini_auc", "validation_normalized_qini_auc"],
        ascending=False,
    )
    leaderboard_path = output / "cv_leaderboard.csv"
    leaderboard.to_csv(leaderboard_path, index=False)
    winner = leaderboard.iloc[0].to_dict()
    summary_payload = {
        "selection_policy": (
            "Validation predictions select the top-k candidates. CV reranks "
            "those candidates using only the original train+validation pool. "
            "The internal test partition is excluded from candidate selection."
        ),
        "top_k": top_k,
        "n_folds": n_folds,
        "seed": seed,
        "input_ledgers": [str(_resolve(path)) for path in ledger_paths],
        "input_plans": [str(_resolve(path)) for path in plan_paths],
        "feature_metadata_globs": list(feature_metadata_globs),
        "split_summary": split_summary,
        "validation_topk_candidates": candidate_rows,
        "cv_leaderboard": leaderboard.to_dict(orient="records"),
        "cv_summaries": cv_summaries,
        "selected_candidate_run_id": winner["run_id"],
        "selected_candidate_hypothesis_id": winner["hypothesis_id"],
        "selected_by": "cv_mean_normalized_qini_auc",
        "leaderboard_path": str(leaderboard_path),
    }
    summary_path = output / "topk_cv_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output / "TOPK_CV_SUMMARY.md").write_text(
        _topk_markdown(summary_payload, leaderboard),
        encoding="utf-8",
    )
    return TopKCrossValidationResult(
        output_dir=str(output),
        summary_path=str(summary_path),
        leaderboard_path=str(leaderboard_path),
        selected_candidate_run_id=str(winner["run_id"]),
        selected_candidate_hypothesis_id=str(winner["hypothesis_id"]),
    )


def _validation_candidate_row(
    record: dict[str, Any],
    *,
    source_ledger: Path,
) -> dict[str, Any] | None:
    if record.get("status") != "success":
        return None
    if record.get("hypothesis_id") == "manual_baseline":
        return None
    artifact_paths = record.get("artifact_paths") or {}
    validation_path = artifact_paths.get("uplift_scores")
    if not validation_path:
        return None
    validation_file = Path(validation_path)
    if not validation_file.exists():
        return None
    scores = pd.read_csv(validation_file)
    validation_normalized = normalized_qini_auc_score(
        scores["target"].to_numpy(),
        scores["treatment_flg"].to_numpy(),
        scores["uplift"].to_numpy(),
    )
    validation_qini = record.get("qini_auc")
    if validation_qini is None:
        return None
    return {
        "run_id": str(record.get("run_id") or ""),
        "hypothesis_id": str(record.get("hypothesis_id") or ""),
        "template_name": str(record.get("template_name") or ""),
        "learner_family": str(record.get("uplift_learner_family") or ""),
        "base_estimator": str(record.get("base_estimator") or ""),
        "feature_recipe_id": str(record.get("feature_recipe_id") or ""),
        "feature_artifact_id": str(record.get("feature_artifact_id") or ""),
        "params_hash": str(record.get("params_hash") or ""),
        "split_seed": record.get("split_seed"),
        "validation_qini_auc": float(validation_qini),
        "validation_normalized_qini_auc": round(float(validation_normalized), 6),
        "validation_uplift_auc": record.get("uplift_auc"),
        "source_ledger": str(source_ledger),
    }


def _load_trial_params(plan_paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    params: dict[str, dict[str, Any]] = {}
    for plan_path in plan_paths:
        path = _resolve(plan_path)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        for spec in payload.get("trial_specs", []):
            hypothesis_id = spec.get("hypothesis_id")
            spec_params = spec.get("params")
            if isinstance(hypothesis_id, str) and isinstance(spec_params, dict):
                params[hypothesis_id] = spec_params
    return params


def _load_feature_artifacts(
    feature_metadata_globs: Iterable[str],
) -> dict[tuple[str, str], UpliftFeatureArtifact]:
    artifacts: dict[tuple[str, str], UpliftFeatureArtifact] = {}
    for pattern in feature_metadata_globs:
        resolved_pattern = (
            str(Path(pattern))
            if Path(pattern).is_absolute()
            else str(ROOT / str(pattern))
        )
        for matched in sorted(glob.glob(resolved_pattern)):
            path = Path(matched)
            artifact = UpliftFeatureArtifact.model_validate_json(
                path.read_text(encoding="utf-8")
            )
            artifacts[(artifact.feature_recipe_id, artifact.feature_artifact_id)] = artifact
    return artifacts


def _feature_artifact_for_candidate(
    candidate: CrossValidationCandidate,
    artifacts: dict[tuple[str, str], UpliftFeatureArtifact],
) -> UpliftFeatureArtifact:
    key = (candidate.feature_recipe_id, candidate.feature_artifact_id)
    if key in artifacts:
        return artifacts[key]
    recipe_matches = [
        artifact
        for (recipe_id, _), artifact in artifacts.items()
        if recipe_id == candidate.feature_recipe_id
    ]
    if len(recipe_matches) == 1:
        return recipe_matches[0]
    raise RuntimeError(
        "No unique feature artifact found for candidate "
        f"{candidate.run_id} ({candidate.feature_recipe_id}, "
        f"{candidate.feature_artifact_id})."
    )


def _topk_markdown(summary: dict[str, Any], leaderboard: pd.DataFrame) -> str:
    lines = [
        "# Validation Top-K Cross-Validation Audit",
        "",
        "Candidates are selected from validation predictions only. The internal test partition is excluded from CV and remains a final audit surface.",
        "",
        f"- Top-k: {summary['top_k']}",
        f"- Folds: {summary['n_folds']}",
        f"- Seed: `{summary['seed']}`",
        f"- Selected by CV: `{summary['selected_candidate_run_id']}` / `{summary['selected_candidate_hypothesis_id']}`",
        "",
        "## CV Leaderboard",
        "",
        "| Validation Rank | Run | Template | Val Norm Qini | CV Mean Norm Qini | CV Std Norm Qini | CV Mean Uplift@10% |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in leaderboard.to_dict(orient="records"):
        lines.append(
            "| {rank} | `{run}` | `{template}` | {val:.6f} | {cv_mean:.6f} | "
            "{cv_std:.6f} | {uplift10:.6f} |".format(
                rank=int(row["validation_rank"]),
                run=row["run_id"],
                template=row["template_name"],
                val=float(row["validation_normalized_qini_auc"]),
                cv_mean=float(row["cv_mean_normalized_qini_auc"]),
                cv_std=float(row["cv_std_normalized_qini_auc"]),
                uplift10=float(row["cv_mean_uplift_top_10pct"]),
            )
        )
    lines.extend(
        [
            "",
            "## Split Boundary",
            "",
        ]
    )
    split = summary.get("split_summary") or {}
    for key, value in split.items():
        lines.append(f"- {key}: `{value}`")
    lines.append("")
    return "\n".join(lines)


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _slug(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="retailhero-uplift/data")
    parser.add_argument("--ledger", action="append", dest="ledgers")
    parser.add_argument("--plan", action="append", dest="plans")
    parser.add_argument(
        "--feature-metadata-glob",
        action="append",
        dest="feature_metadata_globs",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/uplift/cv_top3_validation_only",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260501)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    contract = _build_contract(_resolve(args.data_dir))
    result = run_validation_topk_cross_validation(
        contract,
        ledger_paths=args.ledgers or DEFAULT_LEDGER_PATHS,
        plan_paths=args.plans or DEFAULT_PLAN_PATHS,
        feature_metadata_globs=args.feature_metadata_globs
        or DEFAULT_FEATURE_METADATA_GLOBS,
        output_dir=_resolve(args.output_dir),
        top_k=args.top_k,
        n_folds=args.folds,
        seed=args.seed,
    )
    print(
        "SUMMARY_JSON="
        + json.dumps(
            {
                "output_dir": result.output_dir,
                "summary_path": result.summary_path,
                "leaderboard_path": result.leaderboard_path,
                "selected_candidate_run_id": result.selected_candidate_run_id,
                "selected_candidate_hypothesis_id": (
                    result.selected_candidate_hypothesis_id
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
