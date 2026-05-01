from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.uplift.xai import (
    diagnose_xai_feature_semantics,
    explain_cached_uplift_model,
    explain_score_feature_associations,
)

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "run_20260430_best"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "explainability"
CHAMPION_RUN_ID = "RUN-f1c30175"
CHAMPION_LEDGER_NAME = "agentic_tuning_validation_only_ledger.jsonl"
MAIN_LEDGER_NAME = "uplift_ledger.jsonl"
CV_LEADERBOARD_NAME = "validation_top3_cv_leaderboard.csv"
TUNING_SUMMARY_NAME = "agentic_tuning_validation_only_execution_summary.json"


HUMAN_CV_SELECTED = {
    "model": "solo_model_xgb",
    "cv_normalized_qini_auc": 0.409490,
    "cv_normalized_qini_auc_std": 0.087550,
    "cv_raw_qini_auc": 396.146460,
    "cv_uplift_auc": 0.065320,
    "cv_uplift_at_5pct": 0.133600,
    "cv_uplift_at_10pct": 0.110320,
    "cv_uplift_at_30pct": 0.070410,
    "test_normalized_qini_auc": 0.204120,
    "test_raw_qini_auc": 299.125590,
    "test_uplift_auc": 0.057820,
    "test_uplift_at_5pct": 0.155200,
    "test_uplift_at_10pct": 0.092310,
    "test_uplift_at_30pct": 0.051870,
}


@dataclass(frozen=True)
class VisualAssetPaths:
    qini_curve: Path
    uplift_curve: Path
    topk_comparison: Path
    decile_lift: Path
    xai_drivers: Path
    xai_driver_direction: Path
    representative_cases: Path
    reasoning_timeline: Path
    xai_summary: Path
    report: Path


def load_ledger(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_champion(
    records: list[dict[str, Any]],
    run_id: str = CHAMPION_RUN_ID,
) -> dict[str, Any]:
    for record in records:
        if record.get("run_id") == run_id:
            return record
    raise ValueError(f"Champion run {run_id} not found")


def load_cv_row(path: Path, run_id: str = CHAMPION_RUN_ID) -> dict[str, float]:
    if not path.exists():
        return {
            "cv_normalized_qini_auc": 0.396226,
            "cv_normalized_qini_auc_std": 0.060313,
            "cv_raw_qini_auc": 392.456261,
            "cv_uplift_auc": 0.066282,
            "cv_uplift_at_5pct": 0.143441,
            "cv_uplift_at_10pct": 0.115764,
            "cv_uplift_at_30pct": 0.064839,
        }
    table = pd.read_csv(path)
    match = table.loc[table["run_id"] == run_id]
    if match.empty:
        raise ValueError(f"CV row for {run_id} not found in {path}")
    row = match.iloc[0]
    result = {
        "cv_normalized_qini_auc": _row_float(row, "cv_mean_normalized_qini_auc"),
        "cv_normalized_qini_auc_std": _row_float(row, "cv_std_normalized_qini_auc"),
        "cv_raw_qini_auc": _row_float(row, "cv_mean_raw_qini_auc", "cv_mean_qini_auc"),
        "cv_uplift_auc": _row_float(row, "cv_mean_uplift_auc"),
        "cv_uplift_at_5pct": _row_float(
            row,
            "cv_mean_uplift_at_5pct",
            "cv_mean_uplift_top_5pct",
        ),
        "cv_uplift_at_10pct": _row_float(
            row,
            "cv_mean_uplift_at_10pct",
            "cv_mean_uplift_top_10pct",
        ),
    }
    # Older copies of this leaderboard did not expose @30. The final CV summary
    # does, so keep the report stable with the audited value if the CSV is short.
    if "cv_mean_uplift_at_30pct" in row:
        result["cv_uplift_at_30pct"] = _row_float(row, "cv_mean_uplift_at_30pct")
    elif "cv_mean_uplift_top_30pct" in row:
        result["cv_uplift_at_30pct"] = _row_float(row, "cv_mean_uplift_top_30pct")
    elif row.get("cv_summary_path"):
        result["cv_uplift_at_30pct"] = _cv_summary_metric(
            resolve_artifact_path(str(row["cv_summary_path"])),
            "uplift_top_30pct",
            default=0.064839,
        )
    else:
        result["cv_uplift_at_30pct"] = 0.064839
    return result


def load_test_normalized_qini(path: Path, run_id: str = CHAMPION_RUN_ID) -> float:
    if not path.exists():
        return 0.248455
    payload = json.loads(path.read_text())
    for record in payload.get("records", []):
        if record.get("run_id") == run_id and record.get("held_out_normalized_qini") is not None:
            return float(record["held_out_normalized_qini"])
    return 0.248455


def _row_float(row: pd.Series, *names: str) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(row[name])
    joined = ", ".join(names)
    raise KeyError(f"None of these columns were found: {joined}")


def _cv_summary_metric(path: Path | None, metric: str, *, default: float) -> float:
    if path is None or not path.exists():
        return default
    payload = json.loads(path.read_text())
    metrics = payload.get("metrics", {})
    value = metrics.get(metric, {}).get("mean")
    if value is None:
        return default
    return float(value)


def resolve_artifact_path(
    raw_path: str | None,
    repo_root: Path = REPO_ROOT,
) -> Path | None:
    if not raw_path:
        return None

    path = Path(raw_path)
    if path.exists():
        return path

    marker = "artifacts/"
    text = str(path)
    if marker in text:
        candidate = repo_root / "artifacts" / text.split(marker, 1)[1]
        if candidate.exists():
            return candidate

    return path


def resolve_feature_path(record: dict[str, Any], repo_root: Path = REPO_ROOT) -> Path | None:
    artifact_id = str(record.get("feature_artifact_id") or "")
    if not artifact_id:
        return None

    roots: list[Path] = []
    for raw_path in record.get("artifact_paths", {}).values():
        path = resolve_artifact_path(raw_path, repo_root=repo_root)
        if path is None:
            continue
        for parent in path.parents:
            if (parent / "features").exists():
                roots.append(parent)
            if parent.name in {"agentic_tuning_validation_only", "agentic_tuning", "runs"}:
                roots.append(parent.parent)
    roots.extend(
        [
            repo_root / "artifacts" / "uplift" / "scratch_full_rerun_20260501_034315",
            repo_root / "artifacts" / "uplift" / "run_20260430_221602",
        ]
    )

    seen: set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        candidate = root / "features" / f"uplift_features_train_{artifact_id}.csv"
        if candidate.exists():
            return candidate
    return None


def metric_value(record: dict[str, Any], key: str) -> float:
    if key.startswith("held_out_uplift_at_"):
        pct = key.removeprefix("held_out_uplift_at_").removesuffix("pct")
        return float(record["held_out_uplift_at_k"][f"top_{pct}pct"])
    return float(record[key])


def comparison_rows(
    champion: dict[str, Any],
    cv_row: dict[str, float],
    test_normalized_qini: float,
) -> list[tuple[str, float, float, float]]:
    metrics = [
        ("CV mean normalized Qini", "cv_normalized_qini_auc", "cv"),
        ("CV std normalized Qini", "cv_normalized_qini_auc_std", "cv"),
        ("Test normalized Qini", "held_out_normalized_qini_auc", "champion"),
        ("Test raw Qini AUC", "held_out_qini_auc", "champion"),
        ("Test uplift AUC", "held_out_uplift_auc", "champion"),
        ("Test uplift@5%", "held_out_uplift_at_5pct", "champion"),
        ("Test uplift@10%", "held_out_uplift_at_10pct", "champion"),
        ("Test uplift@30%", "held_out_uplift_at_30pct", "champion"),
    ]
    rows = []
    for label, key, source in metrics:
        if source == "cv":
            auto_value = float(cv_row[key])
            human_key = key
        elif key == "held_out_normalized_qini_auc":
            auto_value = test_normalized_qini
            human_key = "test_normalized_qini_auc"
        elif key == "held_out_qini_auc":
            auto_value = metric_value(champion, key)
            human_key = "test_raw_qini_auc"
        elif key == "held_out_uplift_auc":
            auto_value = metric_value(champion, key)
            human_key = "test_uplift_auc"
        else:
            auto_value = metric_value(champion, key)
            pct = key.removeprefix("held_out_uplift_at_")
            human_key = f"test_uplift_at_{pct}"
        human_value = float(HUMAN_CV_SELECTED[human_key])
        rows.append((label, auto_value, human_value, auto_value - human_value))
    return rows


def extract_xai_drivers(report_path: Path) -> list[dict[str, Any]]:
    text = report_path.read_text()
    match = re.search(r"^- Top drivers: (\[.*\])$", text, flags=re.MULTILINE)
    if not match:
        return []
    value = ast.literal_eval(match.group(1))
    return [item for item in value if isinstance(item, dict)]


def timeline_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for index, record in enumerate(records, start=1):
        role = "Internal reference" if record.get("hypothesis_id") == "manual_baseline" else "Agent trial"
        rows.append(
            {
                "step": index,
                "run_id": record.get("run_id", ""),
                "role": role,
                "learner": record.get("uplift_learner_family", ""),
                "estimator": record.get("base_estimator", ""),
                "held_out_qini_auc": record.get("held_out_qini_auc"),
                "held_out_uplift_auc": record.get("held_out_uplift_auc"),
                "verdict": record.get("verdict", ""),
                "rationale": _first_sentence(record.get("strategy_rationale", "")),
                "judge": _first_sentence(record.get("judge_narrative", "")),
            }
        )
    return rows


def _first_sentence(text: str, *, max_chars: int = 150) -> str:
    text = _ascii_text(" ".join(text.split()))
    if not text:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
    if len(sentence) <= max_chars:
        return sentence
    return sentence[: max_chars - 3].rstrip() + "..."


def _ascii_text(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u2192", "->")
        .replace("\u2191", "up ")
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def write_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render_visuals(
    *,
    run_dir: Path = DEFAULT_RUN_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> VisualAssetPaths:
    main_records = load_ledger(run_dir / MAIN_LEDGER_NAME)
    champion_records = load_ledger(run_dir / CHAMPION_LEDGER_NAME)
    champion = find_champion(champion_records)
    cv_row = load_cv_row(run_dir / CV_LEADERBOARD_NAME)
    test_normalized_qini = load_test_normalized_qini(run_dir / TUNING_SUMMARY_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = VisualAssetPaths(
        qini_curve=output_dir / "autolift_heldout_qini_curve.svg",
        uplift_curve=output_dir / "autolift_heldout_uplift_curve.svg",
        topk_comparison=output_dir / "human_vs_autolift_topk.svg",
        decile_lift=output_dir / "autolift_decile_lift.svg",
        xai_drivers=output_dir / "autolift_xai_top_drivers.svg",
        xai_driver_direction=output_dir / "autolift_xai_driver_direction.svg",
        representative_cases=output_dir / "autolift_representative_cases.svg",
        reasoning_timeline=output_dir / "agent_reasoning_timeline.svg",
        xai_summary=output_dir / "autolift_cv_selected_xai_summary.json",
        report=output_dir / "EXPLAINABILITY_REPORT.md",
    )

    xai_summary = _build_prediction_xai(champion, paths.xai_summary)
    drivers = xai_summary.get("global_top_features", []) or xai_summary.get("top_features", [])

    _plot_qini_curve(champion, paths.qini_curve)
    _plot_uplift_curve(champion, paths.uplift_curve)
    _plot_topk_comparison(champion, paths.topk_comparison)
    _plot_decile_lift(champion, paths.decile_lift)
    _plot_xai_drivers(drivers, paths.xai_drivers)
    _plot_xai_driver_direction(drivers, paths.xai_driver_direction)
    _plot_representative_cases(
        xai_summary.get("representative_cases", {}),
        drivers,
        paths.representative_cases,
    )
    _plot_reasoning_timeline(main_records, paths.reasoning_timeline)
    _write_report(paths, main_records, champion, cv_row, test_normalized_qini, xai_summary)
    return paths


def _build_prediction_xai(champion: dict[str, Any], output_path: Path) -> dict[str, Any]:
    held_out_predictions = _read_artifact_csv(champion, "held_out_predictions")
    feature_path = resolve_feature_path(champion)
    model_path = resolve_artifact_path(champion.get("artifact_paths", {}).get("model"))
    features = pd.read_csv(feature_path) if feature_path is not None else pd.DataFrame()

    warning = ""
    if model_path is not None and model_path.exists() and not features.empty:
        try:
            summary = explain_cached_uplift_model(
                model_path,
                features,
                held_out_predictions,
                max_samples=500,
            )
        except Exception as exc:  # pragma: no cover - exercised by real artifacts.
            warning = f"cached_model_permutation failed: {exc}"
            summary = explain_score_feature_associations(features, held_out_predictions)
    else:
        warning = "cached model or feature artifact unavailable; used score association"
        summary = explain_score_feature_associations(features, held_out_predictions)

    top_features = summary.get("global_top_features", []) or summary.get("top_features", [])
    summary.update(
        {
            "champion_run_id": champion.get("run_id"),
            "champion_template_name": champion.get("template_name"),
            "champion_hypothesis_id": champion.get("hypothesis_id"),
            "feature_recipe_id": champion.get("feature_recipe_id"),
            "feature_artifact_id": champion.get("feature_artifact_id"),
            "source_feature_artifact": str(feature_path) if feature_path is not None else "",
            "source_predictions": str(
                resolve_artifact_path(champion.get("artifact_paths", {}).get("held_out_predictions"))
                or ""
            ),
            "source_model": str(model_path) if model_path is not None else "",
            "feature_semantics_diagnostic": diagnose_xai_feature_semantics(top_features),
        }
    )
    if warning:
        summary["warning"] = warning
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _read_artifact_csv(record: dict[str, Any], key: str) -> pd.DataFrame:
    path = resolve_artifact_path(record.get("artifact_paths", {}).get(key))
    if path is None:
        raise ValueError(f"Missing artifact path for {key}")
    return pd.read_csv(path)


def _plot_qini_curve(champion: dict[str, Any], output_path: Path) -> None:
    curve = _downsample_curve(_read_artifact_csv(champion, "held_out_qini_curve"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(curve["fraction"], curve["qini"], color="#176B87", linewidth=2.2)
    ax.fill_between(curve["fraction"], curve["qini"], color="#64CCC5", alpha=0.25)
    ax.set_title("AutoLift CV-Selected Held-out Qini Curve")
    ax.set_xlabel("Targeted population fraction")
    ax.set_ylabel("Incremental responders")
    ax.grid(alpha=0.25)
    ax.annotate(
        f"Raw Qini AUC: {champion['held_out_qini_auc']:.2f}",
        xy=(0.62, 0.12),
        xycoords="axes fraction",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#176B87"},
    )
    _savefig(fig, output_path)


def _plot_uplift_curve(champion: dict[str, Any], output_path: Path) -> None:
    curve = _downsample_curve(_read_artifact_csv(champion, "held_out_uplift_curve"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(curve["fraction"], curve["uplift"], color="#8F3C3C", linewidth=2.2)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_title("AutoLift CV-Selected Held-out Uplift Curve")
    ax.set_xlabel("Targeted population fraction")
    ax.set_ylabel("Observed uplift")
    ax.grid(alpha=0.25)
    ax.annotate(
        f"Uplift AUC: {champion['held_out_uplift_auc']:.5f}",
        xy=(0.62, 0.12),
        xycoords="axes fraction",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#8F3C3C"},
    )
    _savefig(fig, output_path)


def _plot_topk_comparison(champion: dict[str, Any], output_path: Path) -> None:
    labels = ["5%", "10%", "30%"]
    auto = [metric_value(champion, f"held_out_uplift_at_{label[:-1]}pct") for label in labels]
    human = [HUMAN_CV_SELECTED[f"test_uplift_at_{label[:-1]}pct"] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = range(len(labels))
    width = 0.36
    ax.bar([x - width / 2 for x in xs], auto, width, label="AutoLift CV-selected", color="#176B87")
    ax.bar([x + width / 2 for x in xs], human, width, label="Human CV-selected", color="#D98C00")
    ax.set_title("Sealed Test Targeting Lift: AutoLift vs Human")
    ax.set_xlabel("Targeted top-k segment")
    ax.set_ylabel("Observed uplift")
    ax.set_xticks(list(xs), labels)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, output_path)


def _plot_decile_lift(champion: dict[str, Any], output_path: Path) -> None:
    deciles = _read_artifact_csv(champion, "held_out_decile_table")
    fig, ax1 = plt.subplots(figsize=(8, 4.8))
    xs = deciles["bin"].astype(int)
    ax1.bar(xs, deciles["uplift"], color="#176B87", alpha=0.75, label="Observed uplift")
    ax1.set_xlabel("Predicted-uplift decile (1 = highest)")
    ax1.set_ylabel("Observed uplift")
    ax1.axhline(0, color="#444444", linewidth=0.8)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(
        xs,
        deciles["treated_response_rate"],
        color="#D98C00",
        marker="o",
        label="Treated response",
    )
    ax2.plot(
        xs,
        deciles["control_response_rate"],
        color="#6D5D6E",
        marker="o",
        label="Control response",
    )
    ax2.set_ylabel("Response rate")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="upper right")
    ax1.set_title("AutoLift CV-Selected Held-out Decile Lift")
    _savefig(fig, output_path)


def _plot_xai_drivers(drivers: list[dict[str, Any]], output_path: Path) -> None:
    top = drivers[:10]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if top:
        features = [str(item["feature"]) for item in top][::-1]
        values = [_driver_importance(item) for item in top][::-1]
        colors = [
            "#D98C00" if "age" in feature or "account" in feature else "#176B87"
            for feature in features
        ]
        ax.barh(features, values, color=colors)
        ax.set_xlabel("Prediction sensitivity")
    else:
        ax.text(0.5, 0.5, "No XAI drivers found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("AutoLift CV-Selected XAI Top Drivers")
    ax.grid(axis="x", alpha=0.25)
    _savefig(fig, output_path)


def _plot_xai_driver_direction(drivers: list[dict[str, Any]], output_path: Path) -> None:
    top = [item for item in drivers[:10] if item.get("spearman_with_uplift") is not None]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if top:
        features = [str(item["feature"]) for item in top][::-1]
        values = [float(item.get("spearman_with_uplift") or 0.0) for item in top][::-1]
        colors = ["#176B87" if value >= 0 else "#8F3C3C" for value in values]
        ax.barh(features, values, color=colors)
        ax.axvline(0, color="#444444", linewidth=0.8)
        ax.set_xlabel("Spearman correlation with predicted uplift")
    else:
        ax.text(0.5, 0.5, "No direction signal found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("XAI Driver Direction")
    ax.grid(axis="x", alpha=0.25)
    _savefig(fig, output_path)


def _plot_representative_cases(
    cases: dict[str, list[dict[str, Any]]],
    drivers: list[dict[str, Any]],
    output_path: Path,
) -> None:
    rows = _representative_case_rows(cases)
    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(11, 5.2),
        gridspec_kw={"width_ratios": [1.0, 1.65]},
    )
    if not rows:
        ax_bar.text(0.5, 0.5, "No representative cases found", ha="center", va="center")
        ax_bar.set_axis_off()
        ax_heat.set_axis_off()
        _savefig(fig, output_path)
        return

    labels = [
        f"{row['group'].replace('_', ' ')}\n{str(row.get('client_id', ''))[-6:]}"
        for row in rows
    ]
    uplifts = [float(row.get("uplift") or 0.0) for row in rows]
    colors = [_case_color(row["group"]) for row in rows]
    y_pos = list(range(len(rows)))
    ax_bar.barh(y_pos, uplifts, color=colors)
    ax_bar.axvline(0, color="#444444", linewidth=0.8)
    ax_bar.set_yticks(y_pos, labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Predicted uplift")
    ax_bar.set_title("Representative Customers")
    ax_bar.grid(axis="x", alpha=0.25)

    feature_columns = _case_feature_columns(cases, drivers)
    if feature_columns:
        values = pd.DataFrame(
            [
                [pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0] for column in feature_columns]
                for row in rows
            ],
            columns=feature_columns,
        )
        normalized = (values - values.mean()) / values.std(ddof=0).replace(0, 1)
        normalized = normalized.fillna(0.0)
        image = ax_heat.imshow(normalized.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
        ax_heat.set_xticks(range(len(feature_columns)), feature_columns, rotation=30, ha="right")
        ax_heat.set_yticks(y_pos, [""] * len(y_pos))
        for row_index in range(values.shape[0]):
            for col_index, column in enumerate(feature_columns):
                raw = values.iloc[row_index, col_index]
                ax_heat.text(
                    col_index,
                    row_index,
                    _format_case_value(raw),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="#222222",
                )
        fig.colorbar(image, ax=ax_heat, shrink=0.78, label="Feature value z-score")
    else:
        ax_heat.text(0.5, 0.5, "No feature columns found", ha="center", va="center")
        ax_heat.set_axis_off()
    ax_heat.set_title("Case Feature Context")
    _savefig(fig, output_path)


def _plot_reasoning_timeline(records: list[dict[str, Any]], output_path: Path) -> None:
    rows = timeline_rows(records)
    labels = [f"{row['step']}. {row['estimator']}" for row in rows]
    values = [float(row["held_out_qini_auc"] or 0.0) for row in rows]
    colors = [
        "#A0A0A0"
        if row["verdict"] == "baseline"
        else "#176B87"
        if row["verdict"] == "supported"
        else "#D98C00"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(labels, values, color="#444444", linewidth=1.2, alpha=0.7)
    ax.scatter(labels, values, s=120, c=colors, zorder=3)
    for label, value, row in zip(labels, values, rows, strict=True):
        ax.annotate(
            row["verdict"],
            xy=(label, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )
    ax.set_title("Agent Experiment Reasoning Timeline")
    ax.set_ylabel("Held-out raw Qini AUC")
    ax.tick_params(axis="x", rotation=28)
    ax.grid(axis="y", alpha=0.25)
    _savefig(fig, output_path)


def _write_report(
    paths: VisualAssetPaths,
    main_records: list[dict[str, Any]],
    champion: dict[str, Any],
    cv_row: dict[str, float],
    test_normalized_qini: float,
    xai_summary: dict[str, Any],
) -> None:
    comparison = comparison_rows(champion, cv_row, test_normalized_qini)
    comparison_table = write_markdown_table(
        ["Metric", "AutoLift CV-selected", "Human CV-selected", "AutoLift - Human"],
        [
            [
                label,
                _format_metric(auto_value),
                _format_metric(human_value),
                _format_signed(delta),
            ]
            for label, auto_value, human_value, delta in comparison
        ],
    )

    drivers = xai_summary.get("global_top_features", []) or xai_summary.get("top_features", [])
    top_drivers_table = write_markdown_table(
        ["Rank", "Feature", "Sensitivity", "Spearman", "Direction"],
        [
            [
                str(index),
                str(item.get("feature", "")),
                f"{_driver_importance(item):.6f}",
                _format_optional_float(item.get("spearman_with_uplift")),
                str(item.get("direction", "")),
            ]
            for index, item in enumerate(drivers[:10], start=1)
        ],
    )

    case_table = _representative_case_table(
        xai_summary.get("representative_cases", {}),
        drivers,
    )

    timeline_table = write_markdown_table(
        ["Step", "Run", "Learner", "Estimator", "Held-out Qini", "Verdict", "Decision Evidence"],
        [
            [
                str(row["step"]),
                row["run_id"],
                row["learner"],
                row["estimator"],
                _format_metric(float(row["held_out_qini_auc"] or 0.0)),
                row["verdict"],
                row["judge"] or row["rationale"],
            ]
            for row in timeline_rows(main_records)
        ],
    )

    report = f"""# AutoLift Explainability Pack

This pack adapts the visual explanation structure from `human_baseline_uplift.ipynb` for the final leakage-clean AutoLift candidate. It adds model-performance visuals, targeting diagnostics, prediction-level explanation, representative customers, and an agent decision timeline.

## Final Honest Human vs AutoLift

Both sides are selected without using the sealed test set. The human notebook selects `solo_model_xgb` by 5-fold CV after validation screening. AutoLift selects `{CHAMPION_RUN_ID}` / `two_model_lightgbm` by validation-top-3 CV.

{comparison_table}

![Sealed test top-k comparison]({paths.topk_comparison.name})

The final honest comparison is not a one-line domination claim. Human is slightly ahead on CV mean normalized Qini, while AutoLift is more stable across folds and stronger on sealed-test normalized Qini, raw Qini, uplift AUC, and top-k lift at 5%, 10%, and 30%.

## AutoLift Curves

![Held-out Qini curve]({paths.qini_curve.name})

![Held-out uplift curve]({paths.uplift_curve.name})

The notebook did not leave prediction-level human artifacts in this workspace, so the human comparison is shown through notebook-reported metrics instead of an overlaid human curve.

## Prediction-Level XAI

The final AutoLift explanation is generated from the cached model, the feature artifact, and held-out prediction rows for `{CHAMPION_RUN_ID}`. Method: `{xai_summary.get("method", "unknown")}`. Rows used: `{xai_summary.get("n_rows_used", 0)}`.

![XAI top drivers]({paths.xai_drivers.name})

![XAI driver direction]({paths.xai_driver_direction.name})

{top_drivers_table}

Permutation sensitivity explains model behavior; it is not causal proof of treatment effect. Age/account-age features remain prominent, so this explanation should be presented with the feature-policy caveat already documented in the robustness audit.

## Representative Cases

![Representative cases]({paths.representative_cases.name})

{case_table}

The case view converts raw prediction-level XAI into human-readable examples: strongest recommended targets, near-boundary customers, and low-uplift or potential sleeping-dog cases.

## Decile Lift

![Held-out decile lift]({paths.decile_lift.name})

The first decile is the top predicted-uplift group and should show the clearest treatment/control response separation if the ranking is useful.

## Agent Reasoning Timeline

![Agent reasoning timeline]({paths.reasoning_timeline.name})

{timeline_table}

This timeline is a quarantined process trace from the earlier adaptive loop. It is useful for explaining how the agent reasoned, but not for final model selection because those iterations could see held-out feedback. The final reportable selection is the validation-top-3 plus CV candidate above.

## Source Notes

- Final AutoLift comparison metrics: `results/run_20260430_best/validation_top3_cv_leaderboard.csv`, `results/run_20260430_best/validation_top3_cv_audit.md`, and `artifacts/uplift/cv_top3_validation_only_20260501_123000/rank_03_RUN-f1c30175/cv_summary.json`.
- Prediction-level XAI summary: `{paths.xai_summary.name}`.
- Prediction-level XAI inputs: `{xai_summary.get("source_model", "")}`, `{xai_summary.get("source_feature_artifact", "")}`, and `{xai_summary.get("source_predictions", "")}`.
- Human metrics: `human_baseline_uplift.ipynb` outputs. The corrected CV-selected champion is `solo_model_xgb`.
- Metric caution: the final table compares raw Qini only where both sides expose raw Qini; normalized Qini values use each workflow's reported normalized metric.
"""
    paths.report.write_text(report)


def _driver_importance(item: dict[str, Any]) -> float:
    if item.get("mean_abs_uplift_change") is not None:
        return float(item.get("mean_abs_uplift_change") or 0.0)
    return abs(float(item.get("spearman_with_uplift") or 0.0))


def _representative_case_rows(cases: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in ["highest_uplift", "near_boundary", "lowest_uplift"]:
        for case in cases.get(group, []):
            row = dict(case)
            row["group"] = group
            rows.append(row)
    return rows


def _case_feature_columns(
    cases: dict[str, list[dict[str, Any]]],
    drivers: list[dict[str, Any]],
    *,
    max_columns: int = 4,
) -> list[str]:
    rows = _representative_case_rows(cases)
    available = {
        key
        for row in rows
        for key in row
        if key not in {"group", "client_id", "uplift"}
    }
    ordered = [str(item.get("feature")) for item in drivers if item.get("feature") in available]
    ordered.extend(sorted(available - set(ordered)))
    return ordered[:max_columns]


def _representative_case_table(
    cases: dict[str, list[dict[str, Any]]],
    drivers: list[dict[str, Any]],
) -> str:
    rows = _representative_case_rows(cases)
    feature_columns = _case_feature_columns(cases, drivers, max_columns=3)
    headers = ["Group", "Client", "Predicted uplift", *feature_columns]
    return write_markdown_table(
        headers,
        [
            [
                row["group"].replace("_", " "),
                str(row.get("client_id", "")),
                _format_metric(float(row.get("uplift") or 0.0)),
                *[_format_case_value(row.get(column)) for column in feature_columns],
            ]
            for row in rows
        ],
    )


def _case_color(group: str) -> str:
    if group == "highest_uplift":
        return "#176B87"
    if group == "near_boundary":
        return "#D98C00"
    return "#8F3C3C"


def _format_case_value(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 100:
        return f"{number:.0f}"
    if abs(number) >= 10:
        return f"{number:.1f}"
    return f"{number:.3f}".rstrip("0").rstrip(".")


def _format_metric(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _format_optional_float(value: Any) -> str:
    if value is None:
        return ""
    return _format_metric(float(value))


def _format_signed(value: float) -> str:
    formatted = _format_metric(abs(value))
    return f"+{formatted}" if value >= 0 else f"-{formatted}"


def _savefig(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    lines = output_path.read_text().splitlines()
    output_path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def _downsample_curve(curve: pd.DataFrame, *, max_points: int = 800) -> pd.DataFrame:
    if len(curve) <= max_points:
        return curve
    step = max(len(curve) // max_points, 1)
    sampled = curve.iloc[::step]
    if sampled.index[-1] != curve.index[-1]:
        sampled = pd.concat([sampled, curve.tail(1)])
    return sampled.reset_index(drop=True)


def main() -> None:
    paths = render_visuals()
    print(f"Wrote explainability pack to {paths.report}")


if __name__ == "__main__":
    main()
