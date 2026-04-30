from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "results" / "run_20260430_best"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_DIR / "explainability"
CHAMPION_RUN_ID = "RUN-c5e6e86f"


HUMAN_HELD_OUT_BEST = {
    "label": "Human notebook best held-out row",
    "model": "class_transform_gbm / tuned_class_transform_gbm",
    "source": "human_baseline_uplift.ipynb",
    "held_out_qini_auc": 328.3899,
    "held_out_uplift_auc": 0.0631,
    "held_out_uplift_at_5pct": 0.1637,
    "held_out_uplift_at_10pct": 0.1289,
    "held_out_uplift_at_20pct": 0.0764,
    "held_out_uplift_at_30pct": 0.0627,
}

HUMAN_VALIDATION_CHAMPION = {
    "model": "tuned_solo_model_xgb",
    "validation_qini_auc": 367.03263,
    "held_out_qini_auc": 299.13056,
    "held_out_uplift_auc": 0.05828,
    "held_out_uplift_at_10pct": 0.12779,
    "held_out_uplift_at_30pct": 0.05242,
}


@dataclass(frozen=True)
class VisualAssetPaths:
    qini_curve: Path
    uplift_curve: Path
    topk_comparison: Path
    decile_lift: Path
    xai_drivers: Path
    reasoning_timeline: Path
    report: Path


def load_ledger(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_champion(records: list[dict[str, Any]], run_id: str = CHAMPION_RUN_ID) -> dict[str, Any]:
    for record in records:
        if record.get("run_id") == run_id:
            return record
    raise ValueError(f"Champion run {run_id} not found")


def resolve_artifact_path(raw_path: str | None, repo_root: Path = REPO_ROOT) -> Path | None:
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


def metric_value(record: dict[str, Any], key: str) -> float:
    if key.startswith("held_out_uplift_at_"):
        pct = key.removeprefix("held_out_uplift_at_").removesuffix("pct")
        return float(record["held_out_uplift_at_k"][f"top_{pct}pct"])
    return float(record[key])


def comparison_rows(champion: dict[str, Any]) -> list[tuple[str, float, float, float]]:
    metrics = [
        ("Held-out raw Qini AUC", "held_out_qini_auc"),
        ("Held-out uplift AUC", "held_out_uplift_auc"),
        ("Held-out uplift@5%", "held_out_uplift_at_5pct"),
        ("Held-out uplift@10%", "held_out_uplift_at_10pct"),
        ("Held-out uplift@20%", "held_out_uplift_at_20pct"),
        ("Held-out uplift@30%", "held_out_uplift_at_30pct"),
    ]
    rows = []
    for label, key in metrics:
        auto_value = metric_value(champion, key)
        human_value = float(HUMAN_HELD_OUT_BEST[key])
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
    records = load_ledger(run_dir / "uplift_ledger.jsonl")
    champion = find_champion(records)
    drivers = extract_xai_drivers(run_dir / "final_report.md")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = VisualAssetPaths(
        qini_curve=output_dir / "autolift_heldout_qini_curve.svg",
        uplift_curve=output_dir / "autolift_heldout_uplift_curve.svg",
        topk_comparison=output_dir / "human_vs_autolift_topk.svg",
        decile_lift=output_dir / "autolift_decile_lift.svg",
        xai_drivers=output_dir / "autolift_xai_top_drivers.svg",
        reasoning_timeline=output_dir / "agent_reasoning_timeline.svg",
        report=output_dir / "EXPLAINABILITY_REPORT.md",
    )

    _plot_qini_curve(champion, paths.qini_curve)
    _plot_uplift_curve(champion, paths.uplift_curve)
    _plot_topk_comparison(champion, paths.topk_comparison)
    _plot_decile_lift(champion, paths.decile_lift)
    _plot_xai_drivers(drivers, paths.xai_drivers)
    _plot_reasoning_timeline(records, paths.reasoning_timeline)
    _write_report(paths, records, champion, drivers)
    return paths


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
    ax.set_title("AutoLift Champion Held-out Qini Curve")
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
    ax.set_title("AutoLift Champion Held-out Uplift Curve")
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
    labels = ["5%", "10%", "20%", "30%"]
    auto = [metric_value(champion, f"held_out_uplift_at_{label[:-1]}pct") for label in labels]
    human = [
        HUMAN_HELD_OUT_BEST[f"held_out_uplift_at_{label[:-1]}pct"]
        for label in labels
    ]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    xs = range(len(labels))
    width = 0.36
    ax.bar([x - width / 2 for x in xs], auto, width, label="AutoLift", color="#176B87")
    ax.bar([x + width / 2 for x in xs], human, width, label="Human notebook", color="#D98C00")
    ax.set_title("Held-out Targeting Lift: AutoLift vs Human Notebook")
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
    ax1.set_title("AutoLift Held-out Decile Lift")
    _savefig(fig, output_path)


def _plot_xai_drivers(drivers: list[dict[str, Any]], output_path: Path) -> None:
    top = drivers[:10]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if top:
        features = [item["feature"] for item in top][::-1]
        values = [float(item["mean_abs_uplift_change"]) for item in top][::-1]
        colors = ["#D98C00" if "age" in feature else "#176B87" for feature in features]
        ax.barh(features, values, color=colors)
        ax.set_xlabel("Mean absolute uplift change")
    else:
        ax.text(0.5, 0.5, "No XAI drivers found", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title("AutoLift Champion XAI Top Drivers")
    ax.grid(axis="x", alpha=0.25)
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
    records: list[dict[str, Any]],
    champion: dict[str, Any],
    drivers: list[dict[str, Any]],
) -> None:
    comparison = comparison_rows(champion)
    comparison_table = write_markdown_table(
        ["Metric", "AutoLift", "Human Notebook", "Delta"],
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

    top_drivers_table = write_markdown_table(
        ["Rank", "Feature", "Mean Abs Uplift Change", "Direction"],
        [
            [
                str(index),
                str(item.get("feature", "")),
                f"{float(item.get('mean_abs_uplift_change', 0.0)):.6f}",
                str(item.get("direction", "")),
            ]
            for index, item in enumerate(drivers[:10], start=1)
        ],
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
            for row in timeline_rows(records)
        ],
    )

    report = f"""# AutoLift Explainability Pack

This pack adapts the visual explanation structure from `human_baseline_uplift.ipynb` for the AutoLift run. It adds model-performance visuals, targeting diagnostics, feature-level explanation, and an agent decision timeline.

## Human vs AutoLift

{comparison_table}

![Held-out top-k comparison]({paths.topk_comparison.name})

AutoLift is a narrow held-out Qini leader. The human notebook remains stronger on held-out uplift AUC and top-k targeting lift, so the correct story is competitive automation with transparent reasoning rather than broad metric dominance.

## AutoLift Curves

![Held-out Qini curve]({paths.qini_curve.name})

![Held-out uplift curve]({paths.uplift_curve.name})

The notebook did not leave prediction-level human artifacts in this workspace, so the human comparison is shown through notebook-reported metrics instead of an overlaid human curve.

## Decile Lift

![Held-out decile lift]({paths.decile_lift.name})

The first decile is the top predicted-uplift group and should show the clearest treatment/control response separation if the ranking is useful.

## Feature Explanation

![XAI top drivers]({paths.xai_drivers.name})

{top_drivers_table}

Age-related features remain prominent, so this explanation should be presented with the feature-policy caveat already documented in the robustness audit.

## Agent Reasoning Timeline

![Agent reasoning timeline]({paths.reasoning_timeline.name})

{timeline_table}

This is the agent-specific contribution: each trial carries a hypothesis, feature rationale, expected signal, held-out metrics, judge verdict, XAI summary, and policy recommendation.

## Source Notes

- AutoLift artifacts: `results/run_20260430_best/uplift_ledger.jsonl` and saved champion CSVs under `artifacts/uplift/run_20260430_221602/runs/UT-9fb6c6/`.
- Human metrics: `human_baseline_uplift.ipynb` outputs. The best held-out row is `class_transform_gbm` / `tuned_class_transform_gbm`; the validation-selected champion is `tuned_solo_model_xgb`.
- Metric caution: AutoLift report-table Qini values are normalized, while the human notebook reports raw Qini. This pack compares raw Qini only where both sides expose raw Qini.
"""
    paths.report.write_text(report)


def _format_metric(value: float) -> str:
    if abs(value) >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


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
