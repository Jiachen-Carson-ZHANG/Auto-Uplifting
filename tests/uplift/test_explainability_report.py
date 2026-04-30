import json
from pathlib import Path

import pandas as pd

from demos.uplift_explainability_report import (
    CHAMPION_RUN_ID,
    HUMAN_HELD_OUT_BEST,
    _downsample_curve,
    comparison_rows,
    extract_xai_drivers,
    find_champion,
    load_ledger,
    render_visuals,
    timeline_rows,
)


def test_comparison_rows_uses_human_notebook_best_held_out_metrics():
    champion = {
        "held_out_qini_auc": 331.7694,
        "held_out_uplift_auc": 0.06149,
        "held_out_uplift_at_k": {
            "top_5pct": 0.139969,
            "top_10pct": 0.099709,
            "top_20pct": 0.071709,
            "top_30pct": 0.062456,
        },
    }

    rows = comparison_rows(champion)

    assert rows[0] == (
        "Held-out raw Qini AUC",
        331.7694,
        HUMAN_HELD_OUT_BEST["held_out_qini_auc"],
        331.7694 - HUMAN_HELD_OUT_BEST["held_out_qini_auc"],
    )
    assert rows[2][0] == "Held-out uplift@5%"
    assert rows[2][3] < 0


def test_extract_xai_drivers_parses_final_report_literal(tmp_path):
    report = tmp_path / "final_report.md"
    report.write_text(
        "- Method: cached_model_permutation\n"
        "- Top drivers: [{'feature': 'age_clean', "
        "'mean_abs_uplift_change': 0.013934, "
        "'direction': 'higher_feature_higher_uplift'}]\n"
    )

    drivers = extract_xai_drivers(report)

    assert drivers == [
        {
            "feature": "age_clean",
            "mean_abs_uplift_change": 0.013934,
            "direction": "higher_feature_higher_uplift",
        }
    ]


def test_load_ledger_find_champion_and_timeline_rows(tmp_path):
    ledger = tmp_path / "uplift_ledger.jsonl"
    baseline = _record("RUN-base", "baseline", "manual_baseline")
    champion = _record(CHAMPION_RUN_ID, "supported", "agent-trial")
    ledger.write_text(json.dumps(baseline) + "\n" + json.dumps(champion) + "\n")

    records = load_ledger(ledger)

    assert find_champion(records)["run_id"] == CHAMPION_RUN_ID
    rows = timeline_rows(records)
    assert rows[0]["role"] == "Internal reference"
    assert rows[1]["role"] == "Agent trial"
    assert rows[1]["judge"] == "The model improved held-out Qini."


def test_downsample_curve_keeps_last_point():
    curve = pd.DataFrame({"fraction": range(2000), "qini": range(2000)})

    sampled = _downsample_curve(curve, max_points=100)

    assert len(sampled) <= 101
    assert sampled.iloc[-1]["fraction"] == 1999


def test_render_visuals_writes_markdown_and_svg_assets(tmp_path):
    run_dir = tmp_path / "run"
    output_dir = run_dir / "explainability"
    artifact_dir = tmp_path / "artifact"
    run_dir.mkdir()
    artifact_dir.mkdir()

    _write_champion_artifacts(artifact_dir)
    baseline = _record("RUN-base", "baseline", "manual_baseline")
    champion = _record(CHAMPION_RUN_ID, "supported", "agent-trial")
    champion["artifact_paths"] = {
        "held_out_qini_curve": str(artifact_dir / "held_out_qini_curve.csv"),
        "held_out_uplift_curve": str(artifact_dir / "held_out_uplift_curve.csv"),
        "held_out_decile_table": str(artifact_dir / "held_out_decile_table.csv"),
    }
    (run_dir / "uplift_ledger.jsonl").write_text(
        json.dumps(baseline) + "\n" + json.dumps(champion) + "\n"
    )
    (run_dir / "final_report.md").write_text(
        "- Top drivers: ["
        "{'feature': 'age_clean', 'mean_abs_uplift_change': 0.013934, "
        "'direction': 'higher_feature_higher_uplift'}, "
        "{'feature': 'points_received_total_30d', "
        "'mean_abs_uplift_change': 0.009384, "
        "'direction': 'higher_feature_lower_uplift'}]\n"
    )

    paths = render_visuals(run_dir=run_dir, output_dir=output_dir)

    assert paths.report.exists()
    assert paths.qini_curve.exists()
    assert paths.uplift_curve.exists()
    assert paths.topk_comparison.exists()
    assert paths.decile_lift.exists()
    assert paths.xai_drivers.exists()
    assert paths.reasoning_timeline.exists()

    report = paths.report.read_text()
    assert "AutoLift is a narrow held-out Qini leader" in report
    assert "agent-specific contribution" in report
    assert "autolift_xai_top_drivers.svg" in report


def _record(run_id: str, verdict: str, hypothesis_id: str) -> dict:
    return {
        "run_id": run_id,
        "hypothesis_id": hypothesis_id,
        "template_name": "class_transformation_lightgbm",
        "uplift_learner_family": "class_transformation",
        "base_estimator": "lightgbm",
        "held_out_qini_auc": 331.7694,
        "held_out_uplift_auc": 0.06149,
        "held_out_uplift_at_k": {
            "top_5pct": 0.139969,
            "top_10pct": 0.099709,
            "top_20pct": 0.071709,
            "top_30pct": 0.062456,
        },
        "verdict": verdict,
        "strategy_rationale": "Try LightGBM to capture finer treatment interactions.",
        "judge_narrative": "The model improved held-out Qini.",
        "artifact_paths": {},
    }


def _write_champion_artifacts(path: Path) -> None:
    pd.DataFrame(
        {
            "fraction": [0.1, 0.2, 0.3],
            "qini": [25.0, 50.0, 60.0],
        }
    ).to_csv(path / "held_out_qini_curve.csv", index=False)
    pd.DataFrame(
        {
            "fraction": [0.1, 0.2, 0.3],
            "uplift": [0.14, 0.11, 0.09],
        }
    ).to_csv(path / "held_out_uplift_curve.csv", index=False)
    pd.DataFrame(
        {
            "bin": [1, 2, 3],
            "n": [100, 100, 100],
            "treated_n": [50, 50, 50],
            "control_n": [50, 50, 50],
            "treated_response_rate": [0.55, 0.52, 0.50],
            "control_response_rate": [0.42, 0.45, 0.47],
            "uplift": [0.13, 0.07, 0.03],
            "avg_predicted_uplift": [0.12, 0.08, 0.04],
        }
    ).to_csv(path / "held_out_decile_table.csv", index=False)
