import json
from pathlib import Path

import pandas as pd

from demos.uplift_explainability_report import (
    CHAMPION_RUN_ID,
    HUMAN_CV_SELECTED,
    _downsample_curve,
    comparison_rows,
    extract_xai_drivers,
    find_champion,
    load_cv_row,
    load_ledger,
    load_test_normalized_qini,
    render_visuals,
    resolve_feature_path,
    timeline_rows,
)


def test_comparison_rows_uses_honest_human_cv_selected_metrics():
    champion = {
        "held_out_qini_auc": 309.987113,
        "held_out_uplift_auc": 0.058746,
        "held_out_uplift_at_k": {
            "top_5pct": 0.183569,
            "top_10pct": 0.111772,
            "top_30pct": 0.058085,
        },
    }
    cv_row = {
        "cv_normalized_qini_auc": 0.396226,
        "cv_normalized_qini_auc_std": 0.060313,
        "cv_raw_qini_auc": 392.456261,
        "cv_uplift_auc": 0.066282,
        "cv_uplift_at_5pct": 0.143441,
        "cv_uplift_at_10pct": 0.115764,
        "cv_uplift_at_30pct": 0.064839,
    }

    rows = comparison_rows(champion, cv_row, 0.248455)

    assert rows[0] == (
        "CV mean normalized Qini",
        0.396226,
        HUMAN_CV_SELECTED["cv_normalized_qini_auc"],
        0.396226 - HUMAN_CV_SELECTED["cv_normalized_qini_auc"],
    )
    assert rows[5][0] == "Test uplift@5%"
    assert rows[5][3] > 0


def test_load_cv_row_reads_final_candidate_leaderboard(tmp_path):
    path = tmp_path / "cv.csv"
    pd.DataFrame(
        {
            "run_id": [CHAMPION_RUN_ID],
            "cv_mean_normalized_qini_auc": [0.396226],
            "cv_std_normalized_qini_auc": [0.060313],
            "cv_mean_raw_qini_auc": [392.456261],
            "cv_mean_uplift_auc": [0.066282],
            "cv_mean_uplift_at_5pct": [0.143441],
            "cv_mean_uplift_at_10pct": [0.115764],
            "cv_mean_uplift_at_30pct": [0.064839],
        }
    ).to_csv(path, index=False)

    row = load_cv_row(path)

    assert row["cv_normalized_qini_auc"] == 0.396226
    assert row["cv_uplift_at_30pct"] == 0.064839


def test_load_test_normalized_qini_reads_tuning_summary(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(
        json.dumps(
            {
                "records": [
                    {"run_id": "RUN-other", "held_out_normalized_qini": 0.12},
                    {"run_id": CHAMPION_RUN_ID, "held_out_normalized_qini": 0.248455},
                ]
            }
        )
    )

    assert load_test_normalized_qini(path) == 0.248455


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
    artifact_dir = run_dir / "artifact"
    features_dir = run_dir / "features"
    run_dir.mkdir()
    artifact_dir.mkdir()
    features_dir.mkdir()

    _write_champion_artifacts(artifact_dir)
    _write_feature_artifact(features_dir / "uplift_features_train_artifact123.csv")
    baseline = _record("RUN-base", "baseline", "manual_baseline")
    champion = _record(CHAMPION_RUN_ID, "supported", "agent-trial")
    champion["feature_artifact_id"] = "artifact123"
    champion["artifact_paths"] = {
        "held_out_qini_curve": str(artifact_dir / "held_out_qini_curve.csv"),
        "held_out_uplift_curve": str(artifact_dir / "held_out_uplift_curve.csv"),
        "held_out_decile_table": str(artifact_dir / "held_out_decile_table.csv"),
        "held_out_predictions": str(artifact_dir / "held_out_predictions.csv"),
    }
    (run_dir / "uplift_ledger.jsonl").write_text(
        json.dumps(baseline) + "\n" + json.dumps(champion) + "\n"
    )
    (run_dir / "agentic_tuning_validation_only_ledger.jsonl").write_text(
        json.dumps(champion) + "\n"
    )
    (run_dir / "agentic_tuning_validation_only_execution_summary.json").write_text(
        json.dumps(
            {
                "records": [
                    {
                        "run_id": CHAMPION_RUN_ID,
                        "held_out_normalized_qini": 0.248455,
                    }
                ]
            }
        )
    )
    pd.DataFrame(
        {
            "run_id": [CHAMPION_RUN_ID],
            "cv_mean_normalized_qini_auc": [0.396226],
            "cv_std_normalized_qini_auc": [0.060313],
            "cv_mean_raw_qini_auc": [392.456261],
            "cv_mean_uplift_auc": [0.066282],
            "cv_mean_uplift_at_5pct": [0.143441],
            "cv_mean_uplift_at_10pct": [0.115764],
            "cv_mean_uplift_at_30pct": [0.064839],
        }
    ).to_csv(run_dir / "validation_top3_cv_leaderboard.csv", index=False)
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
    assert paths.xai_driver_direction.exists()
    assert paths.representative_cases.exists()
    assert paths.xai_summary.exists()
    assert paths.reasoning_timeline.exists()

    report = paths.report.read_text()
    assert "Prediction-Level XAI" in report
    assert "Representative Cases" in report
    assert "quarantined process trace" in report
    assert "autolift_xai_top_drivers.svg" in report
    assert "autolift_representative_cases.svg" in report
    xai_summary = json.loads(paths.xai_summary.read_text())
    assert xai_summary["method"] == "score_feature_association"
    assert resolve_feature_path(champion) == features_dir / "uplift_features_train_artifact123.csv"


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
            "top_5pct": 0.183569,
            "top_10pct": 0.111772,
            "top_30pct": 0.058085,
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
    pd.DataFrame(
        {
            "client_id": ["a", "b", "c", "d", "e"],
            "target": [1, 0, 1, 0, 1],
            "treatment_flg": [1, 0, 1, 0, 1],
            "uplift": [0.22, 0.08, 0.03, -0.01, -0.05],
        }
    ).to_csv(path / "held_out_predictions.csv", index=False)


def _write_feature_artifact(path: Path) -> None:
    pd.DataFrame(
        {
            "client_id": ["a", "b", "c", "d", "e"],
            "age_clean": [71, 42, 55, 31, 24],
            "days_to_first_redeem": [12, 50, 91, 200, 500],
            "points_received_total_30d": [2.0, 7.0, 5.0, 12.0, 18.0],
        }
    ).to_csv(path, index=False)
