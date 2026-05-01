import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import pandas as pd

from demos.uplift_run_autonomous_pipeline import (
    _champion_model_path,
    _successful_agent_champion,
)


def test_autonomous_pipeline_demo_runs_with_stub_llm(tmp_path):
    output_dir = tmp_path / "autonomous"
    command = [
        sys.executable,
        "demos/uplift_run_autonomous_pipeline.py",
        "--provider",
        "stub",
        "--data-dir",
        "tests/fixtures/uplift",
        "--output-dir",
        str(output_dir),
        "--max-iterations",
        "1",
        "--run-benchmark",
        "--small-fixture-mode",
    ]

    result = subprocess.run(command, check=True, capture_output=True, text=True)

    summary_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SUMMARY_JSON=")
    )
    summary = json.loads(summary_line.split("=", 1)[1])

    assert summary["provider"] == "stub"
    assert summary["n_agent_trials"] == 1
    assert summary["n_ledger_records"] >= 2
    assert summary["retry_max_trials"] == 5
    assert summary["semantic_features"] is True
    assert "human_semantic_v1" in summary["feature_recipes"]
    assert "human_semantic_v1" in summary["train_feature_artifacts"]
    assert summary["report_path"].endswith("final_report.md")
    assert summary["submission_path"].endswith("uplift_submission.csv")

    log_path = output_dir / "pipeline.log"
    assert log_path.exists()
    log_text = log_path.read_text()
    assert "Stage 1/7: configuration" in log_text
    assert "[features] build start cohort=train" in log_text
    assert "[features] purchase chunk" in log_text
    assert "FINAL SUMMARY TABLE" in log_text
    assert "val_norm_qini" in log_text
    assert "holdout_norm_qini" not in log_text
    assert "SUMMARY_JSON=" in log_text


def test_autonomous_pipeline_champion_selection_excludes_manual_benchmark():
    manual = SimpleNamespace(
        status="success",
        hypothesis_id="manual_baseline",
        qini_auc=999.0,
    )
    agent = SimpleNamespace(
        status="success",
        hypothesis_id="UT-agent",
        qini_auc=1.0,
    )

    assert _successful_agent_champion([manual, agent]) is agent


def test_autonomous_pipeline_champion_selection_uses_validation_only(tmp_path):
    def write_scores(name: str, uplift: list[float]) -> str:
        path = tmp_path / name
        pd.DataFrame(
            {
                "client_id": [f"c{i}" for i in range(8)],
                "uplift": uplift,
                "treatment_flg": [1, 1, 0, 0, 1, 0, 1, 0],
                "target": [1, 1, 1, 0, 0, 0, 1, 0],
            }
        ).to_csv(path, index=False)
        return str(path)

    good = [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3]
    validation_best = SimpleNamespace(
        status="success",
        hypothesis_id="UT-validation-best",
        qini_auc=400.0,
        artifact_paths={
            "uplift_scores": write_scores("validation_best_val.csv", good),
            "held_out_predictions": write_scores(
                "validation_best_held.csv",
                [-value for value in good],
            ),
        },
    )
    held_out_best = SimpleNamespace(
        status="success",
        hypothesis_id="UT-held-out-best",
        qini_auc=300.0,
        artifact_paths={
            "uplift_scores": write_scores("held_out_best_val.csv", [-value for value in good]),
            "held_out_predictions": write_scores("held_out_best_held.csv", good),
        },
    )

    assert _successful_agent_champion([validation_best, held_out_best]) is validation_best


def test_autonomous_pipeline_requires_champion_model_artifact(tmp_path):
    missing_key = SimpleNamespace(
        run_id="RUN-no-model-key",
        hypothesis_id="UT-no-model-key",
        artifact_paths={},
    )
    missing_file = SimpleNamespace(
        run_id="RUN-no-model-file",
        hypothesis_id="UT-no-model-file",
        artifact_paths={"model": str(tmp_path / "missing.pkl")},
    )

    with pytest.raises(RuntimeError, match="missing a saved model artifact"):
        _champion_model_path(missing_key)

    with pytest.raises(RuntimeError, match="model artifact does not exist"):
        _champion_model_path(missing_file)
