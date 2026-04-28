import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


FIXTURE_DIR = Path("tests/fixtures/uplift")


def test_uplift_build_features_demo_outputs_valid_feature_artifact(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "demos/uplift_build_features.py",
            "--data-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(tmp_path),
            "--cohort",
            "train",
            "--chunksize",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SUMMARY_JSON=")
    )
    summary = json.loads(summary_line.removeprefix("SUMMARY_JSON="))
    artifact_path = Path(summary["artifact_path"])
    metadata_path = Path(summary["metadata_path"])
    feature_df = pd.read_csv(artifact_path)

    assert summary["row_count"] == 8
    assert summary["cohort"] == "train"
    assert artifact_path.exists()
    assert metadata_path.exists()
    assert feature_df["client_id"].is_unique
    assert "target" not in feature_df.columns
    assert "treatment_flg" not in feature_df.columns
