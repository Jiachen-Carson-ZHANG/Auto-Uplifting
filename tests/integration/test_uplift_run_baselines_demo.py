import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


FIXTURE_DIR = Path("tests/fixtures/uplift")


def test_uplift_run_baselines_demo_writes_ledger_report_and_submission(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "demos/uplift_run_baselines.py",
            "--data-dir",
            str(FIXTURE_DIR),
            "--output-dir",
            str(tmp_path),
            "--chunksize",
            "2",
            "--small-fixture-mode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SUMMARY_JSON=")
    )
    summary = json.loads(summary_line.removeprefix("SUMMARY_JSON="))
    ledger_path = Path(summary["ledger_path"])
    report_path = Path(summary["report_path"])
    submission_path = Path(summary["submission_path"])
    submission = pd.read_csv(submission_path)

    assert summary["n_records"] == 4
    assert ledger_path.exists()
    assert report_path.exists()
    assert submission_path.exists()
    assert submission.columns.tolist() == ["client_id", "uplift"]
    assert len(submission) == 4
    assert "Internal evaluation uses labeled uplift_train.csv splits" in report_path.read_text()
