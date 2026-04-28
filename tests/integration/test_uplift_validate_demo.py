"""Integration guard for the Uplift V1 dataset validation demo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_uplift_validation_demo_emits_summary_json():
    repo_root = Path(__file__).resolve().parents[2]
    demo_path = repo_root / "demos" / "uplift_validate_dataset.py"
    data_dir = repo_root / "tests" / "fixtures" / "uplift"

    result = subprocess.run(
        [
            sys.executable,
            str(demo_path),
            "--data-dir",
            str(data_dir),
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=True,
    )

    summary_line = next(
        line for line in result.stdout.splitlines() if line.startswith("SUMMARY_JSON=")
    )
    summary = json.loads(summary_line.split("=", 1)[1])

    assert summary["valid"] is True
    assert summary["table_rows"]["train"] == 8
    assert summary["table_rows"]["scoring"] == 4
    assert summary["scoring_is_unlabeled"] is True
    assert summary["balance"]["treatment_counts"] == {"0": 4, "1": 4}
