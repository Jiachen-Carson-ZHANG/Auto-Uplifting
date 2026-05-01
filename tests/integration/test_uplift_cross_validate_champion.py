from __future__ import annotations

import json
from pathlib import Path

from demos.uplift_cross_validate_champion import (
    _build_contract,
    run_cross_validation,
)
from src.models.uplift import UpliftFeatureRecipeSpec
from src.uplift.features import build_feature_table


FIXTURE_DIR = Path("tests/fixtures/uplift")


def test_cross_validate_champion_writes_fold_and_summary_artifacts(tmp_path):
    contract = _build_contract(FIXTURE_DIR)
    feature_artifact = build_feature_table(
        contract,
        recipe=UpliftFeatureRecipeSpec(
            source_tables=["clients", "purchases"],
            feature_groups=["demographic", "rfm", "basket", "points"],
            windows_days=[30],
            temporal_policy="safe_history_until_reference",
            builder_version="v1",
        ),
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )

    result = run_cross_validation(
        contract,
        feature_artifact=feature_artifact,
        output_dir=tmp_path / "cv",
        n_folds=2,
        seed=123,
    )

    summary = json.loads(Path(result.summary_path).read_text())
    assert summary["champion_run_id"] == "RUN-c5e6e86f"
    assert summary["selection_policy"] == "fixed champion only; no model selection inside CV"
    assert summary["n_folds"] == 2
    assert len(summary["folds"]) == 2
    assert Path(result.metrics_path).exists()
    assert (tmp_path / "cv" / "CV_SUMMARY.md").exists()
    assert (tmp_path / "cv" / "fold_01" / "result_card.json").exists()
