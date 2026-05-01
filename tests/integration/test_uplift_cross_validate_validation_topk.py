from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from demos.uplift_cross_validate_champion import _build_contract
from demos.uplift_cross_validate_validation_topk import (
    load_validation_candidates,
    run_validation_topk_cross_validation,
)
from src.models.uplift import UpliftFeatureRecipeSpec
from src.uplift.features import build_feature_table


FIXTURE_DIR = Path("tests/fixtures/uplift")


def test_load_validation_candidates_ranks_by_validation_not_post_selection_metric(
    tmp_path,
):
    good_validation = _write_scores(
        tmp_path / "good_validation.csv",
        [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3],
    )
    bad_validation = _write_scores(
        tmp_path / "bad_validation.csv",
        [-0.9, -0.8, -0.3, -0.2, 0.1, 0.2, -0.7, 0.3],
    )
    ledger = tmp_path / "uplift_ledger.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps(
                    _record(
                        run_id="RUN-validation-best",
                        hypothesis_id="UT-validation-best",
                        uplift_scores=good_validation,
                        qini_auc=10.0,
                        post_selection_qini=1.0,
                    )
                ),
                json.dumps(
                    _record(
                        run_id="RUN-post-selection-best",
                        hypothesis_id="UT-post-selection-best",
                        uplift_scores=bad_validation,
                        qini_auc=9.0,
                        post_selection_qini=9999.0,
                    )
                ),
            ]
        ),
        encoding="utf-8",
    )
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "trial_specs": [
                    {
                        "hypothesis_id": "UT-validation-best",
                        "params": {"max_iter": 250},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    candidates = load_validation_candidates([ledger], plan_paths=[plan], top_k=1)

    assert candidates[0].run_id == "RUN-validation-best"
    assert candidates[0].params == {"max_iter": 250}


def test_validation_topk_cv_excludes_internal_test_partition(tmp_path):
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
    uplift_scores = _write_scores(
        tmp_path / "validation_scores.csv",
        [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3],
    )
    ledger = tmp_path / "uplift_ledger.jsonl"
    record = _record(
        run_id="RUN-validation-best",
        hypothesis_id="UT-validation-best",
        uplift_scores=uplift_scores,
        qini_auc=10.0,
        post_selection_qini=1.0,
    )
    record["feature_recipe_id"] = feature_artifact.feature_recipe_id
    record["feature_artifact_id"] = feature_artifact.feature_artifact_id
    ledger.write_text(json.dumps(record), encoding="utf-8")

    result = run_validation_topk_cross_validation(
        contract,
        ledger_paths=[ledger],
        plan_paths=[],
        feature_metadata_globs=[str(tmp_path / "features" / "*.metadata.json")],
        output_dir=tmp_path / "topk_cv",
        top_k=1,
        n_folds=2,
        seed=123,
    )

    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    split = summary["split_summary"]
    assert result.selected_candidate_run_id == "RUN-validation-best"
    assert split["sealed_internal_test_rows"] > 0
    assert split["cv_pool_rows"] == (
        split["internal_train_rows"] + split["internal_validation_rows"]
    )
    assert split["full_labeled_rows"] == (
        split["cv_pool_rows"] + split["sealed_internal_test_rows"]
    )
    assert "held_out_qini_auc" not in Path(result.summary_path).read_text(
        encoding="utf-8"
    )
    assert Path(result.leaderboard_path).exists()


def _write_scores(path: Path, uplift: list[float]) -> str:
    pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(8)],
            "uplift": uplift,
            "treatment_flg": [1, 1, 0, 0, 1, 0, 1, 0],
            "target": [1, 1, 1, 0, 0, 0, 1, 0],
        }
    ).to_csv(path, index=False)
    return str(path)


def _record(
    *,
    run_id: str,
    hypothesis_id: str,
    uplift_scores: str,
    qini_auc: float,
    post_selection_qini: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "hypothesis_id": hypothesis_id,
        "feature_recipe_id": "recipe123",
        "feature_artifact_id": "artifact123",
        "template_name": "two_model_sklearn",
        "uplift_learner_family": "two_model",
        "base_estimator": "logistic_regression",
        "params_hash": "params123",
        "split_seed": 42,
        "status": "success",
        "qini_auc": qini_auc,
        "uplift_auc": 0.1,
        "held_out_qini_auc": post_selection_qini,
        "artifact_paths": {
            "uplift_scores": uplift_scores,
        },
    }
