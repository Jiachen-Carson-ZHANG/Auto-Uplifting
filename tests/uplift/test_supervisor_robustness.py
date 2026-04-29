from pathlib import Path

import pandas as pd
import pytest

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftExperimentWaveSpec,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftResultCard,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
    UpliftWaveResult,
)
from src.uplift.loop import run_uplift_trials
from src.uplift.splitting import UpliftSplitFrames
from src.uplift.supervisor import (
    evaluate_policy_threshold_stability,
    evaluate_ranking_stability,
    evaluate_robustness,
    evaluate_uplift_stop_policy,
    rank_correlation,
    top_k_overlap,
    validate_wave_spec,
)
from src.uplift.templates import UpliftTemplateOutput


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _contract() -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table=str(FIXTURE_DIR / "clients.csv"),
            purchases_table=str(FIXTURE_DIR / "purchases.csv"),
            train_table=str(FIXTURE_DIR / "uplift_train.csv"),
            scoring_table=str(FIXTURE_DIR / "uplift_test.csv"),
            products_table=str(FIXTURE_DIR / "products.csv"),
        ),
        split_contract=UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.5,
            test_fraction=0.0,
            min_rows_per_partition=1,
            random_seed=7,
        ),
    )


def _prediction_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": [f"c{i:03d}" for i in range(len(scores))],
            "uplift": scores,
        }
    )


def _record(
    run_id: str,
    *,
    predictions_path: str | None = None,
    qini_auc: float = 0.12,
    policy_gain: dict[str, float] | None = None,
    split_seed: int = 7,
) -> UpliftExperimentRecord:
    artifact_paths = {}
    if predictions_path is not None:
        artifact_paths["predictions"] = predictions_path
    return UpliftExperimentRecord(
        run_id=run_id,
        hypothesis_id="UH-robust",
        feature_recipe_id="recipe-a",
        feature_artifact_id="artifact-a",
        template_name="random_baseline",
        uplift_learner_family="random",
        base_estimator="none",
        params_hash=f"params-{run_id}",
        split_seed=split_seed,
        status="success",
        qini_auc=qini_auc,
        uplift_auc=qini_auc,
        policy_gain=policy_gain or {"top_50pct_zero_cost": 1.0},
        artifact_paths=artifact_paths,
    )


def _wave_result() -> UpliftWaveResult:
    return UpliftWaveResult(
        wave_id="UW-robust",
        hypothesis_id="UH-robust",
        action_type="ranking_stability_check",
        status="completed",
        trial_ids=["RUN-a", "RUN-b"],
        failed_trial_ids=[],
        champion_run_id="RUN-a",
        artifact_paths={"ledger": "runs/uplift_ledger.jsonl"},
    )


def _stability_wave(feature_recipe_id: str) -> UpliftExperimentWaveSpec:
    return UpliftExperimentWaveSpec(
        wave_id="UW-stability",
        hypothesis_id="UH-robust",
        action_type="ranking_stability_check",
        rationale="Repeat the champion-like spec across split seeds.",
        trial_specs=[
            UpliftTrialSpec(
                spec_id=f"UT-stability-{seed}",
                hypothesis_id="UH-robust",
                template_name="random_baseline",
                learner_family="random",
                base_estimator="none",
                feature_recipe_id=feature_recipe_id,
                split_seed=seed,
                primary_metric="qini_auc",
            )
            for seed in [7, 11, 13]
        ],
        expected_signal="Ranking is stable across seed repeats.",
        success_criterion="Rank correlation and top-k overlap remain high.",
        abort_on_first_failure=True,
        required_feature_recipe_ids=[feature_recipe_id],
        created_by="manual",
    )


def test_rank_correlation_and_top_k_overlap_detect_ranking_stability():
    left = _prediction_frame([0.9, 0.8, 0.7, 0.1])
    similar = _prediction_frame([0.91, 0.79, 0.69, 0.2])
    reversed_frame = _prediction_frame([0.1, 0.7, 0.8, 0.9])

    assert rank_correlation(left, similar) == pytest.approx(1.0)
    assert rank_correlation(left, reversed_frame) < -0.7
    assert top_k_overlap(left, similar, k=0.5) == 1.0
    assert top_k_overlap(left, reversed_frame, k=0.5) == 0.0


def test_evaluate_ranking_stability_aggregates_repeat_prediction_artifacts(tmp_path):
    first_path = tmp_path / "predictions-a.csv"
    second_path = tmp_path / "predictions-b.csv"
    _prediction_frame([0.9, 0.8, 0.7, 0.1]).to_csv(first_path, index=False)
    _prediction_frame([0.91, 0.79, 0.69, 0.2]).to_csv(second_path, index=False)

    report = evaluate_ranking_stability(
        [
            _record("RUN-a", predictions_path=str(first_path)),
            _record("RUN-b", predictions_path=str(second_path)),
        ],
        top_k=0.5,
        min_rank_correlation=0.95,
        min_top_k_overlap=0.75,
    )

    assert report.stable is True
    assert report.pair_count == 1
    assert report.min_rank_correlation == pytest.approx(1.0)
    assert report.min_top_k_overlap == 1.0
    assert report.limitations == []


def test_evaluate_policy_threshold_stability_detects_unstable_cost_cutoffs():
    stable = evaluate_policy_threshold_stability(
        [
            _record(
                "RUN-a",
                policy_gain={"top_50pct_zero_cost": 1.0, "top_75pct_zero_cost": 0.8},
            ),
            _record(
                "RUN-b",
                policy_gain={"top_50pct_zero_cost": 0.9, "top_75pct_zero_cost": 0.7},
            ),
        ]
    )
    unstable = evaluate_policy_threshold_stability(
        [
            _record(
                "RUN-a",
                policy_gain={"top_50pct_zero_cost": 1.0, "top_75pct_zero_cost": 0.8},
            ),
            _record(
                "RUN-b",
                policy_gain={"top_50pct_zero_cost": 0.6, "top_75pct_zero_cost": 1.2},
            ),
        ]
    )

    assert stable.stable is True
    assert stable.stable_cutoff_by_scenario == {"zero_cost": "top_50pct"}
    assert unstable.stable is False
    assert unstable.unstable_scenarios == ["zero_cost"]


def test_unstable_robustness_forces_stop_policy_to_continue_inconclusive(tmp_path):
    first_path = tmp_path / "predictions-a.csv"
    second_path = tmp_path / "predictions-b.csv"
    _prediction_frame([0.9, 0.8, 0.7, 0.1]).to_csv(first_path, index=False)
    _prediction_frame([0.1, 0.7, 0.8, 0.9]).to_csv(second_path, index=False)
    records = [
        _record("RUN-a", predictions_path=str(first_path), qini_auc=0.12),
        _record("RUN-b", predictions_path=str(second_path), qini_auc=0.11),
    ]
    robustness = evaluate_robustness(
        records,
        top_k=0.5,
        min_rank_correlation=0.8,
        min_top_k_overlap=0.75,
    )

    decision = evaluate_uplift_stop_policy(
        _wave_result(),
        records=records,
        valid_next_actions=["ranking_stability_check"],
        robustness_report=robustness,
    )

    assert robustness.stable is False
    assert decision.stop_reason == "low_information_gain"
    assert decision.hypothesis_status == "inconclusive"
    assert decision.should_stop is False
    assert decision.next_action == "ranking_stability_check"
    assert decision.evidence_summary["robustness"]["stable"] is False


def test_ranking_stability_wave_allows_seed_repeats_for_one_feature_recipe():
    artifact = UpliftFeatureArtifact(
        feature_recipe_id="recipe-a",
        feature_artifact_id="artifact-a",
        dataset_fingerprint="dataset-a",
        builder_version="v1",
        artifact_path="features/a.csv",
        metadata_path="features/a.metadata.json",
        row_count=8,
        columns=["client_id", "feature"],
        generated_columns=["feature"],
        source_tables=["clients"],
        feature_groups=["demographic"],
    )
    wave = _stability_wave("recipe-a")

    validate_wave_spec(wave, feature_artifacts={"recipe-a": artifact})

    assert [spec.split_seed for spec in wave.trial_specs] == [7, 11, 13]
    assert wave.required_feature_recipe_ids == ["recipe-a"]


def test_run_uplift_trials_respects_each_trial_split_seed(tmp_path, monkeypatch):
    features_path = tmp_path / "features.csv"
    pd.DataFrame({"client_id": ["c001", "c002"], "feature": [1.0, 2.0]}).to_csv(
        features_path,
        index=False,
    )
    artifact = UpliftFeatureArtifact(
        feature_recipe_id="recipe-a",
        feature_artifact_id="artifact-a",
        dataset_fingerprint="dataset-a",
        builder_version="v1",
        artifact_path=str(features_path),
        metadata_path=str(tmp_path / "features.metadata.json"),
        row_count=2,
        columns=["client_id", "feature"],
        generated_columns=["feature"],
        source_tables=["clients"],
        feature_groups=["demographic"],
    )
    seen_split_seeds: list[int] = []

    def fake_split(labeled_df, contract):
        seen_split_seeds.append(contract.split_contract.random_seed)
        frame = pd.DataFrame(
            {
                "client_id": ["c001", "c002"],
                "feature": [1.0, 2.0],
                "target": [1, 0],
                "treatment_flg": [1, 0],
            }
        )
        return UpliftSplitFrames(
            train=frame,
            validation=frame,
            test=frame.iloc[[]].copy(),
            strategy="random",
            warnings=[],
        )

    def fake_template(spec, **kwargs):
        predictions = pd.DataFrame(
            {
                "client_id": ["c001", "c002"],
                "target": [1, 0],
                "treatment_flg": [1, 0],
                "uplift": [0.9, 0.1],
            }
        )
        return UpliftTemplateOutput(
            result_card=UpliftResultCard(
                trial_spec_id=spec.spec_id,
                status="success",
                qini_auc=0.1,
                uplift_auc=0.1,
                uplift_at_k={"top_50pct": 0.1},
                policy_gain={"top_50pct_zero_cost": 1.0},
            ),
            predictions=predictions,
            decile_table=pd.DataFrame({"bin": [1]}),
            qini_curve=pd.DataFrame({"fraction": [1.0], "qini": [0.1]}),
            uplift_curve=pd.DataFrame({"fraction": [1.0], "uplift": [0.1]}),
        )

    monkeypatch.setattr("src.uplift.loop.split_labeled_uplift_frame", fake_split)
    monkeypatch.setattr("src.uplift.loop.run_uplift_template", fake_template)

    result = run_uplift_trials(
        _contract(),
        feature_artifact=artifact,
        trial_specs=_stability_wave("recipe-a").trial_specs,
        output_dir=tmp_path / "runs",
    )

    assert seen_split_seeds == [7, 11, 13]
    assert [record.split_seed for record in result.records] == [7, 11, 13]
