from pathlib import Path

import pandas as pd

from src.models.uplift import (
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
)
from src.uplift.features import build_feature_table
from src.uplift.hypotheses import UpliftHypothesisStore
from src.uplift.ledger import UpliftLedger
from src.uplift.llm_client import make_chat_llm
from src.uplift.loop import run_uplift_trials
from src.uplift.evaluation_agents import run_evaluation_phase
from src.uplift.planning_agents import ExperimentPlanningPhase
from src.uplift.policy import build_policy_summary


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


def _feature_artifact(tmp_path):
    return build_feature_table(
        _contract(),
        recipe=UpliftFeatureRecipeSpec(
            source_tables=["clients", "purchases"],
            feature_groups=["demographic", "rfm", "basket", "points"],
            windows_days=[30],
            builder_version="v1",
        ),
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )


def _scores_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "client_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
            "uplift": [0.9, 0.7, 0.3, 0.2, -0.1, -0.2],
            "treatment_flg": [1, 0, 1, 0, 1, 0],
            "target": [1, 0, 1, 0, 0, 0],
        }
    )


def test_pr2_planning_phase_returns_readable_trial_spec(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    hypothesis_store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    planner = ExperimentPlanningPhase(ledger, hypothesis_store, make_chat_llm("stub"))

    spec = planner.run()

    assert spec.trial_id.startswith("UT-")
    assert spec.feature_recipe == "rfm_baseline"
    assert spec.learner_family in {
        "response_model",
        "solo_model",
        "two_model",
        "class_transformation",
    }
    assert spec.base_estimator == "logistic_regression"
    assert spec.params == {"C": 1.0, "max_iter": 1000}
    assert hypothesis_store.query_by_status("proposed")


def test_pr2_evaluation_phase_uses_predictions_and_policy_language(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")

    result = run_evaluation_phase(
        trial_meta={
            "spec_id": "UT-pr2",
            "learner_family": "two_model",
            "hypothesis_text": "RFM features improve treatment ranking.",
        },
        scores_df=_scores_frame(),
        ledger=ledger,
        llm=make_chat_llm("stub"),
        model_dir=None,
        features_df=pd.DataFrame(
            {
                "client_id": ["c1", "c2", "c3", "c4", "c5", "c6"],
                "recency_days": [2, 4, 8, 10, 20, 25],
                "purchase_sum": [90, 70, 30, 20, 5, 1],
            }
        ),
    )

    assert set(result) == {"judge", "xai", "policy"}
    assert result["judge"]["trial_id"] == "UT-pr2"
    assert "computed_metrics" in result["judge"]
    assert result["xai"]["skipped"] is False
    assert result["xai"]["method"] == "score_feature_association"
    assert result["xai"]["global_top_features"]
    assert result["policy"]["recommended_threshold"] == 10
    assert result["policy"]["policy_data"]["targeting_results"][0]["threshold_pct"] == 5


def test_pr2_policy_summary_keeps_business_facing_fields():
    summary = build_policy_summary(
        _scores_frame(),
        coupon_cost=0.5,
        revenue_per_conversion=5.0,
        budget=2.0,
    )

    assert summary["targeting_results"]
    assert summary["budget_result"]["max_coupons_affordable"] == 4
    assert "persuadables" in summary["segment_summary"]
    assert summary["decile_table"]


def test_execution_loop_writes_pr2_uplift_scores_alias(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    trial = UpliftTrialSpec(
        hypothesis_id="pr2-compatible-output",
        template_name="response_model_sklearn",
        learner_family="response_model",
        feature_recipe_id=feature_artifact.feature_recipe_id,
    )

    result = run_uplift_trials(
        contract,
        feature_artifact=feature_artifact,
        trial_specs=[trial],
        output_dir=tmp_path / "runs",
    )

    record = result.records[0]
    scores_path = Path(record.artifact_paths["uplift_scores"])
    scores = pd.read_csv(scores_path)

    assert scores_path.name == "uplift_scores.csv"
    assert scores.columns.tolist() == ["client_id", "uplift", "treatment_flg", "target"]


def test_evaluation_phase_uses_cached_model_for_xai_explanation(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    trial = UpliftTrialSpec(
        hypothesis_id="cached-model-xai",
        template_name="response_model_sklearn",
        learner_family="response_model",
        feature_recipe_id=feature_artifact.feature_recipe_id,
    )
    result = run_uplift_trials(
        contract,
        feature_artifact=feature_artifact,
        trial_specs=[trial],
        output_dir=tmp_path / "runs",
    )
    record = result.records[0]

    evaluation = run_evaluation_phase(
        trial_meta={
            "spec_id": trial.spec_id,
            "learner_family": trial.learner_family,
            "hypothesis_text": "RFM features improve treatment ranking.",
        },
        scores_df=pd.read_csv(record.artifact_paths["uplift_scores"]),
        ledger=UpliftLedger(tmp_path / "runs" / "uplift_ledger.jsonl"),
        llm=make_chat_llm("stub"),
        model_dir=Path(record.artifact_paths["model"]).parent,
        features_df=pd.read_csv(feature_artifact.artifact_path),
    )

    assert evaluation["xai"]["skipped"] is False
    assert evaluation["xai"]["method"] == "cached_model_permutation"
    assert evaluation["xai"]["global_top_features"]
    assert evaluation["xai"]["representative_cases"]
