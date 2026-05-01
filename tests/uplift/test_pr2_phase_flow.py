import json
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
from src.uplift.planning_agents import (
    FeatureSemanticsAgent,
    HypothesisDecision,
    RetrievedContext,
    UpliftStrategySelectionAgent,
    _available_autonomous_warmup_candidates,
    _safe_base_estimator,
    _safe_learner_family,
)
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
    assert spec.learner_family == "two_model"
    assert spec.base_estimator == "gradient_boosting"
    assert spec.params == {
        "n_estimators": 200,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_samples_leaf": 50,
        "subsample": 0.7,
    }
    assert hypothesis_store.query_by_status("proposed")


def test_feature_semantics_agent_selects_approved_recipe(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")

    def llm(system: str, user: str) -> str:
        return (
            '{"feature_recipe":"human_semantic_v1",'
            '"temporal_policy":"post_issue_history",'
            '"rationale":"Age dominates XAI, so test richer behavioral semantics.",'
            '"expected_signal":"Transaction features should enter top XAI drivers.",'
            '"model_family_hints":["class_transformation","two_model"],'
            '"leakage_controls":["audit temporal cutoff"],'
            '"xai_sanity_checks":["age should not be sole dominant feature"]}'
        )

    agent = FeatureSemanticsAgent(ledger, llm)
    decision = agent.run(
        context=RetrievedContext(
            similar_recipes=[],
            supported_hypotheses=[],
            refuted_hypotheses=[],
            best_learner_family="two_model",
            failed_runs=[],
            summary="age_clean dominates prior XAI",
        ),
        available_feature_recipes=["rfm_baseline", "human_semantic_v1"],
    )

    assert decision.feature_recipe == "human_semantic_v1"
    assert decision.temporal_policy == "post_issue_history"


def test_feature_semantics_agent_normalizes_pre_issue_history_synonym(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")

    def llm(system: str, user: str) -> str:
        return (
            '{"feature_recipe":"rfm_baseline",'
            '"temporal_policy":"pre_issue_history",'
            '"rationale":"Use only pre-issue history to avoid leakage.",'
            '"expected_signal":"Strict temporal controls should remain valid.",'
            '"model_family_hints":["two_model"],'
            '"leakage_controls":["pre issue cutoff"],'
            '"xai_sanity_checks":["no post treatment leakage"]}'
        )

    agent = FeatureSemanticsAgent(ledger, llm)
    decision = agent.run(
        context=RetrievedContext(
            similar_recipes=[],
            supported_hypotheses=[],
            refuted_hypotheses=[],
            best_learner_family="two_model",
            failed_runs=[],
            summary="Prefer strict temporal features.",
        ),
        available_feature_recipes=["rfm_baseline", "human_semantic_v1"],
    )

    assert decision.temporal_policy == "pre_issue_only"


def test_feature_semantics_agent_falls_back_to_approved_recipe(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")

    def llm(system: str, user: str) -> str:
        return (
            '{"feature_recipe":"invented_recipe",'
            '"temporal_policy":"post_issue_history",'
            '"rationale":"Try an unavailable recipe.",'
            '"expected_signal":"Unavailable signal.",'
            '"model_family_hints":["two_model"],'
            '"leakage_controls":["audit temporal cutoff"],'
            '"xai_sanity_checks":["age should not be sole dominant feature"]}'
        )

    agent = FeatureSemanticsAgent(ledger, llm)
    decision = agent.run(
        context=RetrievedContext(
            similar_recipes=[],
            supported_hypotheses=[],
            refuted_hypotheses=[],
            best_learner_family="two_model",
            failed_runs=[],
            summary="age_clean dominates prior XAI",
        ),
        available_feature_recipes=["rfm_baseline", "human_semantic_v1"],
    )

    assert decision.feature_recipe == "rfm_baseline"
    assert "unavailable recipe" in decision.rationale


def test_autonomous_strategy_minimal_warmup_ignores_manual_baseline_and_excludes_response(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    manual_spec = UpliftTrialSpec(
        spec_id="manual_baseline",
        hypothesis_id="manual_baseline",
        template_name="two_model_sklearn",
        learner_family="two_model",
        base_estimator="logistic_regression",
        feature_recipe_id="recipe123456",
    )
    ledger.append_result(
        trial_spec=manual_spec,
        feature_artifact_id="artifact123",
        result_status="success",
        qini_auc=0.25,
        uplift_auc=0.125,
        artifact_paths={},
    )
    agent = UpliftStrategySelectionAgent(ledger, make_chat_llm("stub"))

    strategy = agent.run(
        HypothesisDecision(
            action="propose",
            hypothesis="Improve uplift ranking.",
            evidence="manual baseline exists",
            confidence=0.5,
        ),
        RetrievedContext(
            similar_recipes=[],
            supported_hypotheses=[],
            refuted_hypotheses=[],
            best_learner_family="response_model",
            failed_runs=[],
            summary="manual baseline only",
        ),
    )

    assert strategy.learner_family == "two_model"
    assert strategy.base_estimator == "gradient_boosting"
    assert _safe_learner_family("response_model", "two_model") == "two_model"


def test_autonomous_strategy_skips_warmup_and_uses_llm_from_first_trial(tmp_path, monkeypatch):
    # Warmup is empty — agent goes straight to LLM-driven selection from trial 1.
    monkeypatch.setattr(
        "src.uplift.planning_agents._is_estimator_available",
        lambda estimator: estimator != "catboost",
    )
    assert _available_autonomous_warmup_candidates() == []

    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")

    def choose_xgboost(system: str, user: str) -> str:
        return (
            '{"learner_family":"two_model","base_estimator":"xgboost",'
            '"feature_recipe":"rfm_baseline","split_seed":42,'
            '"eval_cutoff":0.2,"rationale":"Benchmark shows LogReg ceiling; try stronger booster."}'
        )

    agent = UpliftStrategySelectionAgent(ledger, choose_xgboost)
    hypothesis = HypothesisDecision(
        action="propose",
        hypothesis="A stronger estimator will improve upon the LogReg benchmark.",
        evidence="manual benchmark qini=0.248",
        confidence=0.6,
    )
    context = RetrievedContext(
        similar_recipes=[],
        supported_hypotheses=[],
        refuted_hypotheses=[],
        best_learner_family="two_model",
        failed_runs=[],
        summary="benchmark only, no agent trials yet",
    )

    first = agent.run(hypothesis, context)

    assert first.learner_family == "two_model"
    assert first.base_estimator == "xgboost"


def test_autonomous_strategy_replaces_duplicate_agent_choice_with_unused_pair(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.uplift.planning_agents._is_estimator_available",
        lambda estimator: estimator != "catboost",
    )
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    for spec_id, family, estimator, qini in [
        ("UT-gb", "two_model", "gradient_boosting", 0.30),
        ("UT-ct", "class_transformation", "gradient_boosting", 0.31),
    ]:
        ledger.append_result(
            trial_spec=UpliftTrialSpec(
                spec_id=spec_id,
                hypothesis_id=spec_id,
                template_name=f"{family}_{estimator}_sklearn",
                learner_family=family,
                base_estimator=estimator,
                feature_recipe_id="recipe123456",
            ),
            feature_artifact_id="artifact123",
            result_status="success",
            qini_auc=qini,
            uplift_auc=qini / 2,
            artifact_paths={},
        )

    def choose_duplicate(system: str, user: str) -> str:
        return (
            '{"learner_family":"class_transformation",'
            '"base_estimator":"gradient_boosting",'
            '"feature_recipe":"rfm_baseline","split_seed":42,'
            '"eval_cutoff":0.3,"rationale":"Repeat current champion."}'
        )

    agent = UpliftStrategySelectionAgent(ledger, choose_duplicate)
    strategy = agent.run(
        HypothesisDecision(
            action="propose",
            hypothesis="Explore the next informative uplift model.",
            evidence="Warmup completed.",
            confidence=0.7,
        ),
        RetrievedContext(
            similar_recipes=[],
            supported_hypotheses=[],
            refuted_hypotheses=[],
            best_learner_family="class_transformation",
            failed_runs=[],
            summary="Both gradient-boosting warmup pairs already ran.",
        ),
    )

    assert (strategy.learner_family, strategy.base_estimator) == (
        "class_transformation",
        "logistic_regression",
    )
    assert "already ran" in strategy.rationale


def test_autonomous_strategy_treats_manual_logistic_reference_as_used_pair(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        "src.uplift.planning_agents._is_estimator_available",
        lambda estimator: estimator != "catboost",
    )
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    for spec_id, hypothesis_id, family, estimator, qini in [
        ("manual_baseline", "manual_baseline", "two_model", "logistic_regression", 0.25),
        ("UT-gb", "UT-gb", "two_model", "gradient_boosting", 0.30),
        ("UT-ct", "UT-ct", "class_transformation", "gradient_boosting", 0.31),
    ]:
        ledger.append_result(
            trial_spec=UpliftTrialSpec(
                spec_id=spec_id,
                hypothesis_id=hypothesis_id,
                template_name="two_model_sklearn"
                if estimator == "logistic_regression"
                else f"{family}_{estimator}_sklearn",
                learner_family=family,
                base_estimator=estimator,
                feature_recipe_id="recipe123456",
            ),
            feature_artifact_id="artifact123",
            result_status="success",
            qini_auc=qini,
            uplift_auc=qini / 2,
            artifact_paths={},
        )

    def choose_manual_duplicate(system: str, user: str) -> str:
        payload = json.loads(user)
        assert ["two_model", "logistic_regression"] in payload["used_model_pairs"]
        return (
            '{"learner_family":"two_model",'
            '"base_estimator":"logistic_regression",'
            '"feature_recipe":"rfm_baseline","split_seed":42,'
            '"eval_cutoff":0.3,"rationale":"Retest fixed logistic reference."}'
        )

    agent = UpliftStrategySelectionAgent(ledger, choose_manual_duplicate)
    strategy = agent.run(
        HypothesisDecision(
            action="propose",
            hypothesis="Explore the next informative uplift model.",
            evidence="Warmup completed.",
            confidence=0.7,
        ),
        RetrievedContext(
            similar_recipes=[],
            supported_hypotheses=[],
            refuted_hypotheses=[],
            best_learner_family="class_transformation",
            failed_runs=[],
            summary="Manual reference and both warmup pairs already ran.",
        ),
    )

    assert (strategy.learner_family, strategy.base_estimator) != (
        "two_model",
        "logistic_regression",
    )
    assert (strategy.learner_family, strategy.base_estimator) == (
        "class_transformation",
        "logistic_regression",
    )


def test_safe_base_estimator_rejects_unavailable_optional_estimators(monkeypatch):
    monkeypatch.setattr(
        "src.uplift.planning_agents._is_estimator_available",
        lambda estimator: estimator != "catboost",
    )

    assert _safe_base_estimator("xgboost", "gradient_boosting") == "xgboost"
    assert _safe_base_estimator("catboost", "gradient_boosting") == "gradient_boosting"


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
    assert result["policy"]["recommended_threshold"] in {5, 10, 20, 30}
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
