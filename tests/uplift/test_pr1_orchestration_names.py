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
from src.uplift.orchestrator import (
    AutoLiftOrchestrator,
    ManualBenchmarkAgent,
    ReportingAgent,
    RetryControllerAgent,
    _trial_from_planning_spec,
)
from src.uplift.planning_agents import ExperimentPlanningPhase, PlanningTrialSpec


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


def _append_record(
    ledger: UpliftLedger,
    *,
    hypothesis_id: str,
    qini_auc: float,
    params: dict | None = None,
):
    spec = UpliftTrialSpec(
        hypothesis_id=hypothesis_id,
        template_name="response_model_sklearn",
        learner_family="response_model",
        feature_recipe_id="recipe123",
        params=params or {"seed": qini_auc},
    )
    return ledger.append_result(
        trial_spec=spec,
        feature_artifact_id="artifact123",
        result_status="success",
        qini_auc=qini_auc,
        uplift_auc=qini_auc / 2,
        uplift_at_k={"top_50pct": qini_auc},
        policy_gain={"top_50pct_zero_cost": qini_auc},
        artifact_paths={"uplift_scores": "uplift_scores.csv"},
    )


def _write_scores(path: Path, uplift: list[float]) -> str:
    scores = pd.DataFrame(
        {
            "client_id": [f"c{i}" for i in range(8)],
            "uplift": uplift,
            "treatment_flg": [1, 1, 0, 0, 1, 0, 1, 0],
            "target": [1, 1, 1, 0, 0, 0, 1, 0],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(path, index=False)
    return str(path)


def test_retry_controller_stops_after_flat_recent_qini(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    _append_record(ledger, hypothesis_id="trial-1", qini_auc=0.100)
    _append_record(ledger, hypothesis_id="trial-2", qini_auc=0.104)
    _append_record(ledger, hypothesis_id="trial-3", qini_auc=0.106)

    decision = RetryControllerAgent(ledger, flat_window=3, flat_qini_threshold=0.01).run()

    assert decision.should_continue is False
    assert "flattened" in decision.reason


def test_retry_controller_treats_different_estimators_as_unique_configs(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    for spec_id, estimator in [("UT-xgb", "xgboost"), ("UT-lgbm", "lightgbm")]:
        spec = UpliftTrialSpec(
            spec_id=spec_id,
            hypothesis_id=spec_id,
            template_name=f"two_model_{estimator}",
            learner_family="two_model",
            base_estimator=estimator,
            feature_recipe_id="recipe123",
            params={},
        )
        ledger.append_result(
            trial_spec=spec,
            feature_artifact_id="artifact123",
            result_status="success",
            qini_auc=0.2,
            uplift_auc=0.1,
            artifact_paths={},
        )

    decision = RetryControllerAgent(
        ledger,
        max_trials=5,
        flat_window=3,
    ).run()

    assert decision.should_continue is True
    assert "Duplicate config" not in decision.reason


def test_reporting_agent_keeps_manual_benchmark_out_of_agent_champion(tmp_path):
    contract = _contract()
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    _append_record(ledger, hypothesis_id="manual_baseline", qini_auc=0.9)
    agent_record = _append_record(ledger, hypothesis_id="agent-trial", qini_auc=0.2)

    reporter = ReportingAgent(contract, ledger, output_path=tmp_path / "final_report.md")

    assert reporter.agent_champion().run_id == agent_record.run_id
    report_path = reporter.run()
    report = Path(report_path).read_text()
    assert "Agent Champion" in report
    assert agent_record.run_id in report
    assert "Manual Benchmark" in report
    assert "Decision" in report
    assert "Hypothesis Loop" in report
    assert "Representative Cases" in report
    assert "Why this answer is credible" in report


def test_reporting_agent_includes_all_trials_and_heldout_caution(tmp_path):
    contract = _contract()
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    manual_spec = UpliftTrialSpec(
        spec_id="manual_baseline",
        hypothesis_id="manual_baseline",
        template_name="two_model_sklearn",
        learner_family="two_model",
        base_estimator="logistic_regression",
        feature_recipe_id="recipe123",
    )
    agent_spec = UpliftTrialSpec(
        spec_id="UT-agent",
        hypothesis_id="UT-agent",
        template_name="class_transformation_gradient_boosting_sklearn",
        learner_family="class_transformation",
        base_estimator="gradient_boosting",
        feature_recipe_id="recipe123",
    )
    ledger.append_result(
        trial_spec=manual_spec,
        feature_artifact_id="artifact123",
        result_status="success",
        qini_auc=296.0,
        uplift_auc=0.044,
        held_out_qini_auc=309.0,
        held_out_uplift_auc=0.046,
        artifact_paths={
            "uplift_scores": _write_scores(
                tmp_path / "manual" / "uplift_scores.csv",
                [0.8, 0.7, 0.2, 0.1, -0.1, -0.2, 0.6, -0.3],
            ),
            "held_out_predictions": _write_scores(
                tmp_path / "manual" / "held_out_predictions.csv",
                [0.7, 0.6, 0.1, 0.0, -0.1, -0.2, 0.5, -0.3],
            ),
        },
    )
    ledger.append_result(
        trial_spec=agent_spec,
        feature_artifact_id="artifact123",
        result_status="success",
        qini_auc=305.0,
        uplift_auc=0.043,
        held_out_qini_auc=297.0,
        held_out_uplift_auc=0.040,
        artifact_paths={
            "uplift_scores": _write_scores(
                tmp_path / "agent" / "uplift_scores.csv",
                [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3],
            ),
            "held_out_predictions": _write_scores(
                tmp_path / "agent" / "held_out_predictions.csv",
                [0.6, 0.5, 0.4, 0.3, -0.1, -0.2, 0.2, -0.3],
            ),
        },
    )

    report = Path(
        ReportingAgent(contract, ledger, output_path=tmp_path / "final_report.md").run()
    ).read_text()

    assert "## All Trials" in report
    assert "| Run | Role | Learner | Estimator | Val Normalized Qini | Val Uplift AUC | Held-out Normalized Qini | Held-out Uplift AUC |" in report
    assert "Manual benchmark has stronger held-out normalized Qini" in report
    assert "validation normalized Qini AUC" in report
    assert "held-out safety reference" not in report
    assert "296.000000" not in report
    assert "309.000000" not in report


def test_reporting_agent_selects_agent_champion_by_validation_only(tmp_path):
    contract = _contract()
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    good_scores = [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3]
    reversed_scores = [-value for value in good_scores]

    for spec_id, val_scores, held_scores, qini_auc in [
        ("UT-validation-best", good_scores, reversed_scores, 400.0),
        ("UT-held-out-best", reversed_scores, good_scores, 300.0),
    ]:
        spec = UpliftTrialSpec(
            spec_id=spec_id,
            hypothesis_id=spec_id,
            template_name="two_model_gradient_boosting_sklearn",
            learner_family="two_model",
            base_estimator="gradient_boosting",
            feature_recipe_id="recipe123",
        )
        ledger.append_result(
            trial_spec=spec,
            feature_artifact_id="artifact123",
            result_status="success",
            qini_auc=qini_auc,
            uplift_auc=0.04,
            held_out_qini_auc=300.0,
            held_out_uplift_auc=0.04,
            artifact_paths={
                "uplift_scores": _write_scores(
                    tmp_path / spec_id / "uplift_scores.csv",
                    val_scores,
                ),
                "held_out_predictions": _write_scores(
                    tmp_path / spec_id / "held_out_predictions.csv",
                    held_scores,
                ),
            },
        )

    reporter = ReportingAgent(contract, ledger, output_path=tmp_path / "final_report.md")

    assert reporter.agent_champion().hypothesis_id == "UT-validation-best"


def test_reporting_agent_summarizes_repeated_seed_stability_groups(tmp_path):
    contract = _contract()
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    scores = [0.9, 0.8, 0.3, 0.2, -0.1, -0.2, 0.7, -0.3]
    for seed in [42, 123]:
        spec = UpliftTrialSpec(
            spec_id=f"UT-seed-{seed}",
            hypothesis_id=f"UT-seed-{seed}",
            template_name="two_model_gradient_boosting_sklearn",
            learner_family="two_model",
            base_estimator="gradient_boosting",
            feature_recipe_id="recipe123",
            params={"n_estimators": 200, "max_depth": 2},
            split_seed=seed,
        )
        ledger.append_result(
            trial_spec=spec,
            feature_artifact_id="artifact123",
            result_status="success",
            qini_auc=300.0,
            uplift_auc=0.04,
            held_out_qini_auc=300.0,
            held_out_uplift_auc=0.04,
            artifact_paths={
                "uplift_scores": _write_scores(
                    tmp_path / f"seed-{seed}" / "uplift_scores.csv",
                    scores,
                ),
                "held_out_predictions": _write_scores(
                    tmp_path / f"seed-{seed}" / "held_out_predictions.csv",
                    scores,
                ),
            },
        )

    report = Path(
        ReportingAgent(contract, ledger, output_path=tmp_path / "final_report.md").run()
    ).read_text()

    assert "## Seed Stability" in report
    assert "| two_model | gradient_boosting |" in report
    assert "| 42, 123 | 2 |" in report


def test_manual_benchmark_agent_runs_named_baseline(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)

    record = ManualBenchmarkAgent(
        contract,
        feature_artifact=feature_artifact,
        output_dir=tmp_path / "runs",
    ).run()

    assert record.hypothesis_id == "manual_baseline"
    assert record.status == "success"
    assert Path(record.artifact_paths["uplift_scores"]).exists()


def test_autolift_orchestrator_runs_planning_execution_retry_and_report(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    ledger = UpliftLedger(tmp_path / "runs" / "uplift_ledger.jsonl")
    planner = ExperimentPlanningPhase(
        ledger,
        UpliftHypothesisStore(tmp_path / "hypotheses.jsonl"),
        make_chat_llm("stub"),
    )

    result = AutoLiftOrchestrator(
        contract=contract,
        planner=planner,
        feature_artifacts_by_name={"rfm_baseline": feature_artifact},
        output_dir=tmp_path / "runs",
        llm=make_chat_llm("stub"),
        run_benchmark=True,
    ).run(max_iterations=1)

    assert result.report_path is not None
    assert Path(result.report_path).exists()
    report = Path(result.report_path).read_text()
    assert result.benchmark_record is not None
    assert result.trial_records
    assert result.evaluation_results
    assert "Retry Decision" in report
    assert "Policy Recommendation" in report
    assert "Hypothesis Loop" in report


def test_autolift_orchestrator_can_tune_before_final_trial(tmp_path):
    contract = _contract()
    feature_artifact = _feature_artifact(tmp_path)
    ledger = UpliftLedger(tmp_path / "runs" / "uplift_ledger.jsonl")
    planner = ExperimentPlanningPhase(
        ledger,
        UpliftHypothesisStore(tmp_path / "hypotheses.jsonl"),
        make_chat_llm("stub"),
    )

    result = AutoLiftOrchestrator(
        contract=contract,
        planner=planner,
        feature_artifacts_by_name={"rfm_baseline": feature_artifact},
        output_dir=tmp_path / "runs",
        llm=make_chat_llm("stub"),
        run_benchmark=False,
        enable_pre_run_tuning=True,
        tuning_split_seeds=(42,),
        tuning_max_param_sets=1,
    ).run(max_iterations=1)

    assert result.trial_records[0].status == "success"
    assert (tmp_path / "runs" / "pre_run_tuning").exists()
    assert "__tune_" in result.trial_records[0].hypothesis_id


def test_trial_from_planning_spec_strips_execution_owned_params(tmp_path):
    feature_artifact = _feature_artifact(tmp_path)
    planning_spec = PlanningTrialSpec(
        trial_id="UT-openai",
        hypothesis="OpenAI may include sklearn-owned params.",
        learner_family="response_model",
        base_estimator="logistic_regression",
        feature_recipe="rfm_baseline",
        params={"C": 1.0, "max_iter": 1000, "random_state": 123},
        split_seed=42,
        eval_cutoff=0.3,
        changes_from_previous="N/A",
        expected_improvement="N/A",
        model="response_model + logistic_regression",
        stop_criteria="N/A",
    )

    trial = _trial_from_planning_spec(planning_spec, feature_artifact)

    assert trial.params == {"C": 1.0, "max_iter": 1000}


def test_trial_from_planning_spec_drops_invalid_param_types_and_ranges(tmp_path):
    feature_artifact = _feature_artifact(tmp_path)
    planning_spec = PlanningTrialSpec(
        trial_id="UT-invalid-params",
        hypothesis="LLM may emit bad sklearn params.",
        learner_family="two_model",
        base_estimator="gradient_boosting",
        feature_recipe="rfm_baseline",
        params={
            "n_estimators": 0,
            "learning_rate": -0.1,
            "max_depth": "deep",
            "min_samples_leaf": 5,
            "subsample": 0.8,
            "random_state": 123,
        },
        split_seed=42,
        eval_cutoff=0.3,
        changes_from_previous="N/A",
        expected_improvement="N/A",
        model="two_model + gradient_boosting",
        stop_criteria="N/A",
    )

    trial = _trial_from_planning_spec(planning_spec, feature_artifact)

    assert trial.params == {"min_samples_leaf": 5, "subsample": 0.8}


def test_trial_from_planning_spec_allows_regularization_params_for_boosters(tmp_path):
    feature_artifact = _feature_artifact(tmp_path)
    planning_spec = PlanningTrialSpec(
        trial_id="UT-regularized-xgb",
        hypothesis="Regularized boosters should reduce unstable uplift rankings.",
        learner_family="two_model",
        base_estimator="xgboost",
        feature_recipe="rfm_baseline",
        params={
            "n_estimators": 400,
            "max_depth": 2,
            "learning_rate": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 10.0,
            "min_child_weight": 20,
            "random_state": 123,
        },
        split_seed=42,
        eval_cutoff=0.3,
        changes_from_previous="N/A",
        expected_improvement="N/A",
        model="two_model + xgboost",
        stop_criteria="N/A",
    )

    trial = _trial_from_planning_spec(planning_spec, feature_artifact)

    assert trial.params == {
        "n_estimators": 400,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_lambda": 10.0,
        "min_child_weight": 20,
    }
