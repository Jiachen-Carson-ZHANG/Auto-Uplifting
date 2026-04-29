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
)
from src.uplift.planning_agents import ExperimentPlanningPhase


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


def test_retry_controller_stops_after_flat_recent_qini(tmp_path):
    ledger = UpliftLedger(tmp_path / "uplift_ledger.jsonl")
    _append_record(ledger, hypothesis_id="trial-1", qini_auc=0.100)
    _append_record(ledger, hypothesis_id="trial-2", qini_auc=0.104)
    _append_record(ledger, hypothesis_id="trial-3", qini_auc=0.106)

    decision = RetryControllerAgent(ledger, flat_window=3, flat_qini_threshold=0.01).run()

    assert decision.should_continue is False
    assert "flattened" in decision.reason


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
