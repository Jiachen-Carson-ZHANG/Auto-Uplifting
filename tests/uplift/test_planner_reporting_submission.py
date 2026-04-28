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
from src.uplift.ledger import UpliftLedger
from src.uplift.planner import UpliftAdvisoryPlanner
from src.uplift.reporting import (
    generate_submission_artifact,
    generate_uplift_report,
    validate_submission_artifact,
)
from src.uplift.templates import fit_uplift_model


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


def _recipe() -> UpliftFeatureRecipeSpec:
    return UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["demographic", "rfm", "basket", "points"],
        windows_days=[30],
        builder_version="v1",
    )


def _feature_artifacts(tmp_path):
    contract = _contract()
    recipe = _recipe()
    train_artifact = build_feature_table(
        contract,
        recipe=recipe,
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    scoring_artifact = build_feature_table(
        contract,
        recipe=recipe,
        output_dir=tmp_path / "features",
        cohort="scoring",
        chunksize=2,
    )
    return contract, train_artifact, scoring_artifact


def test_advisory_planner_returns_registered_trial_without_mutating_contract_fields(tmp_path):
    contract, train_artifact, _ = _feature_artifacts(tmp_path)
    original_target = contract.target_column
    original_treatment = contract.treatment_column

    def fake_llm(prompt: str) -> str:
        assert "Allowed templates" in prompt
        return """
        {
          "hypothesis_id": "llm-h1",
          "template_name": "not_registered",
          "learner_family": "two_model",
          "base_estimator": "logistic_regression",
          "target_column": "bad_target",
          "treatment_column": "bad_treatment"
        }
        """

    planner = UpliftAdvisoryPlanner(llm_call=fake_llm)
    trial = planner.propose_next_trial(
        contract,
        feature_artifact=train_artifact,
        prior_records=[],
    )

    assert trial.template_name in {
        "random_baseline",
        "response_model_sklearn",
        "two_model_sklearn",
        "solo_model_sklearn",
    }
    assert trial.feature_recipe_id == train_artifact.feature_recipe_id
    assert contract.target_column == original_target
    assert contract.treatment_column == original_treatment


def test_generate_uplift_report_is_grounded_in_ledger_records(tmp_path):
    contract, train_artifact, _ = _feature_artifacts(tmp_path)
    ledger = UpliftLedger(tmp_path / "ledger.jsonl")
    spec = UpliftTrialSpec(
        hypothesis_id="baseline-response",
        template_name="response_model_sklearn",
        learner_family="response_model",
        feature_recipe_id=train_artifact.feature_recipe_id,
    )
    ledger.append_result(
        trial_spec=spec,
        feature_artifact_id=train_artifact.feature_artifact_id,
        result_status="success",
        qini_auc=0.12,
        uplift_auc=0.08,
        uplift_at_k={"top_50pct": 0.2},
        policy_gain={"top_50pct_zero_cost": 1.0},
    )

    report_path = generate_uplift_report(
        contract,
        records=ledger.load(),
        output_path=tmp_path / "report.md",
    )
    report = Path(report_path).read_text()

    assert "Internal evaluation uses labeled uplift_train.csv splits" in report
    assert "uplift_test.csv is scoring/submission only" in report
    assert "response_model_sklearn" in report
    assert "0.12" in report
    # The report must distinguish selection metrics, held-out metrics, and
    # the submission-fit model so consumers don't conflate them.
    assert "Validation (selection)" in report
    assert "Held-out test" in report
    assert "retrained on the full labeled" in report


def test_generate_submission_artifact_writes_scoring_only_schema(tmp_path):
    contract, train_artifact, scoring_artifact = _feature_artifacts(tmp_path)
    train_features = pd.read_csv(train_artifact.artifact_path)
    labels = pd.read_csv(contract.table_schema.train_table)
    train_frame = train_features.merge(labels, on=contract.entity_key, how="inner")
    model = fit_uplift_model(
        train_frame,
        learner_family="two_model",
        entity_key=contract.entity_key,
        treatment_col=contract.treatment_column,
        target_col=contract.target_column,
        random_seed=13,
    )
    trial = UpliftTrialSpec(
        hypothesis_id="champion",
        template_name="two_model_sklearn",
        learner_family="two_model",
        feature_recipe_id=train_artifact.feature_recipe_id,
    )

    artifact = generate_submission_artifact(
        contract,
        model=model,
        scoring_feature_artifact=scoring_artifact,
        champion_trial=trial,
        output_path=tmp_path / "submission.csv",
    )
    validate_submission_artifact(contract, artifact)
    submission = pd.read_csv(artifact.artifact_path)

    assert submission.columns.tolist() == ["client_id", "uplift"]
    assert len(submission) == 4
    assert submission["client_id"].tolist() == ["s001", "s002", "s003", "s004"]
    assert "target" not in submission.columns
    assert "treatment_flg" not in submission.columns
