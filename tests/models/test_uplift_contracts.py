from pathlib import Path

import pytest

from src.models.uplift import (
    UpliftEvaluationPolicy,
    UpliftFeatureArtifact,
    UpliftFeatureRecipeSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftSubmissionArtifact,
    UpliftTableSchema,
    UpliftTrialSpec,
)


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _table_schema() -> UpliftTableSchema:
    return UpliftTableSchema(
        clients_table=str(FIXTURE_DIR / "clients.csv"),
        purchases_table=str(FIXTURE_DIR / "purchases.csv"),
        train_table=str(FIXTURE_DIR / "uplift_train.csv"),
        scoring_table=str(FIXTURE_DIR / "uplift_test.csv"),
        products_table=str(FIXTURE_DIR / "products.csv"),
    )


def test_uplift_project_contract_owns_treatment_and_scoring_semantics():
    contract = UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=_table_schema(),
        entity_key="client_id",
        treatment_column="treatment_flg",
        target_column="target",
    )

    assert contract.entity_key == "client_id"
    assert contract.treatment_column == "treatment_flg"
    assert contract.target_column == "target"
    assert contract.table_schema.scoring_table.endswith("uplift_test.csv")
    assert contract.submission_policy == "scoring_only"
    assert contract.evaluation_policy.primary_metric == "qini_auc"


def test_uplift_project_contract_rejects_overlapping_semantic_columns():
    with pytest.raises(ValueError, match="distinct"):
        UpliftProjectContract(
            task_name="bad",
            table_schema=_table_schema(),
            entity_key="client_id",
            treatment_column="target",
            target_column="target",
        )


def test_feature_recipe_id_is_stable_and_ignores_runtime_paths():
    recipe_a = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["rfm", "basket"],
        windows_days=[30, 90],
        builder_version="v1",
        artifact_path="/tmp/run-a/features.parquet",
    )
    recipe_b = UpliftFeatureRecipeSpec(
        source_tables=["purchases", "clients"],
        feature_groups=["basket", "rfm"],
        windows_days=[90, 30],
        builder_version="v1",
        artifact_path="/tmp/run-b/features.parquet",
    )

    assert recipe_a.feature_recipe_id == recipe_b.feature_recipe_id
    assert len(recipe_a.feature_recipe_id) == 12


def test_feature_artifact_id_includes_dataset_fingerprint_and_builder_version():
    recipe = UpliftFeatureRecipeSpec(
        source_tables=["clients"],
        feature_groups=["demographic"],
        windows_days=[],
        builder_version="v1",
    )

    artifact_a = recipe.compute_feature_artifact_id(dataset_fingerprint="abc")
    artifact_b = recipe.compute_feature_artifact_id(dataset_fingerprint="xyz")

    assert artifact_a != artifact_b
    assert len(artifact_a) == 12


def test_feature_recipe_id_includes_pinned_reference_date():
    recipe_a = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["rfm"],
        windows_days=[30],
        reference_date="2019-01-03 12:00:00",
    )
    recipe_b = UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=["rfm"],
        windows_days=[30],
        reference_date="2019-01-04 12:00:00",
    )

    assert recipe_a.reference_date == "2019-01-03T12:00:00"
    assert recipe_a.feature_recipe_id != recipe_b.feature_recipe_id


def test_feature_artifact_uses_declared_entity_key_not_literal_client_id():
    artifact = UpliftFeatureArtifact(
        feature_recipe_id="recipe123456",
        feature_artifact_id="artifact1234",
        dataset_fingerprint="dataset12345",
        builder_version="v1",
        artifact_path="features.csv",
        metadata_path="features.metadata.json",
        entity_key="customer_id",
        row_count=2,
        columns=["customer_id", "feature_a"],
        generated_columns=["feature_a"],
        source_tables=["clients"],
    )

    assert artifact.entity_key == "customer_id"


def test_feature_artifact_rejects_missing_declared_entity_key():
    with pytest.raises(ValueError, match="customer_id"):
        UpliftFeatureArtifact(
            feature_recipe_id="recipe123456",
            feature_artifact_id="artifact1234",
            dataset_fingerprint="dataset12345",
            builder_version="v1",
            artifact_path="features.csv",
            metadata_path="features.metadata.json",
            entity_key="customer_id",
            row_count=2,
            columns=["client_id", "feature_a"],
            generated_columns=["feature_a"],
            source_tables=["clients"],
        )


def test_split_contract_requires_valid_fractions():
    with pytest.raises(ValueError, match="sum to 1.0"):
        UpliftSplitContract(train_fraction=0.8, val_fraction=0.2, test_fraction=0.2)


def test_trial_spec_requires_registered_baseline_family_shape():
    spec = UpliftTrialSpec(
        hypothesis_id="baseline",
        template_name="two_model_sklearn",
        learner_family="two_model",
        base_estimator="logistic_regression",
        feature_recipe_id="abc123def456",
    )

    assert spec.learner_family == "two_model"
    assert spec.template_name == "two_model_sklearn"


def test_submission_artifact_schema_is_scoring_only():
    artifact = UpliftSubmissionArtifact(
        artifact_path="artifacts/submission.csv",
        champion_trial_id="trial-1",
        feature_recipe_id="recipe123456",
        feature_artifact_id="artifact1234",
        row_count=4,
        columns=["client_id", "uplift"],
    )

    assert artifact.columns == ["client_id", "uplift"]
    assert artifact.row_count == 4


def test_submission_artifact_rejects_target_or_treatment_columns():
    with pytest.raises(ValueError, match="submission columns"):
        UpliftSubmissionArtifact(
            artifact_path="artifacts/submission.csv",
            champion_trial_id="trial-1",
            feature_recipe_id="recipe123456",
            feature_artifact_id="artifact1234",
            row_count=4,
            columns=["client_id", "uplift", "target"],
        )


def test_evaluation_policy_cost_scenarios_default_to_sensitivity_cases():
    policy = UpliftEvaluationPolicy()

    assert policy.primary_metric == "qini_auc"
    assert policy.higher_is_better is True
    assert set(policy.cost_scenarios) == {"zero_cost", "low_cost", "medium_cost"}
