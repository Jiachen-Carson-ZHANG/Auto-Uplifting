from pathlib import Path
import shutil

import pandas as pd
import pytest

from src.models.uplift import (
    UpliftExperimentWaveSpec,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
)
from src.uplift.recipe_registry import UpliftFeatureRecipeRegistry
from src.uplift.supervisor import UpliftResearchLoop, validate_wave_spec


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _copy_fixture_dir(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    shutil.copytree(FIXTURE_DIR, data_dir)
    return data_dir


def _contract(data_dir: Path = FIXTURE_DIR) -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table=str(data_dir / "clients.csv"),
            purchases_table=str(data_dir / "purchases.csv"),
            train_table=str(data_dir / "uplift_train.csv"),
            scoring_table=str(data_dir / "uplift_test.csv"),
            products_table=str(data_dir / "products.csv"),
        ),
        split_contract=UpliftSplitContract(
            train_fraction=0.5,
            val_fraction=0.5,
            test_fraction=0.0,
            min_rows_per_partition=1,
            random_seed=7,
        ),
    )


def _wave(feature_recipe_ids: list[str]) -> UpliftExperimentWaveSpec:
    return UpliftExperimentWaveSpec(
        wave_id="UW-expansion-001",
        hypothesis_id="UH-expansion",
        action_type="feature_group_expansion",
        rationale="Compare base features with one approved expanded recipe.",
        trial_specs=[
            UpliftTrialSpec(
                spec_id=f"UT-expansion-{index}",
                hypothesis_id="UH-expansion",
                template_name="random_baseline",
                learner_family="random",
                base_estimator="none",
                feature_recipe_id=feature_recipe_id,
                split_seed=7,
                primary_metric="qini_auc",
            )
            for index, feature_recipe_id in enumerate(feature_recipe_ids, start=1)
        ],
        expected_signal="The expanded recipe changes validation ranking quality.",
        success_criterion="The wave selects a champion from successful trials.",
        abort_on_first_failure=True,
        required_feature_recipe_ids=feature_recipe_ids,
        created_by="manual",
    )


def test_registry_lists_stable_approved_recipe_families():
    registry = UpliftFeatureRecipeRegistry.default()

    assert registry.families() == [
        "base",
        "diversity",
        "engagement",
        "product_category",
        "rfm",
        "windowed",
    ]
    assert registry.recipe_for_family("base").feature_recipe_id == registry.recipe_id_for_family("base")
    assert registry.recipe_for_family("product_category").feature_recipe_id == registry.recipe_id_for_family("product_category")


def test_registry_rejects_unknown_recipe_family():
    registry = UpliftFeatureRecipeRegistry.default()

    with pytest.raises(ValueError, match="unknown feature recipe family"):
        registry.recipe_for_family("invented_family")


def test_registry_builds_and_reuses_cached_artifact(tmp_path):
    registry = UpliftFeatureRecipeRegistry.default()
    contract = _contract()

    artifact_a = registry.get_or_build_artifact(
        contract,
        family="windowed",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    artifact_path = Path(artifact_a.artifact_path)
    original_mtime = artifact_path.stat().st_mtime_ns
    artifact_b = registry.get_or_build_artifact(
        contract,
        family="windowed",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )

    assert artifact_a.feature_recipe_id == registry.recipe_id_for_family("windowed")
    assert artifact_a.feature_artifact_id == artifact_b.feature_artifact_id
    assert Path(artifact_b.artifact_path).stat().st_mtime_ns == original_mtime
    assert registry.artifact_for_recipe_id(artifact_a.feature_recipe_id) == artifact_b


def test_registry_rebuilds_when_dataset_fingerprint_changes(tmp_path):
    registry = UpliftFeatureRecipeRegistry.default()
    data_dir = _copy_fixture_dir(tmp_path)
    contract = _contract(data_dir)

    artifact_a = registry.get_or_build_artifact(
        contract,
        family="windowed",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    with (data_dir / "purchases.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            "c005,t005,2019-01-05 12:00:00,1.0,0.0,0.0,0.0,"
            "50.0,store1,p001,1.0,50.0,\n"
        )
    artifact_b = registry.get_or_build_artifact(
        contract,
        family="windowed",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )

    assert artifact_b.dataset_fingerprint != artifact_a.dataset_fingerprint
    assert artifact_b.feature_artifact_id != artifact_a.feature_artifact_id
    assert registry.artifact_for_recipe_id(artifact_a.feature_recipe_id) == artifact_b


def test_product_category_recipe_builds_one_row_per_customer_features(tmp_path):
    registry = UpliftFeatureRecipeRegistry.default()
    artifact = registry.get_or_build_artifact(
        _contract(),
        family="product_category",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    feature_df = pd.read_csv(artifact.artifact_path)

    assert artifact.row_count == 8
    assert feature_df["client_id"].is_unique
    assert "target" not in feature_df.columns
    assert "treatment_flg" not in feature_df.columns
    assert {
        "product_unique_count_lifetime",
        "product_level_1_unique_count_lifetime",
        "product_segment_unique_count_lifetime",
        "own_trademark_quantity_share_lifetime",
    }.issubset(feature_df.columns)


def test_feature_group_expansion_wave_consumes_registry_built_artifacts(tmp_path):
    registry = UpliftFeatureRecipeRegistry.default()
    contract = _contract()
    base = registry.get_or_build_artifact(
        contract,
        family="base",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    expanded = registry.get_or_build_artifact(
        contract,
        family="product_category",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    artifacts = {
        base.feature_recipe_id: base,
        expanded.feature_recipe_id: expanded,
    }
    wave = _wave(list(artifacts))

    validate_wave_spec(wave, feature_artifacts=artifacts)
    result = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts,
        output_dir=tmp_path / "runs",
    ).run_wave(wave)

    assert result.status == "completed"
    assert result.champion_run_id in result.trial_ids


def test_feature_group_expansion_rejects_unregistered_expansion_artifact(tmp_path):
    registry = UpliftFeatureRecipeRegistry.default()
    contract = _contract()
    base = registry.get_or_build_artifact(
        contract,
        family="base",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    rfm = registry.get_or_build_artifact(
        contract,
        family="rfm",
        output_dir=tmp_path / "features",
        cohort="train",
        chunksize=2,
    )
    artifacts = {base.feature_recipe_id: base, rfm.feature_recipe_id: rfm}

    with pytest.raises(ValueError, match="approved expansion feature group"):
        validate_wave_spec(_wave(list(artifacts)), feature_artifacts=artifacts)
