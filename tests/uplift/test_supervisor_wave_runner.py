from pathlib import Path

import pytest

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftExperimentWaveSpec,
    UpliftFeatureRecipeSpec,
    UpliftHypothesis,
    UpliftProjectContract,
    UpliftSplitContract,
    UpliftTableSchema,
    UpliftTrialSpec,
    UpliftWaveResult,
)
from src.uplift.features import build_feature_table
from src.uplift.hypotheses import UpliftHypothesisStore
from src.uplift.loop import UpliftLoopResult
from src.uplift.supervisor import UpliftResearchLoop


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


def _recipe(
    window_days: int,
    *,
    feature_groups: list[str] | None = None,
) -> UpliftFeatureRecipeSpec:
    return UpliftFeatureRecipeSpec(
        source_tables=["clients", "purchases"],
        feature_groups=feature_groups or ["demographic", "rfm", "basket", "points"],
        windows_days=[window_days],
        builder_version="v1",
    )


def _feature_artifacts(tmp_path: Path):
    contract = _contract()
    artifacts = [
        build_feature_table(
            contract,
            recipe=_recipe(7),
            output_dir=tmp_path / "features",
            cohort="train",
            chunksize=2,
        ),
        build_feature_table(
            contract,
            recipe=_recipe(30),
            output_dir=tmp_path / "features",
            cohort="train",
            chunksize=2,
        ),
    ]
    return contract, {artifact.feature_recipe_id: artifact for artifact in artifacts}


def _build_artifacts_for_recipes(
    tmp_path: Path,
    recipes: list[UpliftFeatureRecipeSpec],
):
    contract = _contract()
    artifacts = [
        build_feature_table(
            contract,
            recipe=recipe,
            output_dir=tmp_path / "features",
            cohort="train",
            chunksize=2,
        )
        for recipe in recipes
    ]
    return contract, {artifact.feature_recipe_id: artifact for artifact in artifacts}


def _wave(
    feature_recipe_ids: list[str],
    *,
    action_type: str = "recipe_comparison",
    hypothesis_id: str = "UH-recipe",
    abort_on_first_failure: bool = False,
    template_name: str = "random_baseline",
) -> UpliftExperimentWaveSpec:
    return UpliftExperimentWaveSpec(
        wave_id="UW-recipe-001",
        hypothesis_id=hypothesis_id,
        action_type=action_type,
        rationale=f"Compare deterministic feature recipes for {action_type}.",
        trial_specs=[
            UpliftTrialSpec(
                spec_id=f"UT-recipe-{index}",
                hypothesis_id=hypothesis_id,
                template_name=template_name,
                learner_family="random",
                base_estimator="none",
                feature_recipe_id=feature_recipe_id,
                split_seed=7,
                primary_metric="qini_auc",
            )
            for index, feature_recipe_id in enumerate(feature_recipe_ids, start=1)
        ],
        expected_signal="A feature recipe changes validation qini_auc.",
        success_criterion="A champion run is selected from successful trials.",
        abort_on_first_failure=abort_on_first_failure,
        required_feature_recipe_ids=feature_recipe_ids,
        created_by="manual",
    )


def test_research_loop_runs_recipe_comparison_wave_and_links_hypothesis_store(tmp_path):
    contract, artifacts_by_recipe = _feature_artifacts(tmp_path)
    feature_recipe_ids = list(artifacts_by_recipe)
    hypothesis = UpliftHypothesis(
        hypothesis_id="UH-recipe",
        question="Which feature recipe is more useful?",
        hypothesis_text="Purchase-window features should move uplift ranking quality.",
        stage_origin="manual",
        action_type="recipe_comparison",
    )
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")
    store.append(hypothesis)
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
        hypothesis_store=store,
    )

    result = loop.run_wave(_wave(feature_recipe_ids))

    assert result.status == "completed"
    assert len(result.trial_ids) == 2
    assert result.failed_trial_ids == []
    assert result.champion_run_id in result.trial_ids
    assert Path(result.artifact_paths["ledger"]).exists()
    assert "selected_metric_summary" not in UpliftWaveResult.model_fields

    latest = store.get_latest("UH-recipe")
    assert latest is not None
    assert latest.status == "under_test"
    assert latest.wave_ids == ["UW-recipe-001"]
    assert latest.trial_ids == result.trial_ids


def test_research_loop_runs_window_sweep_wave_using_cached_artifacts(tmp_path):
    contract, artifacts_by_recipe = _build_artifacts_for_recipes(
        tmp_path,
        [_recipe(7), _recipe(30)],
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    result = loop.run_wave(
        _wave(list(artifacts_by_recipe), action_type="window_sweep")
    )

    assert result.status == "completed"
    assert len(result.trial_ids) == 2
    assert result.champion_run_id in result.trial_ids


def test_research_loop_runs_feature_ablation_wave_using_cached_artifacts(tmp_path):
    contract, artifacts_by_recipe = _build_artifacts_for_recipes(
        tmp_path,
        [
            _recipe(30, feature_groups=["demographic", "rfm", "basket", "points"]),
            _recipe(30, feature_groups=["demographic", "rfm", "basket"]),
        ],
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    result = loop.run_wave(
        _wave(list(artifacts_by_recipe), action_type="feature_ablation")
    )

    assert result.status == "completed"
    assert len(result.trial_ids) == 2
    assert result.champion_run_id in result.trial_ids


def test_window_sweep_rejects_unapproved_windows_before_kernel_execution(
    tmp_path, monkeypatch
):
    contract, artifacts_by_recipe = _build_artifacts_for_recipes(
        tmp_path,
        [_recipe(7), _recipe(365)],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trial kernel should not run after preflight failure")

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fail_if_called,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    with pytest.raises(ValueError, match="unapproved window"):
        loop.run_wave(_wave(list(artifacts_by_recipe), action_type="window_sweep"))


def test_feature_ablation_rejects_unknown_feature_groups_before_kernel_execution(
    tmp_path, monkeypatch
):
    contract, artifacts_by_recipe = _build_artifacts_for_recipes(
        tmp_path,
        [
            _recipe(30, feature_groups=["demographic", "rfm", "mystery"]),
            _recipe(30, feature_groups=["demographic", "rfm"]),
        ],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trial kernel should not run after preflight failure")

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fail_if_called,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    with pytest.raises(ValueError, match="unknown feature group"):
        loop.run_wave(_wave(list(artifacts_by_recipe), action_type="feature_ablation"))


def test_feature_ablation_rejects_multi_group_differences_before_kernel_execution(
    tmp_path, monkeypatch
):
    contract, artifacts_by_recipe = _build_artifacts_for_recipes(
        tmp_path,
        [
            _recipe(30, feature_groups=["demographic", "rfm", "basket", "points"]),
            _recipe(30, feature_groups=["demographic", "rfm"]),
        ],
    )

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trial kernel should not run after preflight failure")

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fail_if_called,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    with pytest.raises(ValueError, match="exactly one known feature group"):
        loop.run_wave(_wave(list(artifacts_by_recipe), action_type="feature_ablation"))


def test_wave_preflight_rejects_unknown_feature_recipe_before_kernel_execution(
    tmp_path, monkeypatch
):
    contract, artifacts_by_recipe = _feature_artifacts(tmp_path)
    feature_recipe_ids = list(artifacts_by_recipe)
    unknown_feature_recipe_id = "missing-recipe"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trial kernel should not run after preflight failure")

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fail_if_called,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    with pytest.raises(ValueError, match="unknown feature recipe"):
        loop.run_wave(_wave([feature_recipe_ids[0], unknown_feature_recipe_id]))


def test_wave_preflight_rejects_unknown_template_before_kernel_execution(
    tmp_path, monkeypatch
):
    contract, artifacts_by_recipe = _feature_artifacts(tmp_path)

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trial kernel should not run after preflight failure")

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fail_if_called,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    with pytest.raises(ValueError, match="unknown uplift template"):
        loop.run_wave(_wave(list(artifacts_by_recipe), template_name="missing_template"))


def test_wave_preflight_rejects_unknown_hypothesis_before_kernel_execution(
    tmp_path, monkeypatch
):
    contract, artifacts_by_recipe = _feature_artifacts(tmp_path)
    store = UpliftHypothesisStore(tmp_path / "hypotheses.jsonl")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("trial kernel should not run after preflight failure")

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fail_if_called,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
        hypothesis_store=store,
    )

    with pytest.raises(ValueError, match="unknown hypothesis_id"):
        loop.run_wave(_wave(list(artifacts_by_recipe), hypothesis_id="UH-missing"))


def test_wave_abort_on_first_failure_stops_after_failed_trial(tmp_path, monkeypatch):
    contract, artifacts_by_recipe = _feature_artifacts(tmp_path)
    calls: list[str] = []

    def fake_run_uplift_trials(
        contract,
        *,
        feature_artifact,
        trial_specs,
        output_dir,
    ):
        spec = trial_specs[0]
        calls.append(spec.spec_id)
        record = UpliftExperimentRecord(
            run_id=f"RUN-{spec.spec_id}",
            hypothesis_id=spec.hypothesis_id,
            feature_recipe_id=spec.feature_recipe_id,
            feature_artifact_id=feature_artifact.feature_artifact_id,
            template_name=spec.template_name,
            uplift_learner_family=spec.learner_family,
            base_estimator=spec.base_estimator,
            params_hash="fake",
            split_seed=spec.split_seed,
            status="failed",
            error="template failed",
        )
        return UpliftLoopResult(
            records=[record],
            ledger_path=str(tmp_path / "runs" / "uplift_ledger.jsonl"),
            output_dir=str(output_dir),
        )

    monkeypatch.setattr(
        "src.uplift.supervisor.waves.run_uplift_trials",
        fake_run_uplift_trials,
    )
    loop = UpliftResearchLoop(
        contract=contract,
        feature_artifacts=artifacts_by_recipe,
        output_dir=tmp_path / "runs",
    )

    result = loop.run_wave(_wave(list(artifacts_by_recipe), abort_on_first_failure=True))

    assert calls == ["UT-recipe-1"]
    assert result.status == "blocked"
    assert result.failed_trial_ids == ["RUN-UT-recipe-1"]
    assert result.blocked_reason == "template failed"
    assert result.champion_run_id is None
