from src.models.uplift import UpliftExperimentWaveSpec, UpliftTrialSpec
from src.uplift.recipe_registry import UpliftFeatureRecipeRegistry


def test_m5_gate_exposes_registry_and_feature_group_expansion_action():
    registry = UpliftFeatureRecipeRegistry.default()

    assert "product_category" in registry.families()
    assert "diversity" in registry.families()
    assert (
        UpliftExperimentWaveSpec(
            wave_id="UW-expansion",
            hypothesis_id="UH-expansion",
            action_type="feature_group_expansion",
            rationale="Compare approved feature groups.",
            trial_specs=[
                UpliftTrialSpec(
                    spec_id="UT-base",
                    hypothesis_id="UH-expansion",
                    template_name="random_baseline",
                    learner_family="random",
                    base_estimator="none",
                    feature_recipe_id="recipe-base",
                ),
                UpliftTrialSpec(
                    spec_id="UT-expanded",
                    hypothesis_id="UH-expansion",
                    template_name="random_baseline",
                    learner_family="random",
                    base_estimator="none",
                    feature_recipe_id="recipe-expanded",
                ),
            ],
            expected_signal="Expanded features change ranking quality.",
            success_criterion="Champion is selected.",
            abort_on_first_failure=True,
            required_feature_recipe_ids=["recipe-base", "recipe-expanded"],
            created_by="manual",
        )
        is not None
    )
