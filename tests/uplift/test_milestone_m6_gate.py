from src.models.uplift import UpliftExperimentWaveSpec, UpliftTrialSpec
from src.uplift.supervisor import (
    evaluate_policy_threshold_stability,
    evaluate_ranking_stability,
    evaluate_robustness,
)


def test_m6_gate_exposes_ranking_stability_action_and_robustness_helpers():
    wave = UpliftExperimentWaveSpec(
        wave_id="UW-stability",
        hypothesis_id="UH-stability",
        action_type="ranking_stability_check",
        rationale="Repeat one champion-like spec across seeds.",
        trial_specs=[
            UpliftTrialSpec(
                spec_id=f"UT-seed-{seed}",
                hypothesis_id="UH-stability",
                template_name="random_baseline",
                learner_family="random",
                base_estimator="none",
                feature_recipe_id="recipe-a",
                split_seed=seed,
            )
            for seed in [7, 11]
        ],
        expected_signal="Targeting rank is stable across seed repeats.",
        success_criterion="Rank and top-k diagnostics pass.",
        abort_on_first_failure=True,
        required_feature_recipe_ids=["recipe-a"],
        created_by="manual",
    )

    assert wave.action_type == "ranking_stability_check"
    assert callable(evaluate_ranking_stability)
    assert callable(evaluate_policy_threshold_stability)
    assert callable(evaluate_robustness)
