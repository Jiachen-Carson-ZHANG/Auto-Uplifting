from src.models.uplift import UpliftExperimentWaveSpec


def test_m2b_gate_records_original_deterministic_wave_actions():
    m2b_actions = {
        "recipe_comparison",
        "window_sweep",
        "feature_ablation",
    }
    later_milestone_actions = {
        "cost_sensitivity",
        "response_overlap_disambiguation",
        "ranking_stability_check",
        "feature_group_expansion",
    }

    assert m2b_actions.isdisjoint(later_milestone_actions)
    assert "action_type" in UpliftExperimentWaveSpec.model_fields
