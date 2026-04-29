from pathlib import Path

from src.models.uplift import UpliftExperimentWaveSpec, UpliftWaveResult


def test_m2a_wave_contracts_match_design_gate():
    assert {
        "wave_id",
        "hypothesis_id",
        "action_type",
        "rationale",
        "trial_specs",
        "expected_signal",
        "success_criterion",
        "abort_on_first_failure",
        "required_feature_recipe_ids",
        "created_by",
    }.issubset(UpliftExperimentWaveSpec.model_fields)

    assert {
        "wave_id",
        "hypothesis_id",
        "action_type",
        "status",
        "trial_ids",
        "failed_trial_ids",
        "blocked_reason",
        "champion_run_id",
        "artifact_paths",
    }.issubset(UpliftWaveResult.model_fields)
    assert "selected_metric_summary" not in UpliftWaveResult.model_fields


def test_m2a_supervisor_boundary_does_not_import_generic_experiment_index():
    supervisor_dir = Path("src/uplift/supervisor")
    contents = "\n".join(
        path.read_text(encoding="utf-8") for path in supervisor_dir.glob("*.py")
    )

    assert "ExperimentIndex" not in contents
