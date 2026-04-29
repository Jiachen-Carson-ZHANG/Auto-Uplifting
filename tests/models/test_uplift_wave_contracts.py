import pytest
from pydantic import ValidationError

from src.models.uplift import (
    UpliftExperimentWaveSpec,
    UpliftTrialSpec,
    UpliftWaveResult,
)


def _trial(
    spec_id: str,
    feature_recipe_id: str,
    *,
    hypothesis_id: str = "UH-recipe",
    template_name: str = "random_baseline",
    learner_family: str = "random",
    base_estimator: str = "none",
    split_seed: int = 7,
    primary_metric: str = "qini_auc",
) -> UpliftTrialSpec:
    return UpliftTrialSpec(
        spec_id=spec_id,
        hypothesis_id=hypothesis_id,
        template_name=template_name,
        learner_family=learner_family,
        base_estimator=base_estimator,
        feature_recipe_id=feature_recipe_id,
        split_seed=split_seed,
        primary_metric=primary_metric,
    )


def _wave(**updates) -> UpliftExperimentWaveSpec:
    payload = {
        "wave_id": "UW-recipe-001",
        "hypothesis_id": "UH-recipe",
        "action_type": "recipe_comparison",
        "rationale": "Compare two deterministic feature recipes under the same learner.",
        "trial_specs": [
            _trial("UT-recipe-a", "recipe-a"),
            _trial("UT-recipe-b", "recipe-b"),
        ],
        "expected_signal": "One recipe improves validation qini_auc.",
        "success_criterion": "Champion run is selected from successful trial records.",
        "abort_on_first_failure": True,
        "required_feature_recipe_ids": ["recipe-a", "recipe-b"],
        "created_by": "manual",
    }
    payload.update(updates)
    return UpliftExperimentWaveSpec(**payload)


def test_recipe_comparison_wave_canonicalizes_required_recipe_ids():
    wave = _wave(required_feature_recipe_ids=["recipe-b", "recipe-a", "recipe-a"])

    assert wave.required_feature_recipe_ids == ["recipe-b", "recipe-a"]
    assert len(wave.trial_specs) == 2


def test_wave_spec_supports_deterministic_actions_and_rejects_deferred_actions():
    for action_type in [
        "cost_sensitivity",
        "recipe_comparison",
        "ranking_stability_check",
        "window_sweep",
        "feature_ablation",
        "feature_group_expansion",
    ]:
        if action_type == "ranking_stability_check":
            wave = _wave(
                action_type=action_type,
                trial_specs=[
                    _trial("UT-seed-a", "recipe-a", split_seed=7),
                    _trial("UT-seed-b", "recipe-a", split_seed=11),
                ],
                required_feature_recipe_ids=["recipe-a"],
            )
        else:
            wave = _wave(action_type=action_type)
        assert wave.action_type == action_type

    with pytest.raises(ValidationError, match="supported wave actions"):
        _wave(action_type="response_overlap_disambiguation")


def test_recipe_comparison_requires_two_to_four_trials():
    with pytest.raises(ValidationError, match="2 to 4 trial_specs"):
        _wave(trial_specs=[_trial("UT-one", "recipe-a")])

    with pytest.raises(ValidationError, match="2 to 4 trial_specs"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-b"),
                _trial("UT-c", "recipe-c"),
                _trial("UT-d", "recipe-d"),
                _trial("UT-e", "recipe-e"),
            ],
            required_feature_recipe_ids=[
                "recipe-a",
                "recipe-b",
                "recipe-c",
                "recipe-d",
                "recipe-e",
            ],
        )


def test_feature_ablation_wave_requires_exactly_two_trials():
    with pytest.raises(ValidationError, match="feature_ablation waves require exactly 2"):
        _wave(
            action_type="feature_ablation",
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-b"),
                _trial("UT-c", "recipe-c"),
            ],
            required_feature_recipe_ids=["recipe-a", "recipe-b", "recipe-c"],
        )


def test_recipe_comparison_requires_feature_recipe_contrast():
    with pytest.raises(ValidationError, match="at least two distinct feature_recipe_id"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-a"),
            ],
            required_feature_recipe_ids=["recipe-a"],
        )


def test_recipe_comparison_trials_share_template_split_and_metric_contract():
    with pytest.raises(ValidationError, match="same template_name"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial(
                    "UT-b",
                    "recipe-b",
                    template_name="response_model_sklearn",
                    learner_family="response_model",
                    base_estimator="logistic_regression",
                ),
            ]
        )

    with pytest.raises(ValidationError, match="same split_seed"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-b", split_seed=11),
            ]
        )

    with pytest.raises(ValidationError, match="same primary_metric"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-b", primary_metric="uplift_auc"),
            ]
        )


def test_wave_spec_rejects_trials_for_other_hypotheses_or_recipes():
    with pytest.raises(ValidationError, match="same hypothesis_id"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-b", hypothesis_id="UH-other"),
            ]
        )

    with pytest.raises(ValidationError, match="required_feature_recipe_ids"):
        _wave(
            trial_specs=[
                _trial("UT-a", "recipe-a"),
                _trial("UT-b", "recipe-c"),
            ]
        )


def test_wave_result_links_to_trial_run_ids_without_metric_snapshot():
    result = UpliftWaveResult(
        wave_id="UW-recipe-001",
        hypothesis_id="UH-recipe",
        action_type="recipe_comparison",
        status="completed",
        trial_ids=["RUN-a", "RUN-b"],
        failed_trial_ids=[],
        blocked_reason=None,
        champion_run_id="RUN-b",
        artifact_paths={"ledger": "runs/uplift_ledger.jsonl"},
    )

    assert result.champion_run_id == "RUN-b"
    assert "selected_metric_summary" not in UpliftWaveResult.model_fields


def test_wave_result_accepts_m2b_deterministic_actions():
    result = UpliftWaveResult(
        wave_id="UW-window-001",
        hypothesis_id="UH-window",
        action_type="window_sweep",
        status="completed",
        trial_ids=["RUN-a", "RUN-b"],
        failed_trial_ids=[],
        blocked_reason=None,
        champion_run_id="RUN-a",
        artifact_paths={"ledger": "runs/uplift_ledger.jsonl"},
    )

    assert result.action_type == "window_sweep"


def test_wave_result_status_must_match_failure_or_blocking_fields():
    with pytest.raises(ValidationError, match="blocked_reason"):
        UpliftWaveResult(
            wave_id="UW-recipe-001",
            hypothesis_id="UH-recipe",
            action_type="recipe_comparison",
            status="blocked",
            trial_ids=[],
            failed_trial_ids=[],
            blocked_reason=None,
            champion_run_id=None,
            artifact_paths={},
        )

    with pytest.raises(ValidationError, match="failed_trial_ids"):
        UpliftWaveResult(
            wave_id="UW-recipe-001",
            hypothesis_id="UH-recipe",
            action_type="recipe_comparison",
            status="completed",
            trial_ids=["RUN-a"],
            failed_trial_ids=["RUN-a"],
            blocked_reason=None,
            champion_run_id="RUN-a",
            artifact_paths={},
        )
