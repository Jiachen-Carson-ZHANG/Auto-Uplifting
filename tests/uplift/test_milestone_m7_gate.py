from src.models.uplift import UpliftTrialSpec
from src.uplift.templates import REGISTERED_UPLIFT_TEMPLATES


def test_m7_gate_exposes_bounded_gradient_boosting_template_without_optuna():
    assert (
        REGISTERED_UPLIFT_TEMPLATES["two_model_gradient_boosting_sklearn"]
        == "two_model"
    )
    spec = UpliftTrialSpec(
        hypothesis_id="UH-m7",
        template_name="two_model_gradient_boosting_sklearn",
        learner_family="two_model",
        base_estimator="gradient_boosting",
        feature_recipe_id="recipe-a",
    )

    assert spec.base_estimator == "gradient_boosting"
    assert "optuna" not in REGISTERED_UPLIFT_TEMPLATES
