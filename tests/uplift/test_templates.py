import builtins
from pathlib import Path

import pandas as pd
import pytest

from src.models.uplift import UpliftTrialSpec
from src.uplift.templates import (
    REGISTERED_UPLIFT_TEMPLATES,
    _make_classifier,
    fit_uplift_model,
    run_uplift_template,
)


FIXTURE_DIR = Path("tests/fixtures/uplift")


def _labeled_feature_frame() -> pd.DataFrame:
    features = pd.DataFrame(
        {
            "client_id": [f"c{i:03d}" for i in range(1, 9)],
            "feature_a": [0.0, 0.2, 0.7, 0.9, 0.1, 0.3, 0.8, 1.0],
            "feature_b": [1, 1, 0, 0, 1, 1, 0, 0],
        }
    )
    labels = pd.read_csv(FIXTURE_DIR / "uplift_train.csv")
    return features.merge(labels, on="client_id", how="inner")


def test_registered_uplift_templates_are_the_v1_baseline_ladder():
    assert {
        "random_baseline",
        "response_model_sklearn",
        "two_model_sklearn",
        "solo_model_sklearn",
        "two_model_gradient_boosting_sklearn",
        "solo_model_gradient_boosting_sklearn",
        "response_model_gradient_boosting_sklearn",
        "class_transformation_sklearn",
        "class_transformation_gradient_boosting_sklearn",
        "two_model_xgboost",
        "two_model_lightgbm",
        "two_model_catboost",
    }.issubset(set(REGISTERED_UPLIFT_TEMPLATES))


def test_run_uplift_template_returns_predictions_and_metrics_for_each_baseline():
    frame = _labeled_feature_frame()
    train = frame.iloc[4:].copy()
    test = frame.iloc[:4].copy()

    for template_name, learner_family in [
        ("random_baseline", "random"),
        ("response_model_sklearn", "response_model"),
        ("two_model_sklearn", "two_model"),
        ("solo_model_sklearn", "solo_model"),
        ("two_model_gradient_boosting_sklearn", "two_model"),
        ("solo_model_gradient_boosting_sklearn", "solo_model"),
        ("response_model_gradient_boosting_sklearn", "response_model"),
        ("class_transformation_sklearn", "class_transformation"),
        ("class_transformation_gradient_boosting_sklearn", "class_transformation"),
    ]:
        base_estimator = (
            "gradient_boosting"
            if "gradient_boosting" in template_name
            else "logistic_regression"
        )
        spec = UpliftTrialSpec(
            hypothesis_id="baseline",
            template_name=template_name,
            learner_family=learner_family,
            base_estimator=base_estimator,
            feature_recipe_id="recipe123456",
            split_seed=3,
        )

        output = run_uplift_template(
            spec,
            train_df=train,
            eval_df=test,
            entity_key="client_id",
            treatment_col="treatment_flg",
            target_col="target",
            cutoff_grid=[0.5],
        )

        assert output.predictions.shape[0] == len(test)
        assert output.predictions["uplift"].notna().all()
        assert output.result_card.status == "success"
        assert output.result_card.qini_auc is not None


def test_fit_uplift_model_predicts_scoring_rows_without_target_or_treatment():
    frame = _labeled_feature_frame()
    scoring = frame[["client_id", "feature_a", "feature_b"]].copy()
    model = fit_uplift_model(
        frame,
        learner_family="two_model",
        entity_key="client_id",
        treatment_col="treatment_flg",
        target_col="target",
        random_seed=11,
    )

    uplift = model.predict_uplift(scoring)

    assert len(uplift) == len(scoring)
    assert pd.Series(uplift).notna().all()
    assert "target" not in model.feature_columns
    assert "treatment_flg" not in model.feature_columns


def test_fit_uplift_model_supports_bounded_gradient_boosting_two_model():
    frame = _labeled_feature_frame()
    scoring = frame[["client_id", "feature_a", "feature_b"]].copy()

    model = fit_uplift_model(
        frame,
        learner_family="two_model",
        base_estimator="gradient_boosting",
        entity_key="client_id",
        treatment_col="treatment_flg",
        target_col="target",
        random_seed=11,
    )
    uplift = model.predict_uplift(scoring)

    assert model.base_estimator == "gradient_boosting"
    assert len(uplift) == len(scoring)
    assert pd.Series(uplift).notna().all()
    assert "target" not in model.feature_columns
    assert "treatment_flg" not in model.feature_columns


def test_fit_uplift_model_supports_class_transformation_family():
    frame = _labeled_feature_frame()
    scoring = frame[["client_id", "feature_a", "feature_b"]].copy()

    model = fit_uplift_model(
        frame,
        learner_family="class_transformation",
        base_estimator="logistic_regression",
        entity_key="client_id",
        treatment_col="treatment_flg",
        target_col="target",
        random_seed=11,
    )
    uplift = model.predict_uplift(scoring)

    assert model.learner_family == "class_transformation"
    assert len(uplift) == len(scoring)
    assert pd.Series(uplift).between(-1.0, 1.0).all()


def test_gradient_boosting_template_rejects_base_estimator_mismatch():
    frame = _labeled_feature_frame()
    spec = UpliftTrialSpec(
        hypothesis_id="m7",
        template_name="two_model_gradient_boosting_sklearn",
        learner_family="two_model",
        base_estimator="logistic_regression",
        feature_recipe_id="recipe123456",
        split_seed=3,
    )

    try:
        run_uplift_template(
            spec,
            train_df=frame.iloc[4:].copy(),
            eval_df=frame.iloc[:4].copy(),
            entity_key="client_id",
            treatment_col="treatment_flg",
            target_col="target",
            cutoff_grid=[0.5],
        )
    except ValueError as exc:
        assert "base_estimator=gradient_boosting" in str(exc)
    else:
        raise AssertionError("template should reject base_estimator mismatch")


def test_optional_booster_branches_report_missing_dependencies(monkeypatch):
    real_import = builtins.__import__

    def blocked_optional_import(name, *args, **kwargs):
        if name in {"lightgbm", "catboost"}:
            raise ImportError(f"blocked {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_optional_import)

    with pytest.raises(ImportError, match="lightgbm is required"):
        _make_classifier(base_estimator="lightgbm", random_seed=7)

    with pytest.raises(ImportError, match="catboost is required"):
        _make_classifier(base_estimator="catboost", random_seed=7)
