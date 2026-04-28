from pathlib import Path

import pandas as pd

from src.models.uplift import UpliftTrialSpec
from src.uplift.templates import (
    REGISTERED_UPLIFT_TEMPLATES,
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
    assert set(REGISTERED_UPLIFT_TEMPLATES) == {
        "random_baseline",
        "response_model_sklearn",
        "two_model_sklearn",
        "solo_model_sklearn",
    }


def test_run_uplift_template_returns_predictions_and_metrics_for_each_baseline():
    frame = _labeled_feature_frame()
    train = frame.iloc[4:].copy()
    test = frame.iloc[:4].copy()

    for template_name, learner_family in [
        ("random_baseline", "random"),
        ("response_model_sklearn", "response_model"),
        ("two_model_sklearn", "two_model"),
        ("solo_model_sklearn", "solo_model"),
    ]:
        spec = UpliftTrialSpec(
            hypothesis_id="baseline",
            template_name=template_name,
            learner_family=learner_family,
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
