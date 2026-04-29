"""In-house sklearn uplift baseline templates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.uplift import UpliftEvaluationPolicy, UpliftResultCard, UpliftTrialSpec
from src.uplift.metrics import UpliftMetricResult, evaluate_uplift_predictions


REGISTERED_UPLIFT_TEMPLATES: Dict[str, str] = {
    "random_baseline": "random",
    "response_model_sklearn": "response_model",
    "two_model_sklearn": "two_model",
    "solo_model_sklearn": "solo_model",
    "response_model_gradient_boosting_sklearn": "response_model",
    "two_model_gradient_boosting_sklearn": "two_model",
    "solo_model_gradient_boosting_sklearn": "solo_model",
    "class_transformation_sklearn": "class_transformation",
    "class_transformation_gradient_boosting_sklearn": "class_transformation",
    "two_model_xgboost": "two_model",
    "two_model_lightgbm": "two_model",
    "two_model_catboost": "two_model",
}

REGISTERED_UPLIFT_TEMPLATE_BASE_ESTIMATORS: Dict[str, str] = {
    "response_model_sklearn": "logistic_regression",
    "two_model_sklearn": "logistic_regression",
    "solo_model_sklearn": "logistic_regression",
    "response_model_gradient_boosting_sklearn": "gradient_boosting",
    "two_model_gradient_boosting_sklearn": "gradient_boosting",
    "solo_model_gradient_boosting_sklearn": "gradient_boosting",
    "class_transformation_sklearn": "logistic_regression",
    "class_transformation_gradient_boosting_sklearn": "gradient_boosting",
    "two_model_xgboost": "xgboost",
    "two_model_lightgbm": "lightgbm",
    "two_model_catboost": "catboost",
}


@dataclass
class UpliftTemplateOutput:
    """Template result plus row-level evaluation predictions.

    ``predictions``/``decile_table``/``qini_curve``/``uplift_curve`` are always
    derived from the validation frame used for champion selection. When a
    held-out test frame is supplied, the same artifacts on that frame are
    surfaced via the ``held_out_*`` attributes.
    """

    result_card: UpliftResultCard
    predictions: pd.DataFrame
    decile_table: pd.DataFrame
    qini_curve: pd.DataFrame
    uplift_curve: pd.DataFrame
    held_out_predictions: pd.DataFrame | None = None
    held_out_decile_table: pd.DataFrame | None = None
    held_out_qini_curve: pd.DataFrame | None = None
    held_out_uplift_curve: pd.DataFrame | None = None
    model: "FittedUpliftModel | None" = None


class _ConstantProbabilityModel:
    """Fallback probability model for tiny one-class slices."""

    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        proba = np.full(len(frame), self.probability, dtype=float)
        return np.column_stack([1.0 - proba, proba])


@dataclass
class FittedUpliftModel:
    """Fitted uplift model wrapper with a common scoring interface."""

    learner_family: str
    feature_columns: List[str]
    random_seed: int
    base_estimator: str = "logistic_regression"
    model: object | None = None
    treatment_model: object | None = None
    control_model: object | None = None

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.feature_columns].apply(pd.to_numeric, errors="coerce")

    def predict_uplift(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict uplift for rows that do not contain target/treatment columns."""
        if self.learner_family == "random":
            rng = np.random.RandomState(self.random_seed)
            return rng.random_sample(len(frame))

        features = self._features(frame)
        if self.learner_family == "response_model":
            return _positive_probability(self.model, features)

        if self.learner_family == "two_model":
            treated = _positive_probability(self.treatment_model, features)
            control = _positive_probability(self.control_model, features)
            return treated - control

        if self.learner_family == "solo_model":
            treated_features = features.copy()
            control_features = features.copy()
            treated_features["__treatment__"] = 1
            control_features["__treatment__"] = 0
            treated = _positive_probability(self.model, treated_features)
            control = _positive_probability(self.model, control_features)
            return treated - control

        if self.learner_family == "class_transformation":
            transformed = _positive_probability(self.model, features)
            return (2.0 * transformed) - 1.0

        raise ValueError(f"unsupported learner_family: {self.learner_family}")


def _feature_columns(
    frame: pd.DataFrame,
    *,
    entity_key: str,
    treatment_col: str,
    target_col: str,
) -> List[str]:
    forbidden = {entity_key, treatment_col, target_col}
    return [col for col in frame.columns if col not in forbidden]


def _make_classifier(
    *,
    base_estimator: str,
    random_seed: int,
    params: dict | None = None,
) -> Pipeline | _ConstantProbabilityModel:
    extra_params = dict(params or {})
    if base_estimator == "gradient_boosting":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    GradientBoostingClassifier(
                        **{
                            "n_estimators": 50,
                            "learning_rate": 0.05,
                            "max_depth": 2,
                            **extra_params,
                        },
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    if base_estimator == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        **{
                            "max_iter": 500,
                            "solver": "liblinear",
                            **extra_params,
                        },
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    if base_estimator == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("xgboost is required for base_estimator='xgboost'") from exc
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    XGBClassifier(
                        **{
                            "n_estimators": 100,
                            "max_depth": 5,
                            "learning_rate": 0.1,
                            "eval_metric": "logloss",
                            **extra_params,
                        },
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    if base_estimator == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("lightgbm is required for base_estimator='lightgbm'") from exc
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    LGBMClassifier(
                        **{
                            "n_estimators": 100,
                            "max_depth": 5,
                            "learning_rate": 0.1,
                            "verbose": -1,
                            **extra_params,
                        },
                        random_state=random_seed,
                    ),
                ),
            ]
        )
    if base_estimator == "catboost":
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError("catboost is required for base_estimator='catboost'") from exc
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    CatBoostClassifier(
                        **{
                            "iterations": 100,
                            "depth": 5,
                            "learning_rate": 0.1,
                            "verbose": 0,
                            **extra_params,
                        },
                        random_seed=random_seed,
                    ),
                ),
            ]
        )
    raise ValueError(f"unsupported base_estimator: {base_estimator}")


def _fit_binary_classifier(
    features: pd.DataFrame,
    target: Sequence[int],
    *,
    base_estimator: str,
    random_seed: int,
    params: dict | None = None,
) -> Pipeline | _ConstantProbabilityModel:
    y = np.asarray(target).astype(int)
    if len(np.unique(y)) < 2:
        return _ConstantProbabilityModel(float(y.mean()) if len(y) else 0.0)
    model = _make_classifier(
        base_estimator=base_estimator,
        random_seed=random_seed,
        params=params,
    )
    model.fit(features, y)
    return model


def _positive_probability(model: object | None, features: pd.DataFrame) -> np.ndarray:
    if model is None:
        raise ValueError("model is not fitted")
    proba = model.predict_proba(features)
    return np.asarray(proba)[:, 1]


def fit_uplift_model(
    train_df: pd.DataFrame,
    *,
    learner_family: Literal[
        "random",
        "response_model",
        "two_model",
        "solo_model",
        "class_transformation",
    ],
    base_estimator: Literal[
        "logistic_regression",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
        "catboost",
    ] = "logistic_regression",
    entity_key: str,
    treatment_col: str,
    target_col: str,
    random_seed: int,
    params: dict | None = None,
) -> FittedUpliftModel:
    """Fit one in-house uplift baseline using train rows only."""
    columns = _feature_columns(
        train_df,
        entity_key=entity_key,
        treatment_col=treatment_col,
        target_col=target_col,
    )
    features = train_df[columns].apply(pd.to_numeric, errors="coerce")
    target = train_df[target_col].astype(int)

    if learner_family == "random":
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            base_estimator=base_estimator,
        )

    if learner_family == "response_model":
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            base_estimator=base_estimator,
            model=_fit_binary_classifier(
                features,
                target,
                base_estimator=base_estimator,
                random_seed=random_seed,
                params=params,
            ),
        )

    if learner_family == "two_model":
        treatment_mask = train_df[treatment_col].astype(int) == 1
        treatment_model = _fit_binary_classifier(
            features[treatment_mask],
            target[treatment_mask],
            base_estimator=base_estimator,
            random_seed=random_seed,
            params=params,
        )
        control_model = _fit_binary_classifier(
            features[~treatment_mask],
            target[~treatment_mask],
            base_estimator=base_estimator,
            random_seed=random_seed,
            params=params,
        )
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            base_estimator=base_estimator,
            treatment_model=treatment_model,
            control_model=control_model,
        )

    if learner_family == "solo_model":
        solo_features = features.copy()
        solo_features["__treatment__"] = train_df[treatment_col].astype(int)
        solo_model = _fit_binary_classifier(
            solo_features,
            target,
            base_estimator=base_estimator,
            random_seed=random_seed,
            params=params,
        )
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            base_estimator=base_estimator,
            model=solo_model,
        )

    if learner_family == "class_transformation":
        transformed_target = (
            train_df[target_col].astype(int) == train_df[treatment_col].astype(int)
        ).astype(int)
        transformed_model = _fit_binary_classifier(
            features,
            transformed_target,
            base_estimator=base_estimator,
            random_seed=random_seed,
            params=params,
        )
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            base_estimator=base_estimator,
            model=transformed_model,
        )

    raise ValueError(f"unsupported learner_family: {learner_family}")


def _score_frame(
    model: FittedUpliftModel,
    frame: pd.DataFrame,
    *,
    entity_key: str,
    treatment_col: str,
    target_col: str,
    policy: UpliftEvaluationPolicy,
) -> tuple[pd.DataFrame, UpliftMetricResult]:
    predicted = model.predict_uplift(frame)
    metric_result = evaluate_uplift_predictions(
        frame[target_col].to_numpy(),
        frame[treatment_col].to_numpy(),
        predicted,
        policy,
    )
    predictions = pd.DataFrame(
        {
            entity_key: frame[entity_key].to_numpy(),
            "target": frame[target_col].to_numpy(),
            "treatment_flg": frame[treatment_col].to_numpy(),
            "uplift": predicted,
        }
    )
    return predictions, metric_result


def run_uplift_template(
    spec: UpliftTrialSpec,
    *,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    entity_key: str,
    treatment_col: str,
    target_col: str,
    cutoff_grid: List[float],
    held_out_df: pd.DataFrame | None = None,
) -> UpliftTemplateOutput:
    """Fit a registered uplift template, score validation, and optionally score a held-out test frame.

    Validation metrics on ``eval_df`` drive champion selection. When
    ``held_out_df`` is supplied (typically the test partition), the same
    fitted model is also scored on it to produce honest held-out metrics
    that callers can report alongside the validation metrics.
    """
    expected_family = REGISTERED_UPLIFT_TEMPLATES.get(spec.template_name)
    if expected_family is None:
        raise ValueError(f"unknown uplift template: {spec.template_name}")
    if expected_family != spec.learner_family:
        raise ValueError(
            f"template {spec.template_name} expects learner_family={expected_family}"
        )
    expected_base_estimator = REGISTERED_UPLIFT_TEMPLATE_BASE_ESTIMATORS.get(
        spec.template_name
    )
    if (
        expected_base_estimator is not None
        and spec.base_estimator != expected_base_estimator
    ):
        raise ValueError(
            f"template {spec.template_name} expects "
            f"base_estimator={expected_base_estimator}"
        )

    model = fit_uplift_model(
        train_df,
        learner_family=spec.learner_family,
        base_estimator=expected_base_estimator or spec.base_estimator,
        entity_key=entity_key,
        treatment_col=treatment_col,
        target_col=target_col,
        random_seed=spec.split_seed,
        params=spec.params,
    )
    policy = UpliftEvaluationPolicy(cutoff_grid=cutoff_grid)
    predictions, metric_result = _score_frame(
        model,
        eval_df,
        entity_key=entity_key,
        treatment_col=treatment_col,
        target_col=target_col,
        policy=policy,
    )

    held_out_predictions = None
    held_out_decile = None
    held_out_qini = None
    held_out_uplift = None
    held_out_qini_auc = None
    held_out_uplift_auc = None
    held_out_uplift_at_k: Dict[str, float] = {}
    held_out_policy_gain: Dict[str, float] = {}

    if held_out_df is not None and not held_out_df.empty:
        held_out_predictions, held_out_metrics = _score_frame(
            model,
            held_out_df,
            entity_key=entity_key,
            treatment_col=treatment_col,
            target_col=target_col,
            policy=policy,
        )
        held_out_decile = held_out_metrics.decile_table
        held_out_qini = held_out_metrics.qini_curve
        held_out_uplift = held_out_metrics.uplift_curve
        held_out_qini_auc = held_out_metrics.qini_auc
        held_out_uplift_auc = held_out_metrics.uplift_auc
        held_out_uplift_at_k = held_out_metrics.uplift_at_k
        held_out_policy_gain = held_out_metrics.policy_gain

    result = UpliftResultCard(
        trial_spec_id=spec.spec_id,
        status="success",
        qini_auc=metric_result.qini_auc,
        uplift_auc=metric_result.uplift_auc,
        uplift_at_k=metric_result.uplift_at_k,
        policy_gain=metric_result.policy_gain,
        held_out_qini_auc=held_out_qini_auc,
        held_out_uplift_auc=held_out_uplift_auc,
        held_out_uplift_at_k=held_out_uplift_at_k,
        held_out_policy_gain=held_out_policy_gain,
    )
    return UpliftTemplateOutput(
        result_card=result,
        predictions=predictions,
        decile_table=metric_result.decile_table,
        qini_curve=metric_result.qini_curve,
        uplift_curve=metric_result.uplift_curve,
        held_out_predictions=held_out_predictions,
        held_out_decile_table=held_out_decile,
        held_out_qini_curve=held_out_qini,
        held_out_uplift_curve=held_out_uplift,
        model=model,
    )
