"""In-house sklearn uplift baseline templates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence

import numpy as np
import pandas as pd
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


def _make_classifier(random_seed: int) -> Pipeline | _ConstantProbabilityModel:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    solver="liblinear",
                    random_state=random_seed,
                ),
            ),
        ]
    )


def _fit_binary_classifier(
    features: pd.DataFrame,
    target: Sequence[int],
    *,
    random_seed: int,
) -> Pipeline | _ConstantProbabilityModel:
    y = np.asarray(target).astype(int)
    if len(np.unique(y)) < 2:
        return _ConstantProbabilityModel(float(y.mean()) if len(y) else 0.0)
    model = _make_classifier(random_seed)
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
    learner_family: Literal["random", "response_model", "two_model", "solo_model"],
    entity_key: str,
    treatment_col: str,
    target_col: str,
    random_seed: int,
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
        )

    if learner_family == "response_model":
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            model=_fit_binary_classifier(features, target, random_seed=random_seed),
        )

    if learner_family == "two_model":
        treatment_mask = train_df[treatment_col].astype(int) == 1
        treatment_model = _fit_binary_classifier(
            features[treatment_mask],
            target[treatment_mask],
            random_seed=random_seed,
        )
        control_model = _fit_binary_classifier(
            features[~treatment_mask],
            target[~treatment_mask],
            random_seed=random_seed,
        )
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            treatment_model=treatment_model,
            control_model=control_model,
        )

    if learner_family == "solo_model":
        solo_features = features.copy()
        solo_features["__treatment__"] = train_df[treatment_col].astype(int)
        solo_model = _fit_binary_classifier(
            solo_features,
            target,
            random_seed=random_seed,
        )
        return FittedUpliftModel(
            learner_family=learner_family,
            feature_columns=columns,
            random_seed=random_seed,
            model=solo_model,
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

    model = fit_uplift_model(
        train_df,
        learner_family=spec.learner_family,
        entity_key=entity_key,
        treatment_col=treatment_col,
        target_col=target_col,
        random_seed=spec.split_seed,
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
    )
