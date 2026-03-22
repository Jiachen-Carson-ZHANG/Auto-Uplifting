from __future__ import annotations
import json
import logging
import math
from typing import Any, Optional, Tuple
from src.models.results import RunResult, ModelEntry

logger = logging.getLogger(__name__)


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable primitives.

    Always recurse into containers rather than relying on json.dumps as a test —
    some AutoGluon types (e.g. FeatureMetadata) have __iter__ that fools the
    stdlib json encoder but are rejected by pydantic_core serialization.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(i) for i in obj]
    return str(obj)


class ResultParser:
    """Converts AutoGluon predictor output into a RunResult."""

    @staticmethod
    def from_predictor(
        predictor: Any,
        fit_time: float,
        primary_metric_value: float,
    ) -> Tuple[RunResult, Optional[float]]:
        leaderboard_entries = []
        overfitting_gap = None
        try:
            lb = predictor.leaderboard(extra_info=True)
            best_row = lb.iloc[0]
            _score_train_raw = float(best_row["score_train"]) if "score_train" in lb.columns else None
            score_train = (
                _score_train_raw
                if _score_train_raw is not None and not math.isnan(_score_train_raw)
                else None
            )
            score_val = float(best_row["score_val"])
            if score_train is not None:
                overfitting_gap = round(score_train - score_val, 4)
            for row in lb.itertuples():
                _st = getattr(row, "score_train", None)
                leaderboard_entries.append(ModelEntry(
                    model_name=row.model,
                    score_val=row.score_val,
                    fit_time=getattr(row, "fit_time", None) or getattr(row, "fit_time_marginal", 0.0),
                    pred_time=getattr(row, "pred_time_val", None) or getattr(row, "pred_time", 0.0),
                    stack_level=getattr(row, "stack_level", 1),
                    score_train=_st if _st is not None and not math.isnan(float(_st)) else None,
                ))
        except Exception as e:
            logger.warning("leaderboard(extra_info=True) failed, falling back to basic leaderboard: %s", e)
            try:
                lb = predictor.leaderboard()
                for row in lb.itertuples():
                    leaderboard_entries.append(ModelEntry(
                        model_name=row.model,
                        score_val=row.score_val,
                        fit_time=getattr(row, "fit_time", None) or getattr(row, "fit_time_marginal", 0.0),
                        pred_time=getattr(row, "pred_time_val", None) or getattr(row, "pred_time", 0.0),
                        stack_level=getattr(row, "stack_level", 1),
                    ))
            except Exception:
                pass

        result = RunResult(
            status="success",
            primary_metric=primary_metric_value,
            leaderboard=leaderboard_entries,
            best_model_name=getattr(predictor, "model_best", None),
            fit_time_seconds=fit_time,
        )
        return result, overfitting_gap

    @staticmethod
    def from_error(error_msg: str) -> RunResult:
        return RunResult(
            status="failed",
            error=error_msg,
        )
