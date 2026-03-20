from __future__ import annotations
import json
import logging
from typing import Any
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
        run_id: str,
        fit_time: float,
        artifacts_dir: str,
        primary_metric_value: float,
    ) -> RunResult:
        leaderboard_entries = []
        overfitting_gap = None
        try:
            lb = predictor.leaderboard(extra_info=True)
            best_row = lb.iloc[0]
            score_train = float(best_row["score_train"]) if "score_train" in lb.columns else None
            score_val = float(best_row["score_val"])
            if score_train is not None:
                overfitting_gap = round(score_train - score_val, 4)
            for row in lb.itertuples():
                leaderboard_entries.append(ModelEntry(
                    model_name=row.model,
                    score_val=row.score_val,
                    fit_time=row.fit_time,
                    pred_time=row.pred_time,
                    stack_level=getattr(row, "stack_level", 1),
                    score_train=getattr(row, "score_train", None),
                ))
        except Exception as e:
            logger.warning("leaderboard(extra_info=True) failed, falling back to basic leaderboard: %s", e)
            try:
                lb = predictor.leaderboard()
                for row in lb.itertuples():
                    leaderboard_entries.append(ModelEntry(
                        model_name=row.model,
                        score_val=row.score_val,
                        fit_time=row.fit_time,
                        pred_time=row.pred_time,
                        stack_level=getattr(row, "stack_level", 1),
                    ))
            except Exception:
                pass

        return RunResult(
            run_id=run_id,
            status="success",
            primary_metric=primary_metric_value,
            leaderboard=leaderboard_entries,
            best_model_name=getattr(predictor, "model_best", None),
            fit_time_seconds=fit_time,
            artifacts_dir=artifacts_dir,
            error=None,
            raw_info={},  # predictor.info() is megabytes of metadata; leaderboard covers what the agent needs
            diagnostics_overfitting_gap=overfitting_gap,
        )

    @staticmethod
    def from_error(error_msg: str) -> RunResult:
        return RunResult(
            status="failed",
            error=error_msg,
        )
