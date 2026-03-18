from __future__ import annotations
import json
from typing import Any
from src.models.results import RunResult, ModelEntry


def _to_json_safe(obj: Any) -> Any:
    """Recursively convert an object to JSON-serializable primitives."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
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
            raw_info=_to_json_safe(predictor.info()) if hasattr(predictor, "info") else {},
        )

    @staticmethod
    def from_error(run_id: str, error_msg: str, artifacts_dir: str) -> RunResult:
        return RunResult(
            run_id=run_id,
            status="failed",
            primary_metric=None,
            leaderboard=[],
            best_model_name=None,
            fit_time_seconds=0.0,
            artifacts_dir=artifacts_dir,
            error=error_msg,
            raw_info={},
        )
