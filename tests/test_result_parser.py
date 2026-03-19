import pytest
from unittest.mock import MagicMock
import pandas as pd
from src.execution.result_parser import ResultParser


def _make_predictor(val_scores, train_scores, best_model="WeightedEnsemble_L2"):
    lb_basic = pd.DataFrame({
        "model": ["WeightedEnsemble_L2", "GBM"],
        "score_val": val_scores,
        "fit_time": [10.0, 8.0],
        "pred_time": [0.1, 0.1],
        "stack_level": [2, 1],
    })
    lb_extra = lb_basic.copy()
    lb_extra["score_train"] = train_scores
    p = MagicMock()
    p.leaderboard.side_effect = lambda extra_info=False: lb_extra if extra_info else lb_basic
    p.model_best = best_model
    return p


def test_overfitting_gap_computed():
    predictor = _make_predictor(
        val_scores=[0.87, 0.85],
        train_scores=[0.95, 0.93],
    )
    result = ResultParser.from_predictor(predictor, "run_0001", 10.0, "/tmp", 0.87)
    assert result.leaderboard[0].score_train == pytest.approx(0.95)
    assert result.diagnostics_overfitting_gap == pytest.approx(0.95 - 0.87)


def test_overfitting_gap_none_when_extra_info_fails():
    """extra_info=True fails → fallback to basic leaderboard → gap is None but entries populated."""
    lb_basic = pd.DataFrame({
        "model": ["WeightedEnsemble_L2"],
        "score_val": [0.87],
        "fit_time": [10.0],
        "pred_time": [0.1],
        "stack_level": [2],
    })
    p = MagicMock()
    p.model_best = "WeightedEnsemble_L2"
    def leaderboard_side_effect(extra_info=False):
        if extra_info:
            raise ValueError("extra_info not supported")
        return lb_basic
    p.leaderboard.side_effect = leaderboard_side_effect
    result = ResultParser.from_predictor(p, "run_0001", 10.0, "/tmp", 0.87)
    assert result.diagnostics_overfitting_gap is None
    assert len(result.leaderboard) == 1          # basic leaderboard still works
    assert result.leaderboard[0].score_train is None  # no train score in fallback
