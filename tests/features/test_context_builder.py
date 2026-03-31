"""Tests for src/features/context_builder.py — FeatureContextBuilder."""
import pytest

from src.features.context_builder import FeatureContextBuilder
from src.models.feature_engineering import FeatureHistoryEntry
from src.models.results import DataProfile
from src.models.task import TaskSpec


@pytest.fixture
def task():
    return TaskSpec(
        task_name="churn_prediction",
        task_type="binary",
        data_path="data/churn.csv",
        target_column="churned",
        eval_metric="roc_auc",
        description="Predict customer churn",
    )


@pytest.fixture
def profile():
    return DataProfile(n_rows=1000, n_features=20)


@pytest.fixture
def builder():
    return FeatureContextBuilder()


class TestFeatureContextBuilder:
    def test_build_returns_string(self, builder, task, profile):
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            available_templates=["rfm_recency"], budget_remaining=10, budget_used=2,
        )
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_contains_task_info(self, builder, task, profile):
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            available_templates=[], budget_remaining=10, budget_used=0,
        )
        assert "churn_prediction" in ctx
        assert "binary" in ctx
        assert "churned" in ctx
        assert "roc_auc" in ctx

    def test_contains_profile(self, builder, task, profile):
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            available_templates=[], budget_remaining=10, budget_used=0,
        )
        assert "1000" in ctx
        assert "20" in ctx

    def test_contains_leaderboard(self, builder, task, profile):
        lb = [{"model": "LightGBM", "score_val": 0.85}]
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=lb, feature_importances={},
            history=[], incumbent_metric=0.85,
            available_templates=[], budget_remaining=10, budget_used=0,
        )
        assert "LightGBM" in ctx
        assert "0.85" in ctx

    def test_contains_importances(self, builder, task, profile):
        imps = {"recency_days": 0.25, "frequency": 0.15}
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances=imps,
            history=[], incumbent_metric=None,
            available_templates=[], budget_remaining=10, budget_used=0,
        )
        assert "recency_days" in ctx
        assert "0.25" in ctx

    def test_contains_templates(self, builder, task, profile):
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            available_templates=["rfm_recency", "log1p"],
            budget_remaining=10, budget_used=0,
        )
        assert "rfm_recency" in ctx
        assert "log1p" in ctx

    def test_contains_history(self, builder, task, profile):
        hist = [FeatureHistoryEntry(
            entry_id="h1", action="add",
            feature_spec_json="{}", dataset_name="test",
            task_type="binary", metric_before=0.80, metric_after=0.82,
            observed_outcome="improved AUC",
            distilled_takeaway="RFM features helped",
        )]
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances={},
            history=hist, incumbent_metric=None,
            available_templates=[], budget_remaining=10, budget_used=0,
        )
        assert "improved AUC" in ctx
        assert "RFM features helped" in ctx

    def test_contains_budget(self, builder, task, profile):
        ctx = builder.build(
            task=task, data_profile=profile,
            leaderboard=[], feature_importances={},
            history=[], incumbent_metric=None,
            available_templates=[], budget_remaining=8, budget_used=2,
        )
        assert "8" in ctx
        assert "2" in ctx
