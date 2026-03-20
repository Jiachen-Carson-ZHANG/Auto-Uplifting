from __future__ import annotations
from src.models.task import ExperimentPlan, RunConfig


class ConfigMapper:
    """Translates ExperimentPlan into a RunConfig with AutoGluon-ready kwargs."""

    @staticmethod
    def to_run_config(
        plan: ExperimentPlan,
        data_path: str,
        output_dir: str,
    ) -> RunConfig:
        kwargs: dict = {
            "eval_metric": plan.eval_metric,
            "time_limit": plan.time_limit,
            "presets": plan.presets,
        }

        # Model families → AutoGluon hyperparameters dict
        if plan.model_families:
            kwargs["hyperparameters"] = {family: {} for family in plan.model_families}

        # Validation policy
        val = plan.validation_policy
        if val.get("num_bag_folds", 0) > 0:
            kwargs["num_bag_folds"] = val["num_bag_folds"]
        elif val.get("holdout_frac", 0.0) > 0.0:
            kwargs["holdout_frac"] = val["holdout_frac"]

        # Feature policy
        excluded = plan.feature_policy.get("exclude_columns", [])
        if excluded:
            kwargs["excluded_columns"] = excluded

        # Custom hyperparameter overrides
        if plan.hyperparameters:
            kwargs["hyperparameters"] = plan.hyperparameters

        return RunConfig(
            autogluon_kwargs=kwargs,
            data_path=data_path,
            output_dir=output_dir,
        )
