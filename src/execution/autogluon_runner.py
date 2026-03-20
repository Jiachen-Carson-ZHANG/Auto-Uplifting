from __future__ import annotations
import logging
import os
import time
import contextlib
import pandas as pd
from src.models.task import RunConfig
from src.models.results import RunResult
from src.execution.result_parser import ResultParser


@contextlib.contextmanager
def _log_to_file(log_path: str):
    """Temporarily add a FileHandler to the autogluon logger and suppress stdout handlers."""
    ag_logger = logging.getLogger("autogluon")
    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    # Suppress stdout handlers on autogluon logger during fit
    prev_handlers = ag_logger.handlers[:]
    for h in prev_handlers:
        ag_logger.removeHandler(h)
    ag_logger.addHandler(file_handler)
    try:
        yield
    finally:
        ag_logger.removeHandler(file_handler)
        file_handler.close()
        for h in prev_handlers:
            ag_logger.addHandler(h)


class AutoGluonRunner:
    """Runs AutoGluon TabularPredictor for a given RunConfig."""

    def __init__(self, target_column: str) -> None:
        self.target_column = target_column

    def run(self, config: RunConfig) -> RunResult:
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            return ResultParser.from_error(
                "AutoGluon not installed. Run: pip install autogluon.tabular"
            )

        os.makedirs(config.output_dir, exist_ok=True)
        df = pd.read_csv(config.data_path)
        log_path = os.path.join(config.output_dir, "training.log")

        kwargs = dict(config.autogluon_kwargs)
        kwargs["path"] = config.output_dir

        predictor = TabularPredictor(
            label=self.target_column,
            verbosity=1,  # minimal stdout; full details go to training.log
            **{k: v for k, v in kwargs.items()
               if k in ("eval_metric", "path", "problem_type")}
        )

        fit_kwargs = {k: v for k, v in kwargs.items()
                      if k not in ("eval_metric", "path", "problem_type")}

        start = time.time()
        try:
            with _log_to_file(log_path):
                predictor.fit(df, **fit_kwargs)
        except Exception as e:
            return ResultParser.from_error(str(e))
        fit_time = time.time() - start

        # Get validation score from leaderboard
        lb = predictor.leaderboard(silent=True)
        primary_metric = float(lb["score_val"].max()) if not lb.empty else 0.0

        return ResultParser.from_predictor(
            predictor=predictor,
            run_id=config.run_id,
            fit_time=fit_time,
            artifacts_dir=config.output_dir,
            primary_metric_value=primary_metric,
        )
