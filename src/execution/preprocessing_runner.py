"""
Executes a PreprocessingPlan against a CSV file.

Strategies
----------
identity  — copy data unchanged (shutil.copy2)
generated — exec() the plan.code, call preprocess(df), save result;
            falls back to identity on any exec/call error

Input : data_path (CSV path), PreprocessingPlan, output_dir (will be created)
Output: path to preprocessed_data.csv inside output_dir
"""
from __future__ import annotations
import logging
import shutil
from pathlib import Path

import pandas as pd

from src.models.preprocessing import PreprocessingPlan

logger = logging.getLogger(__name__)


class PreprocessingExecutor:
    """
    Applies a PreprocessingPlan to a CSV file and saves the result.
    Supports strategy="identity" and strategy="generated".
    Falls back to identity if generated strategy fails at execution time.
    """

    def run(self, data_path: str, plan: PreprocessingPlan, output_dir: str) -> str:
        """
        Returns the absolute path to the preprocessed CSV.
        Creates output_dir if it does not exist.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "preprocessed_data.csv"

        if plan.strategy == "identity" or not plan.code:
            shutil.copy2(data_path, out_path)
            return str(out_path)

        # strategy == "generated"
        try:
            df = pd.read_csv(data_path)
            namespace: dict = {}
            exec(plan.code, namespace)  # noqa: S102
            preprocess_fn = namespace["preprocess"]
            result = preprocess_fn(df.copy())
            result.to_csv(out_path, index=False)
            logger.info("PreprocessingExecutor: generated strategy applied (%d cols, %d rows)", len(result.columns), len(result))
            return str(out_path)
        except Exception as exc:
            logger.warning("PreprocessingExecutor: generated strategy failed, falling back to identity: %s", exc)
            shutil.copy2(data_path, out_path)
            return str(out_path)
