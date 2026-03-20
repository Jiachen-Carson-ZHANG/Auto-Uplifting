"""
Executes a PreprocessingPlan against a CSV file.

Phase 4a: identity — copies data unchanged.
Phase 4b: executes generated code through ValidationHarness (not yet built).

Input : data_path (CSV path), PreprocessingPlan, output_dir (will be created)
Output: path to preprocessed_data.csv inside output_dir
"""
from __future__ import annotations
import shutil
from pathlib import Path
from src.models.preprocessing import PreprocessingPlan


class PreprocessingExecutor:
    """
    Applies a PreprocessingPlan to a CSV file and saves the result.
    Currently only supports strategy="identity".
    """

    def run(self, data_path: str, plan: PreprocessingPlan, output_dir: str) -> str:
        """
        Returns the absolute path to the preprocessed CSV.
        Creates output_dir if it does not exist.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "preprocessed_data.csv"

        if plan.strategy == "identity":
            shutil.copy2(data_path, out_path)
            return str(out_path)

        raise NotImplementedError(
            f"PreprocessingExecutor: strategy '{plan.strategy}' is not yet implemented. "
            f"Phase 4b will add 'generated' strategy support."
        )
