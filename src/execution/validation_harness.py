"""
Validates generated preprocess(df) code by running it in a subprocess.

Subprocess isolation prevents crashes and imports from affecting the main process.

6-check validation pipeline:
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. No exception raised in subprocess                           │
  │  2. Returns a DataFrame                                         │
  │  3. len(result) >= 0.5 * len(original)  ← shape check          │
  │  4. target_col in result.columns        ← target preserved      │
  │  5. result[target_col].isnull().sum() == 0  ← no NaN in target  │
  │  6. columns differ OR values differ     ← not identity          │
  └─────────────────────────────────────────────────────────────────┘

Input : code (str), data_path (str), target_col (str), timeout (int)
Output: ValidationResult(passed, error)
"""
from __future__ import annotations
import json
import logging
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_RUNNER_TEMPLATE = textwrap.dedent("""
import json
import sys
import pandas as pd

data_path = {data_path!r}
target_col = {target_col!r}
original_cols = {original_cols!r}
original_len = {original_len!r}

{code}

try:
    df_orig = pd.read_csv(data_path)
    result = preprocess(df_orig.copy())

    if not isinstance(result, pd.DataFrame):
        print(json.dumps({{"ok": False, "error": f"preprocess() returned {{type(result).__name__}}, expected DataFrame"}}))
        sys.exit(0)

    if len(result) < 0.5 * original_len:
        pct = round(100 * len(result) / original_len, 1)
        print(json.dumps({{"ok": False, "error": f"shape check failed: result has {{len(result)}} rows ({{pct}}% of original {{original_len}})"}}))
        sys.exit(0)

    if target_col not in result.columns:
        print(json.dumps({{"ok": False, "error": f"target column '{{target_col}}' missing from result"}}))
        sys.exit(0)

    if result[target_col].isnull().sum() > 0:
        n = int(result[target_col].isnull().sum())
        print(json.dumps({{"ok": False, "error": f"target column '{{target_col}}' has {{n}} NaN values"}}))
        sys.exit(0)

    # Diff check: must change columns or values
    cols_changed = set(result.columns) != set(original_cols)
    if not cols_changed:
        try:
            common = [c for c in original_cols if c in result.columns]
            values_changed = not df_orig[common].reset_index(drop=True).equals(result[common].reset_index(drop=True))
        except Exception:
            values_changed = True
    else:
        values_changed = True

    if not cols_changed and not values_changed:
        print(json.dumps({{"ok": False, "error": "diff check failed: preprocess() returned identical data (identity transform)"}}))
        sys.exit(0)

    print(json.dumps({{"ok": True, "error": None, "n_cols": len(result.columns), "n_rows": len(result)}}))

except Exception as exc:
    print(json.dumps({{"ok": False, "error": str(exc)}}))
""")


@dataclass
class ValidationResult:
    passed: bool
    error: Optional[str] = None


class ValidationHarness:
    """Runs generated preprocess(df) in a subprocess and validates the result."""

    def __init__(self, timeout: int = 30) -> None:
        self._timeout = timeout

    def validate(self, code: str, data_path: str, target_col: str) -> ValidationResult:
        """
        Execute code in a subprocess. Return ValidationResult.
        Never raises — all failures return passed=False with an error message.
        """
        import pandas as pd
        try:
            df_orig = pd.read_csv(data_path, nrows=10000)
            original_cols = list(df_orig.columns)
            original_len = len(pd.read_csv(data_path))
        except Exception as exc:
            return ValidationResult(passed=False, error=f"Could not read data_path: {exc}")

        script = _RUNNER_TEMPLATE.format(
            data_path=data_path,
            target_col=target_col,
            original_cols=original_cols,
            original_len=original_len,
            code=code,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            logger.warning("ValidationHarness: subprocess timed out after %ds", self._timeout)
            return ValidationResult(passed=False, error=f"subprocess timed out after {self._timeout}s")
        finally:
            import os
            try:
                os.unlink(script_path)
            except OSError:
                pass

        if proc.returncode != 0 and not proc.stdout.strip():
            stderr = proc.stderr.strip()[:500]
            logger.warning("ValidationHarness: subprocess crashed: %s", stderr)
            return ValidationResult(passed=False, error=f"subprocess crashed: {stderr}")

        stdout = proc.stdout.strip()
        if not stdout:
            return ValidationResult(passed=False, error="subprocess produced no output")

        try:
            result = json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            return ValidationResult(passed=False, error=f"could not parse subprocess output: {exc}")

        if result.get("ok"):
            logger.debug(
                "ValidationHarness: PASS — %d cols, %d rows",
                result.get("n_cols", 0), result.get("n_rows", 0),
            )
            return ValidationResult(passed=True)
        else:
            error = result.get("error", "unknown error")
            logger.debug("ValidationHarness: FAIL — %s", error)
            return ValidationResult(passed=False, error=error)
