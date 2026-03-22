"""Tests for ValidationHarness — subprocess-isolated preprocess(df) validation."""
from __future__ import annotations
import os
import tempfile
import pytest
import pandas as pd
from src.execution.validation_harness import ValidationHarness, ValidationResult


@pytest.fixture()
def titanic_csv(tmp_path):
    """Minimal Titanic-like CSV with 10 rows."""
    df = pd.DataFrame({
        "PassengerId": range(1, 11),
        "Survived": [0, 1, 1, 0, 1, 0, 0, 1, 1, 0],
        "Pclass": [3, 1, 3, 1, 3, 3, 1, 3, 2, 3],
        "Name": [
            "Braund, Mr. Owen Harris",
            "Cumings, Mrs. John Bradley",
            "Heikkinen, Miss. Laina",
            "Futrelle, Mrs. Jacques Heath",
            "Allen, Mr. William Henry",
            "Moran, Mr. James",
            "McCarthy, Mr. Timothy J",
            "Palsson, Master. Gosta Leonard",
            "Johnson, Mrs. Oscar W",
            "Nasser, Mrs. Nicholas",
        ],
        "Age": [22.0, 38.0, 26.0, 35.0, 35.0, None, 54.0, 2.0, 27.0, 14.0],
        "SibSp": [1, 1, 0, 1, 0, 0, 0, 3, 0, 1],
        "Parch": [0, 0, 0, 0, 0, 0, 0, 1, 2, 0],
    })
    path = tmp_path / "titanic.csv"
    df.to_csv(path, index=False)
    return str(path)


def harness():
    return ValidationHarness(timeout=30)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_valid_code_passes(titanic_csv):
    code = """
def preprocess(df):
    import pandas as pd
    df['Title'] = df['Name'].str.extract(r', (\\w+)\\.', expand=False)
    return df
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is True
    assert result.error is None


def test_family_size_feature_passes(titanic_csv):
    code = """
def preprocess(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    return df
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is True


# ── Fail: preprocess not defined ──────────────────────────────────────────────

def test_preprocess_not_defined_fails(titanic_csv):
    code = "x = 1  # no preprocess function"
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert result.error is not None


# ── Fail: returns non-DataFrame ───────────────────────────────────────────────

def test_returns_non_dataframe_fails(titanic_csv):
    code = """
def preprocess(df):
    return {"not": "a dataframe"}
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "DataFrame" in result.error


# ── Fail: drops too many rows ────────────────────────────────────────────────

def test_drops_too_many_rows_fails(titanic_csv):
    code = """
def preprocess(df):
    return df.head(2)  # keeps only 2 of 10 rows — below 50% threshold
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "shape check" in result.error


# ── Fail: drops target column ────────────────────────────────────────────────

def test_drops_target_column_fails(titanic_csv):
    code = """
def preprocess(df):
    return df.drop(columns=['Survived'])
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "Survived" in result.error


# ── Fail: NaN in target column ───────────────────────────────────────────────

def test_nan_in_target_fails(titanic_csv):
    code = """
def preprocess(df):
    import pandas as pd
    df['Survived'] = df['Survived'].astype(float)
    df.loc[0, 'Survived'] = float('nan')
    return df
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "NaN" in result.error


# ── Fail: identity transform (diff check) ────────────────────────────────────

def test_identity_transform_fails_diff_check(titanic_csv):
    code = """
def preprocess(df):
    return df  # no-op: same columns, same values
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "identity" in result.error or "diff check" in result.error


# ── Pass: dtype-only change passes diff check ────────────────────────────────

def test_dtype_change_passes_diff_check(titanic_csv):
    code = """
def preprocess(df):
    import pandas as pd
    df['Pclass'] = df['Pclass'].astype(str)
    return df
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is True


# ── Fail: exception in code ───────────────────────────────────────────────────

def test_exception_in_code_fails(titanic_csv):
    code = """
def preprocess(df):
    raise ValueError("intentional error")
"""
    result = harness().validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "intentional error" in result.error


# ── Timeout ───────────────────────────────────────────────────────────────────

def test_timeout_returns_failed_result(titanic_csv):
    code = """
def preprocess(df):
    import time
    time.sleep(999)
    return df
"""
    h = ValidationHarness(timeout=2)
    result = h.validate(code, titanic_csv, "Survived")
    assert result.passed is False
    assert "timed out" in result.error
