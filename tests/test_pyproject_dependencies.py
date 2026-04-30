import tomllib
from pathlib import Path


def test_xgboost_and_lightgbm_are_regular_runtime_dependencies():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    dependencies = pyproject["project"]["dependencies"]

    assert "xgboost>=2.0.0" in dependencies
    assert "lightgbm>=4.0.0" in dependencies


def test_requirements_file_matches_runtime_booster_dependencies():
    requirements = Path("requirements.txt").read_text().splitlines()

    assert "xgboost>=2.0.0" in requirements
    assert "lightgbm>=4.0.0" in requirements
