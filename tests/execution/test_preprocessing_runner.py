import pytest
import pandas as pd
from pathlib import Path
from src.models.preprocessing import PreprocessingPlan
from src.execution.preprocessing_runner import PreprocessingExecutor


def test_identity_copies_file(tmp_path):
    # Write a tiny CSV
    src = tmp_path / "data.csv"
    src.write_text("a,b,label\n1,2,0\n3,4,1\n")

    plan = PreprocessingPlan(strategy="identity")
    executor = PreprocessingExecutor()
    out_path = executor.run(str(src), plan, str(tmp_path / "out"))

    # Output file must exist and have same content
    assert Path(out_path).exists()
    original = pd.read_csv(str(src))
    result = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(original, result)


def test_identity_output_named_preprocessed_data(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,label\n1,0\n")
    plan = PreprocessingPlan(strategy="identity")
    out_path = PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
    assert Path(out_path).name == "preprocessed_data.csv"


def test_generated_strategy_applies_transformation(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("SibSp,Parch,label\n1,0,0\n2,1,1\n")
    code = (
        "def preprocess(df):\n"
        "    import pandas as pd\n"
        "    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n"
        "    return df\n"
    )
    plan = PreprocessingPlan(strategy="generated", code=code)
    out_path = PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
    result = pd.read_csv(out_path)
    assert "FamilySize" in result.columns
    assert result["FamilySize"].tolist() == [2, 4]


def test_generated_strategy_falls_back_to_identity_on_bad_code(tmp_path):
    src = tmp_path / "data.csv"
    src.write_text("a,label\n1,0\n2,1\n")
    bad_code = "def preprocess(df, extra_arg): return df"  # wrong signature
    plan = PreprocessingPlan(strategy="generated", code=bad_code)
    out_path = PreprocessingExecutor().run(str(src), plan, str(tmp_path / "out"))
    # Falls back to identity — output matches original
    original = pd.read_csv(str(src))
    result = pd.read_csv(out_path)
    pd.testing.assert_frame_equal(original, result)
