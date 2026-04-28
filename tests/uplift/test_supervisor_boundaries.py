from pathlib import Path


def test_first_uplift_supervisor_slice_does_not_import_generic_experiment_index():
    uplift_sources = list(Path("src/uplift").glob("*.py"))

    offenders = [
        path
        for path in uplift_sources
        if "src.memory.experiment_index" in path.read_text(encoding="utf-8")
        or "experiment_index" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []
