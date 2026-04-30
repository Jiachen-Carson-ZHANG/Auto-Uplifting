import json
from pathlib import Path


NOTEBOOK_PATH = Path("notebooks/bt5153_autonomous_pipeline_demo.ipynb")


def _notebook_source() -> str:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
    )


def test_autonomous_pipeline_demo_notebook_exists_and_uses_real_orchestrator():
    assert NOTEBOOK_PATH.exists()

    source = _notebook_source()

    assert "AutoLiftOrchestrator" in source
    assert "orchestrator.run(max_iterations=MAX_ITERATIONS)" in source
    assert "RetryControllerAgent" in source
    assert "PLANNING_MODEL" in source
    assert "EVALUATION_MODEL" in source
    assert "planner_llm = make_chat_llm(PROVIDER, PLANNING_MODEL, API_KEY)" in source
    assert "evaluation_llm = make_chat_llm(PROVIDER, EVALUATION_MODEL, API_KEY)" in source
    assert "build_feature_table" in source
    assert "generate_submission_artifact" in source


def test_autonomous_pipeline_demo_notebook_keeps_secrets_in_local_env():
    source = _notebook_source()

    assert "load_local_env(ROOT / \".env\")" in source
    assert "OPENAI_API_KEY" in source
    assert "GEMINI_API_KEY" in source
    assert "ANTHROPIC_API_KEY" in source
    assert "sk-" not in source
    assert "AIza" not in source
