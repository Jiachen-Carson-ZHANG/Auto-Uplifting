#!/usr/bin/env python3
"""
Hybrid Agentic ML Framework — Main Entrypoint

Usage:
    python3 main.py
    python3 main.py --config configs/project.yaml
"""
import argparse
import json
import os
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on env vars set externally

from src.models.task import TaskSpec
from src.llm.backend import create_backend
from src.session import ExperimentSession


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_seed_ideas(path: str = "configs/seed_ideas.json") -> list:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def main():
    parser = argparse.ArgumentParser(description="Hybrid Agentic ML Framework")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    search_config = load_config("configs/search.yaml")

    # Task
    task_cfg = config["task"]
    task = TaskSpec(
        task_name=task_cfg["name"],
        task_type=task_cfg["type"],
        data_path=task_cfg["data_path"],
        target_column=task_cfg["target_column"],
        eval_metric=task_cfg["eval_metric"],
        constraints=task_cfg.get("constraints", {}),
        description=task_cfg.get("description", ""),
    )

    # LLM
    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = llm_cfg["model"]
    api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"Warning: {api_key_env} not set. LLM calls will fail.")

    llm = create_backend(provider=provider, model=model, api_key=api_key)

    # Search config
    search = search_config.get("search", {})
    num_candidates = search.get("num_candidates", 3)
    max_optimize = search.get("max_optimize_iterations", 5)
    eval_metric = task_cfg.get("eval_metric", "roc_auc")
    higher_is_better = search.get("higher_is_better", {}).get(eval_metric, True)

    # Session
    seed_ideas = load_seed_ideas()
    session = ExperimentSession(
        task=task,
        llm=llm,
        experiments_dir=config["session"]["experiments_dir"],
        num_candidates=num_candidates,
        max_optimize_iterations=max_optimize,
        higher_is_better=higher_is_better,
        seed_ideas=seed_ideas,
    )

    session.run()


if __name__ == "__main__":
    main()
