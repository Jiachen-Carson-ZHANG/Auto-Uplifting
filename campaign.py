#!/usr/bin/env python3
"""
Campaign entrypoint — runs multiple ExperimentSessions via CampaignOrchestrator.

Usage:
    python3 campaign.py
    python3 campaign.py --config configs/project.yaml
"""
import argparse
import os
import yaml

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.models.task import TaskSpec
from src.models.campaign import CampaignConfig
from src.llm.backend import create_backend
from src.orchestration.campaign import CampaignOrchestrator


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Agentic ML Campaign — multi-session optimization")
    parser.add_argument("--config", default="configs/project.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    search_config = load_config("configs/search.yaml")

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

    llm_cfg = config["llm"]
    provider = llm_cfg["provider"]
    model = llm_cfg["model"]
    api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        print(f"Warning: {api_key_env} not set. LLM calls will fail.")

    llm = create_backend(provider=provider, model=model, api_key=api_key)

    search = search_config.get("search", {})
    num_candidates = search.get("num_candidates", 3)
    max_optimize = search.get("max_optimize_iterations", 5)
    eval_metric = task_cfg.get("eval_metric", "roc_auc")
    higher_is_better = search.get("higher_is_better", {}).get(eval_metric, True)

    campaign_cfg = config.get("campaign", {})
    campaign_config = CampaignConfig(
        max_sessions=campaign_cfg.get("max_sessions", 5),
        plateau_threshold=campaign_cfg.get("plateau_threshold", 0.002),
        plateau_window=campaign_cfg.get("plateau_window", 3),
        preprocessing_bank_path=campaign_cfg.get(
            "preprocessing_bank_path", "experiments/preprocessing_bank.jsonl"
        ),
    )

    orchestrator = CampaignOrchestrator(
        task=task,
        llm=llm,
        config=campaign_config,
        experiments_dir=config["session"]["experiments_dir"],
        num_candidates=num_candidates,
        max_optimize_iterations=max_optimize,
        higher_is_better=higher_is_better,
        case_store_path=config["session"].get("case_store_path"),
    )

    result = orchestrator.run()
    best_str = f"{result.best_metric:.4f}" if result.best_metric is not None else "N/A"
    print(f"\nCampaign complete: best={best_str} | sessions={len(result.sessions)} | stopped={result.stopped_reason}")


if __name__ == "__main__":
    main()
