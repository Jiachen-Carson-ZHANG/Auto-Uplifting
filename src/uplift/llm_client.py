"""Shared LLM call helpers for PR2-style planning and evaluation phases."""
from __future__ import annotations

import json
from typing import Callable

LLMCall = Callable[[str], str]
ChatLLM = Callable[[str, str], str]

_DEFAULTS: dict[str, str] = {
    "ollama": "qwen2.5-coder:7b",
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "claude": "claude-3-5-haiku-latest",
    "stub": "stub",
}


def make_chat_llm(
    provider: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
) -> ChatLLM:
    """Return a `(system, user) -> text` chat callable."""
    resolved = model or _DEFAULTS.get(provider, "stub")

    if provider == "stub":
        return _stub_chat

    if provider == "ollama":
        def _ollama(system: str, user: str) -> str:
            from openai import OpenAI

            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            response = client.chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content or "{}"

        return _ollama

    if provider == "openai":
        def _openai(system: str, user: str) -> str:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.1,
            )
            return response.choices[0].message.content or "{}"

        return _openai

    if provider == "gemini":
        def _gemini(system: str, user: str) -> str:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel(
                model_name=resolved,
                system_instruction=system,
            )
            response = gemini_model.generate_content(user)
            return response.text or "{}"

        return _gemini

    if provider == "claude":
        def _claude(system: str, user: str) -> str:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model=resolved,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return message.content[0].text

        return _claude

    raise ValueError(
        "unknown provider: "
        f"{provider!r}. Choose one of: ollama, openai, gemini, claude, stub"
    )


def make_llm_call(
    provider: str = "ollama",
    model: str | None = None,
    api_key: str | None = None,
) -> LLMCall:
    """Return a single-string prompt callable for older advisory helpers."""
    chat = make_chat_llm(provider, model, api_key)

    def _call(prompt: str) -> str:
        return chat("", prompt)

    return _call


def _stub_chat(system: str, user: str) -> str:
    """Deterministic offline stub for tests and demos without API keys."""
    system_l = system.lower()
    if "case retrieval" in system_l:
        return json.dumps(
            {
                "similar_recipes": [],
                "supported_hypotheses": [],
                "refuted_hypotheses": [],
                "best_learner_family": "response_model",
                "failed_runs": [],
                "summary": "Cold start. No prior uplift trials are available.",
            }
        )
    if "uplift strategy" in system_l:
        return json.dumps(
            {
                "learner_family": "response_model",
                "base_estimator": "logistic_regression",
                "feature_recipe": "rfm_baseline",
                "split_seed": 42,
                "eval_cutoff": 0.3,
                "rationale": "Safe warm-start baseline for the first iteration.",
            }
        )
    if "trial spec" in system_l:
        return json.dumps(
            {
                "hypothesis": "RFM features improve treatment ranking.",
                "changes_from_previous": "Cold start first trial.",
                "expected_improvement": "Establish a baseline uplift ranking.",
                "model": "response_model + logistic_regression",
                "params": {"C": 1.0, "max_iter": 1000},
                "feature_recipe": "rfm_baseline",
                "stop_criteria": "Stop if Qini AUC does not improve after 3 trials.",
            }
        )
    if "evaluation judge" in system_l or "judge" in system_l:
        return json.dumps(
            {
                "verdict": "inconclusive",
                "reasoning": "Stub mode computed metrics but did not use a live judge.",
                "champion_comparison": "first_run",
                "confidence": "low",
                "key_evidence": ["stub mode"],
            }
        )
    if "xai" in system_l or "shap" in system_l:
        return json.dumps(
            {
                "top_features": [],
                "stability": "unknown",
                "business_plausible": True,
                "leakage_detected": False,
                "leakage_reason": None,
                "hypothesis_alignment": "mixed",
                "alignment_reason": "Stub mode.",
                "summary": "XAI skipped in stub mode.",
            }
        )
    if "policy" in system_l:
        return json.dumps(
            {
                "recommended_threshold": 10,
                "recommendation_rationale": "Stub mode default threshold.",
                "operational_verdict": "marginal",
                "verdict_rationale": "Stub mode.",
                "new_hypothesis": None,
                "summary": "Policy summary generated in stub mode.",
            }
        )
    if "hypothesis reasoning" in system_l or "hypothesis agent" in system_l:
        return json.dumps(
            {
                "action": "propose",
                "hypothesis": "RFM features improve treatment ranking.",
                "evidence": "Cold start with no prior ledger evidence.",
                "confidence": 0.5,
            }
        )
    return "{}"
