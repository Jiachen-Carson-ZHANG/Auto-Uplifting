"""Shared LLM factory for all planning and evaluation agents.

Provides two callable types:
    LLMCall  = Callable[[str], str]         -- single-string prompt (used by UpliftAdvisoryPlanner)
    ChatLLM  = Callable[[str, str], str]    -- (system, user) -> reply (used by planning/eval agents)

Usage
-----
    from src.uplift.llm_client import make_chat_llm, make_llm_call

    # Ollama (local, no key needed)
    llm = make_chat_llm("ollama", model="qwen2.5-coder:7b")

    # Gemini free tier
    llm = make_chat_llm("gemini", model="gemini-2.0-flash", api_key="...")

    # Claude
    llm = make_chat_llm("claude", model="claude-haiku-4-5-20251001", api_key="...")

    # Stub (offline smoke-test, echoes prompts)
    llm = make_chat_llm("stub")
"""
from __future__ import annotations

from typing import Callable, Optional

LLMCall = Callable[[str], str]
ChatLLM = Callable[[str, str], str]

_DEFAULTS: dict[str, str] = {
    "ollama":  "qwen2.5-coder:7b",
    "gemini":  "gemini-2.0-flash",
    "claude":  "claude-haiku-4-5-20251001",
    "openai":  "gpt-4o-mini",
    "stub":    "stub",
}


def make_chat_llm(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatLLM:
    """Return a ChatLLM = Callable[[system, user], str]."""
    resolved = model or _DEFAULTS.get(provider, "stub")

    if provider == "stub":
        return _stub_chat

    if provider == "ollama":
        def _ollama(system: str, user: str) -> str:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
            resp = client.chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
            )
            return resp.choices[0].message.content
        return _ollama

    if provider == "openai":
        def _openai(system: str, user: str) -> str:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=resolved,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=0.1,
            )
            return resp.choices[0].message.content
        return _openai

    if provider == "gemini":
        def _gemini(system: str, user: str) -> str:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            m = genai.GenerativeModel(model_name=resolved, system_instruction=system)
            return m.generate_content(user).text
        return _gemini

    if provider == "claude":
        def _claude(system: str, user: str) -> str:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=resolved,
                max_tokens=2048,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
        return _claude

    raise ValueError(f"Unknown provider: {provider!r}. Choose: ollama, openai, gemini, claude, stub")


def make_llm_call(
    provider: str = "ollama",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMCall:
    """Return LLMCall = Callable[[str], str] for UpliftAdvisoryPlanner."""
    chat = make_chat_llm(provider, model, api_key)
    def _call(prompt: str) -> str:
        return chat("", prompt)
    return _call


# ── stub implementation ───────────────────────────────────────────────────────

def _stub_chat(system: str, user: str) -> str:
    """Deterministic offline stub — returns valid JSON for each agent type."""
    import json
    s = system.lower()
    if "case retrieval" in s:
        return json.dumps({
            "similar_recipes": [],
            "supported_hypotheses": [],
            "refuted_hypotheses": [],
            "best_learner_family": "solo_model",
            "failed_runs": [],
            "summary": "Cold start — no prior trial history.",
        })
    if "hypothesis" in s:
        return json.dumps({
            "action": "propose",
            "hypothesis": "RFM features with 90-day recency window improve AUUC",
            "evidence": "Cold start — no prior evidence",
            "confidence": 0.5,
        })
    if "uplift strategy" in s:
        return json.dumps({
            "learner_family": "solo_model",
            "base_estimator": "logistic_regression",
            "feature_recipe": "rfm_baseline",
            "split_seed": 42,
            "eval_cutoff": 0.3,
            "rationale": "Default warm-start selection",
        })
    if "trial spec" in s:
        return json.dumps({
            "hypothesis": "RFM features with 90-day recency window improve AUUC",
            "changes_from_previous": "Cold start — first trial",
            "expected_improvement": "Establish AUUC baseline",
            "model": "solo_model + logistic_regression",
            "params": {},
            "feature_recipe": "rfm_baseline",
            "stop_criteria": "AUUC < 0.50 after 5 consecutive trials",
        })
    if "evaluation judge" in s or "judge" in s:
        return json.dumps({
            "verdict": "inconclusive",
            "reasoning": "Stub response — no live LLM.",
            "champion_comparison": "first_run",
            "confidence": "low",
            "key_evidence": ["stub mode"],
        })
    if "xai" in s or "shap" in s:
        return json.dumps({
            "top_features": [],
            "stability": "unknown",
            "business_plausible": True,
            "leakage_detected": False,
            "leakage_reason": None,
            "hypothesis_alignment": "mixed",
            "alignment_reason": "Stub response.",
            "summary": "XAI stub — no live LLM.",
        })
    if "policy" in s:
        return json.dumps({
            "recommended_threshold": 10,
            "recommendation_rationale": "Stub response.",
            "operational_verdict": "marginal",
            "verdict_rationale": "Stub response.",
            "new_hypothesis": None,
            "summary": "Policy stub — no live LLM.",
        })
    return "{}"
