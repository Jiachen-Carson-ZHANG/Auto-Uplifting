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
    """Deterministic offline stub for tests and demos without API keys.

    Branches parse the ``user`` JSON payload so verdicts and strategy choices
    are grounded in the actual trial metrics rather than being fixed strings.
    """
    system_l = system.lower()

    # ------------------------------------------------------------------ #
    # Attempt to parse the user message for metric-aware responses.        #
    # Branches that don't need it ignore _user_payload.                    #
    # ------------------------------------------------------------------ #
    try:
        _user_payload: dict = json.loads(user) if user else {}
    except (json.JSONDecodeError, TypeError):
        _user_payload = {}

    if "case retrieval" in system_l:
        records = _user_payload if isinstance(_user_payload, list) else []
        families = list({r.get("learner_family") for r in records if r.get("learner_family")})
        best = max(records, key=lambda r: r.get("qini_auc") or 0.0, default={})
        return json.dumps(
            {
                "similar_recipes": [r for r in records if r.get("qini_auc")],
                "supported_hypotheses": [r.get("feature_recipe_id", "") for r in records if (r.get("qini_auc") or 0) >= 0.05],
                "refuted_hypotheses": [],
                "best_learner_family": best.get("learner_family", "response_model"),
                "failed_runs": [r for r in records if r.get("status") == "failure"],
                "summary": (
                    f"Retrieved {len(records)} prior trial(s). "
                    f"Families tried: {families or ['none']}. "
                    f"Best Qini so far: {best.get('qini_auc', 'N/A')}."
                ),
            }
        )

    if "uplift strategy" in system_l:
        mean_qini: dict = _user_payload.get("mean_qini_by_family", {})
        best_family: str = _user_payload.get("best_family_so_far", "class_transformation")
        # Determine which estimators have been tried at this best family by
        # inspecting how many unique families have mean_qini entries.
        n_families_tried = len(mean_qini)
        # After 4+ LR warmup trials, escalate to gradient_boosting then xgboost.
        if n_families_tried >= 4:
            tried_note = _user_payload.get("context_summary", "")
            if "xgboost" in tried_note:
                estimator = "lightgbm"
                rationale = "XGBoost already tested; trying LightGBM for speed/depth tradeoff."
            elif "gradient_boosting" in tried_note or "gradient boosting" in tried_note:
                estimator = "xgboost"
                rationale = "Gradient boosting tested; escalating to XGBoost for better tree learning."
            else:
                estimator = "gradient_boosting"
                rationale = "Logistic regression warmup complete; gradient boosting is the natural next escalation."
        else:
            estimator = "logistic_regression"
            rationale = "Warmup phase: establish LR baselines across learner families."
        return json.dumps(
            {
                "learner_family": best_family,
                "base_estimator": estimator,
                "feature_recipe": "rfm_baseline",
                "split_seed": 42,
                "eval_cutoff": 0.3,
                "rationale": rationale,
            }
        )

    if "trial spec" in system_l:
        strategy = _user_payload.get("strategy", {})
        family = strategy.get("learner_family", "class_transformation")
        estimator = strategy.get("base_estimator", "logistic_regression")
        hyp = _user_payload.get("active_hypothesis", "Stronger estimators improve uplift ranking.")
        return json.dumps(
            {
                "hypothesis": hyp,
                "changes_from_previous": f"Switching base estimator to {estimator}.",
                "expected_improvement": "Higher Qini AUC from non-linear decision boundaries.",
                "model": f"{family} + {estimator}",
                "params": _user_payload.get("estimator_params", {"C": 1.0, "max_iter": 1000}),
                "feature_recipe": "rfm_baseline",
                "stop_criteria": "Stop if Qini AUC does not improve by ≥0.01 after 3 consecutive trials.",
            }
        )

    if "evaluation judge" in system_l or "judge" in system_l:
        metrics: dict = _user_payload.get("computed_metrics", {})
        qini = metrics.get("qini_auc")
        uplift_auc = metrics.get("uplift_auc")
        u10 = metrics.get("uplift_at_10pct")
        prior = _user_payload.get("prior_champion")
        trial_status = _user_payload.get("trial_meta", {}).get("trial_status", "success")

        if trial_status != "success" or qini is None:
            return json.dumps({
                "verdict": "inconclusive",
                "reasoning": "Trial did not complete successfully; no metric evidence available.",
                "champion_comparison": "N/A",
                "confidence": "low",
                "key_evidence": ["trial_status != success"],
            })

        # Derive a metric-grounded verdict.
        if qini >= 0.15:
            verdict = "supported"
            confidence = "high"
            reasoning = (
                f"Qini AUC of {qini:.4f} is well above the 0.05 threshold. "
                f"Uplift AUC {uplift_auc:.4f} and top-10% lift {u10:.4f} confirm "
                f"positive treatment heterogeneity. Hypothesis is supported."
            )
        elif qini >= 0.05:
            verdict = "supported"
            confidence = "moderate"
            reasoning = (
                f"Qini AUC of {qini:.4f} exceeds the minimum 0.05 evidence bar. "
                f"The model identifies persuadable customers above random. "
                f"Hypothesis is supported with moderate confidence."
            )
        elif qini > 0:
            verdict = "inconclusive"
            confidence = "low"
            reasoning = (
                f"Qini AUC of {qini:.4f} is weakly positive but below the 0.05 "
                f"significance threshold. More trials needed."
            )
        else:
            verdict = "contradicted"
            confidence = "high"
            reasoning = f"Qini AUC of {qini:.4f} is non-positive; model has no uplift signal."

        # Attach delta vs prior champion when available.
        delta_note = "first_run"
        if isinstance(prior, dict) and prior.get("qini_auc") is not None:
            delta = qini - prior["qini_auc"]
            delta_note = f"delta vs prior champion: {delta:+.4f}"

        return json.dumps({
            "verdict": verdict,
            "reasoning": reasoning,
            "champion_comparison": delta_note,
            "confidence": confidence,
            "key_evidence": [
                f"Qini AUC = {qini:.4f}",
                f"Uplift AUC = {uplift_auc:.4f}" if uplift_auc is not None else "Uplift AUC N/A",
                f"Uplift@10% = {u10:.4f}" if u10 is not None else "Uplift@10% N/A",
                delta_note,
            ],
        })

    if "xai" in system_l or "shap" in system_l:
        shap = _user_payload.get("shap_result", {})
        top = shap.get("global_top_features") or shap.get("top_features") or []
        leakage = _user_payload.get("leakage_auto_flag", False)
        return json.dumps(
            {
                "top_features": top[:5] if top else [],
                "stability": "acceptable",
                "business_plausible": not leakage,
                "leakage_detected": leakage,
                "leakage_reason": "Auto-flagged by deterministic leakage check." if leakage else None,
                "hypothesis_alignment": "aligned" if not leakage else "suspect",
                "alignment_reason": (
                    "Top features are demographic and behavioural signals consistent with "
                    "treatment heterogeneity in retail loyalty programs."
                    if not leakage else
                    "Potential leakage detected; interpret with caution."
                ),
                "summary": (
                    f"Top driver: {top[0].get('feature', 'unknown') if top else 'unknown'}. "
                    f"Leakage flag: {leakage}."
                ),
            }
        )

    if "policy" in system_l:
        targeting = _user_payload.get("targeting_results", [])
        elbow = _user_payload.get("elbow_threshold_pct", 10)
        # Pick the cutoff with best positive ROI; fall back to elbow.
        best_cutoff = elbow
        for t in targeting:
            if isinstance(t.get("roi"), (int, float)) and t["roi"] > 0:
                best_cutoff = t.get("cutoff_pct", elbow)
                break
        return json.dumps(
            {
                "recommended_threshold": best_cutoff,
                "recommendation_rationale": (
                    f"Elbow analysis selects top-{best_cutoff}% as the targeting cutoff "
                    f"that balances incremental lift against coupon cost. "
                    f"Tighter targeting reduces volume but concentrates spend on high-uplift customers."
                ),
                "operational_verdict": "deploy_with_caution",
                "verdict_rationale": (
                    "Positive incremental conversions are achievable but depend on "
                    "coupon cost assumptions. Validate cost model before production."
                ),
                "new_hypothesis": (
                    "Gradient boosting or XGBoost estimators will produce higher Qini AUC "
                    "and sharper uplift stratification than logistic regression."
                ),
                "summary": f"Recommend targeting top-{best_cutoff}% of scored customers.",
            }
        )

    if "hypothesis reasoning" in system_l or "hypothesis agent" in system_l:
        context = _user_payload.get("retrieved_context", {})
        latest = _user_payload.get("latest_trial_result") or {}
        summary = context.get("summary", "")
        best_family = context.get("best_learner_family", "class_transformation")
        latest_qini = latest.get("qini_auc")
        # Escalate hypothesis based on what the context summary reveals.
        if "xgboost" in summary.lower():
            hypothesis = "LightGBM will match XGBoost quality with faster training on this dataset."
            action = "propose"
        elif "gradient" in summary.lower():
            hypothesis = "XGBoost with depth-5 trees will further improve over gradient boosting."
            action = "propose"
        elif latest_qini is not None and latest_qini >= 0.20:
            hypothesis = (
                "Gradient boosting base estimator will improve Qini AUC beyond logistic regression "
                f"for the {best_family} learner family."
            )
            action = "validate" if latest_qini >= 0.22 else "propose"
        else:
            hypothesis = "RFM features with demographic signals improve treatment ranking above random."
            action = "propose"
        return json.dumps(
            {
                "action": action,
                "hypothesis": hypothesis,
                "evidence": summary or f"Best Qini so far: {latest_qini}.",
                "confidence": 0.75 if latest_qini and latest_qini >= 0.20 else 0.5,
                "experiment_action_type": "recipe_comparison",
            }
        )

    return "{}"
