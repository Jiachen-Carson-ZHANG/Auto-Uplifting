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


def _is_openai_reasoning_model(model: str) -> bool:
    return model.startswith("o")


def openai_chat_completion_kwargs(model: str, system: str, user: str) -> dict:
    """Build Chat Completions kwargs, respecting reasoning-model constraints."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if not _is_openai_reasoning_model(model):
        kwargs["temperature"] = 0.1
    return kwargs


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
                **openai_chat_completion_kwargs(resolved, system, user)
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


def _parse_stub_payload(user: str):
    try:
        payload = json.loads(user) if user.strip() else {}
    except (AttributeError, json.JSONDecodeError, TypeError):
        return {}
    return payload if isinstance(payload, (dict, list)) else {}


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and value == value


def _record_metric(record: dict) -> float | None:
    for key in ("held_out_qini_auc", "qini_auc", "normalized_qini_auc"):
        value = record.get(key)
        if _is_number(value):
            return float(value)
    return None


def _record_family(record: dict) -> str:
    return str(
        record.get("uplift_learner_family")
        or record.get("learner_family")
        or record.get("family")
        or "two_model"
    )


def _record_estimator(record: dict) -> str:
    return str(record.get("base_estimator") or record.get("estimator") or "gradient_boosting")


def _best_stub_record(records: list[dict]) -> dict | None:
    scored = [record for record in records if _record_metric(record) is not None]
    return max(scored, key=lambda record: _record_metric(record) or float("-inf"), default=None)


def _stub_tuning_search_space(payload: dict) -> dict:
    candidates = payload.get("candidates", [])
    candidates = candidates if isinstance(candidates, list) else []
    search_spaces = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        estimator = _record_estimator(candidate)
        template_name = str(
            candidate.get("template_name")
            or f"{_record_family(candidate)}_{estimator}"
        )
        if estimator == "lightgbm":
            search_space = {
                "n_estimators": [300, 400],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "num_leaves": [7, 15],
            }
            rationale = (
                "Compact LightGBM search around depth, leaves, and learning rate."
            )
        elif estimator == "xgboost":
            search_space = {
                "n_estimators": [300, 400],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "min_child_weight": [10, 20],
            }
            rationale = (
                "Compact XGBoost search around depth and child-weight regularization."
            )
        elif estimator == "gradient_boosting":
            search_space = {
                "n_estimators": [120, 200],
                "learning_rate": [0.03, 0.05],
                "max_depth": [2, 3],
                "min_samples_leaf": [50, 100],
            }
            rationale = "Compact sklearn boosting search around tree size and shrinkage."
        elif estimator == "random_forest":
            search_space = {
                "n_estimators": [200, 300],
                "max_depth": [4, 6],
                "min_samples_leaf": [50, 100],
                "max_features": ["sqrt"],
                "n_jobs": [-1],
            }
            rationale = "Compact forest search around depth and leaf regularization."
        elif estimator == "logistic_regression":
            search_space = {
                "C": [0.1, 0.3, 1.0, 3.0],
                "max_iter": [1000],
            }
            rationale = "Compact logistic search around regularization strength."
        else:
            search_space = {}
            rationale = "No deterministic stub search space for this estimator."
        search_spaces.append(
            {
                "template_name": template_name,
                "rationale": rationale,
                "search_space": search_space,
            }
        )
    return {
        "rationale": (
            "Stub planner proposes bounded deterministic tuning rooms for the "
            "selected internal AutoLift candidates."
        ),
        "search_spaces": search_spaces,
    }


def _summarize_stub_records(records: list[dict]) -> dict:
    successful = [record for record in records if record.get("status") == "success"]
    failed = [record for record in records if record.get("status") == "failed"]
    best = _best_stub_record(successful)
    family_scores: dict[str, list[float]] = {}
    for record in successful:
        metric = _record_metric(record)
        if metric is not None:
            family_scores.setdefault(_record_family(record), []).append(metric)
    best_family = max(
        family_scores,
        key=lambda family: sum(family_scores[family]) / len(family_scores[family]),
        default="two_model",
    )
    similar = sorted(
        [
            {
                "run_id": record.get("run_id"),
                "feature_recipe_id": record.get("feature_recipe_id"),
                "learner_family": _record_family(record),
                "base_estimator": _record_estimator(record),
                "qini_auc": record.get("qini_auc"),
                "held_out_qini_auc": record.get("held_out_qini_auc"),
                "verdict": record.get("verdict"),
            }
            for record in successful
        ],
        key=lambda record: (
            record.get("held_out_qini_auc")
            if _is_number(record.get("held_out_qini_auc"))
            else record.get("qini_auc", float("-inf"))
        ),
        reverse=True,
    )[:5]
    if best is None:
        summary = "Cold start. No prior uplift trials are available."
    else:
        summary = (
            f"{len(successful)} successful prior trial(s). "
            f"Best observed pair: {_record_family(best)} + {_record_estimator(best)} "
            f"with metric={_record_metric(best):.6g}."
        )
    return {
        "similar_recipes": similar,
        "supported_hypotheses": [
            str(record.get("hypothesis_id"))
            for record in successful
            if record.get("verdict") == "supported"
        ],
        "refuted_hypotheses": [
            str(record.get("hypothesis_id"))
            for record in successful
            if record.get("verdict") == "refuted"
        ],
        "best_learner_family": best_family,
        "failed_runs": [
            {"run_id": record.get("run_id"), "error": record.get("error")}
            for record in failed[:5]
        ],
        "summary": summary,
    }


def _stub_chat(system: str, user: str) -> str:
    """Deterministic offline stub for tests and demos without API keys."""
    system_l = system.lower()
    payload = _parse_stub_payload(user)
    if "case retrieval" in system_l:
        records = payload if isinstance(payload, list) else payload.get("records", [])
        records = [record for record in records if isinstance(record, dict)]
        return json.dumps(_summarize_stub_records(records))
    if "feature semantics" in system_l and "uplift strategy" not in system_l:
        approved = payload.get("available_feature_recipes", []) if isinstance(payload, dict) else []
        context_summary = (
            payload.get("context_summary", "").lower() if isinstance(payload, dict) else ""
        )
        prior_records = (
            payload.get("prior_records", []) if isinstance(payload, dict) else []
        )
        record_text = json.dumps(prior_records).lower()
        should_probe_semantics = (
            "human_semantic_v1" in approved
            and ("age_clean" in context_summary or "age_dominance" in record_text)
        )
        return json.dumps(
            {
                "feature_recipe": "human_semantic_v1"
                if should_probe_semantics
                else "rfm_baseline",
                "temporal_policy": "post_issue_history"
                if should_probe_semantics
                else "pre_issue_only",
                "rationale": (
                    "Prior evidence suggests age-heavy XAI, so test richer "
                    "behavioral semantic features."
                    if should_probe_semantics
                    else "Stub mode keeps the deterministic baseline feature recipe."
                ),
                "expected_signal": (
                    "Purchase, points, and tenure features should enter the top XAI drivers."
                    if should_probe_semantics
                    else "Establish a comparable baseline feature surface."
                ),
                "model_family_hints": ["two_model", "class_transformation"],
                "leakage_controls": ["No target/treatment columns", "Respect temporal policy"],
                "xai_sanity_checks": ["Check whether age dominates top features"],
            }
        )
    if "autolift tuning planner" in system_l or "tuning search space" in system_l:
        return json.dumps(
            _stub_tuning_search_space(payload if isinstance(payload, dict) else {})
        )
    if "uplift strategy" in system_l:
        successful_records = []
        unused_pairs = []
        mean_qini = {}
        feature_recipe = "rfm_baseline"
        if isinstance(payload, dict):
            used_pairs = payload.get("used_model_pairs", [])
            unused_pairs = payload.get("unused_model_pairs", [])
            mean_qini = payload.get("mean_qini_by_family", {})
            feature_semantics = payload.get("feature_semantics") or {}
            feature_recipe = feature_semantics.get("feature_recipe") or "rfm_baseline"
            successful_records = used_pairs if isinstance(used_pairs, list) else []
        has_agent_signal = (
            isinstance(mean_qini, dict)
            and any(_is_number(value) and value != 0 for value in mean_qini.values())
        )
        if (
            has_agent_signal
            and successful_records
            and isinstance(unused_pairs, list)
            and unused_pairs
        ):
            pair = unused_pairs[0]
            learner_family = pair[0] if len(pair) > 0 else "two_model"
            base_estimator = pair[1] if len(pair) > 1 else "gradient_boosting"
            rationale = (
                "Reuse ledger evidence and test the next unused available model pair."
            )
        else:
            learner_family = "two_model"
            base_estimator = "gradient_boosting"
            feature_recipe = "rfm_baseline"
            rationale = "Safe evidence-driven default after minimal warmup."
        return json.dumps(
            {
                "learner_family": learner_family,
                "base_estimator": base_estimator,
                "feature_recipe": feature_recipe,
                "split_seed": 42,
                "eval_cutoff": 0.3,
                "rationale": rationale,
            }
        )
    if "trial spec" in system_l:
        strategy = payload.get("strategy", {}) if isinstance(payload, dict) else {}
        estimator_params = (
            payload.get("estimator_params", {}) if isinstance(payload, dict) else {}
        )
        feature_semantics = (
            payload.get("feature_semantics", {}) if isinstance(payload, dict) else {}
        )
        learner_family = strategy.get("learner_family", "two_model")
        base_estimator = strategy.get("base_estimator", "gradient_boosting")
        feature_recipe = strategy.get("feature_recipe", "rfm_baseline")
        semantic_note = feature_semantics.get("rationale") or "No semantic change."
        return json.dumps(
            {
                "hypothesis": "RFM features improve treatment ranking.",
                "changes_from_previous": f"Cold start first trial. {semantic_note}",
                "expected_improvement": feature_semantics.get(
                    "expected_signal",
                    "Establish a baseline uplift ranking.",
                ),
                "model": f"{learner_family} + {base_estimator}",
                "params": estimator_params,
                "feature_recipe": feature_recipe,
                "stop_criteria": "Stop if Qini AUC does not improve after 3 trials.",
            }
        )
    if "evaluation judge" in system_l or "judge" in system_l:
        metrics = payload.get("computed_metrics", {}) if isinstance(payload, dict) else {}
        qini = metrics.get("normalized_qini_auc", metrics.get("qini_auc"))
        uplift_auc = metrics.get("uplift_auc")
        verdict = "supported" if _is_number(qini) and qini >= 0.05 else "inconclusive"
        if _is_number(qini) and qini <= -0.01:
            verdict = "refuted"
        reasoning = (
            f"Stub judge grounded verdict in computed metrics: "
            f"normalized_qini={qini}, uplift_auc={uplift_auc}."
        )
        return json.dumps(
            {
                "verdict": verdict,
                "reasoning": reasoning,
                "champion_comparison": "metric_aware_stub",
                "confidence": "medium" if verdict == "supported" else "low",
                "key_evidence": [
                    f"normalized_qini_auc={qini}",
                    f"uplift_auc={uplift_auc}",
                ],
            }
        )
    if "xai" in system_l or "shap" in system_l:
        shap_result = payload.get("shap_result", {}) if isinstance(payload, dict) else {}
        top_features = (
            shap_result.get("global_top_features")
            or shap_result.get("top_features")
            or []
        )
        leakage_flag = bool(payload.get("leakage_auto_flag")) if isinstance(payload, dict) else False
        diagnostic = (
            payload.get("feature_semantics_diagnostic", {})
            if isinstance(payload, dict)
            else {}
        )
        feature_names = [
            item.get("feature", str(item)) if isinstance(item, dict) else str(item)
            for item in top_features[:5]
        ]
        return json.dumps(
            {
                "top_features": feature_names,
                "stability": "observed" if feature_names else "unknown",
                "business_plausible": not diagnostic.get("age_dominance_warning", False),
                "leakage_detected": leakage_flag,
                "leakage_reason": "automatic leakage signal" if leakage_flag else None,
                "hypothesis_alignment": "mixed",
                "alignment_reason": (
                    "Top features were summarized from deterministic XAI output."
                    if feature_names
                    else "Stub mode had no XAI features."
                ),
                "summary": ", ".join(feature_names) if feature_names else "XAI skipped in stub mode.",
            }
        )
    if "hypothesis reasoning" in system_l or "hypothesis agent" in system_l:
        context = payload.get("retrieved_context", {}) if isinstance(payload, dict) else {}
        latest = payload.get("latest_trial_result") if isinstance(payload, dict) else None
        summary = context.get("summary", "") if isinstance(context, dict) else ""
        if isinstance(latest, dict) and latest.get("qini_auc") is not None:
            evidence = f"Latest trial qini_auc={latest.get('qini_auc')}; {summary}"
            hypothesis = "Try a different learner family or feature recipe to improve held-out uplift ranking."
        else:
            evidence = "Cold start with no prior ledger evidence."
            hypothesis = "RFM features improve treatment ranking."
        return json.dumps(
            {
                "action": "propose",
                "hypothesis": hypothesis,
                "evidence": evidence,
                "confidence": 0.5,
            }
        )
    if "policy" in system_l:
        targeting = (
            payload.get("targeting_results", []) if isinstance(payload, dict) else []
        )
        elbow = payload.get("elbow_threshold_pct", 10) if isinstance(payload, dict) else 10
        best = None
        for result in targeting:
            if not isinstance(result, dict):
                continue
            if best is None or (result.get("roi") or float("-inf")) > (
                best.get("roi") or float("-inf")
            ):
                best = result
        threshold = (
            best.get("threshold_pct", elbow)
            if isinstance(best, dict) and best.get("roi") is not None and best["roi"] > 0
            else elbow
        )
        rationale = (
            f"Use {threshold}% because it is the best positive-ROI cutoff."
            if isinstance(best, dict) and best.get("roi") is not None and best["roi"] > 0
            else f"Use elbow threshold {threshold}% because all ROI options are weak or negative."
        )
        return json.dumps(
            {
                "recommended_threshold": threshold,
                "recommendation_rationale": rationale,
                "operational_verdict": "marginal",
                "verdict_rationale": "Metric-aware stub policy derived from deterministic targeting results.",
                "new_hypothesis": None,
                "summary": rationale,
            }
        )
    return "{}"
