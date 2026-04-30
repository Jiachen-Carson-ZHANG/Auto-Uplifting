import json

from src.uplift.llm_client import make_chat_llm, openai_chat_completion_kwargs


def test_openai_chat_completion_kwargs_omits_temperature_for_reasoning_models():
    kwargs = openai_chat_completion_kwargs("o3", "system", "user")

    assert kwargs["model"] == "o3"
    assert "temperature" not in kwargs


def test_openai_chat_completion_kwargs_keeps_low_temperature_for_non_reasoning_models():
    kwargs = openai_chat_completion_kwargs("gpt-4o-mini", "system", "user")

    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.1


def test_stub_case_retrieval_summarizes_prior_records():
    llm = make_chat_llm("stub")

    response = llm(
        "case retrieval",
        json.dumps(
            [
                {
                    "run_id": "RUN-a",
                    "status": "success",
                    "uplift_learner_family": "two_model",
                    "base_estimator": "gradient_boosting",
                    "qini_auc": 0.21,
                    "held_out_qini_auc": 0.24,
                    "verdict": "supported",
                },
                {
                    "run_id": "RUN-b",
                    "status": "failed",
                    "error": "bad params",
                },
            ]
        ),
    )

    payload = json.loads(response)
    assert payload["best_learner_family"] == "two_model"
    assert payload["similar_recipes"][0]["run_id"] == "RUN-a"
    assert payload["failed_runs"][0]["run_id"] == "RUN-b"
    assert "1 successful prior trial" in payload["summary"]


def test_stub_judge_uses_computed_metrics():
    llm = make_chat_llm("stub")

    response = llm(
        "evaluation judge",
        json.dumps(
            {
                "computed_metrics": {
                    "normalized_qini_auc": 0.19,
                    "uplift_auc": 0.045,
                }
            }
        ),
    )

    payload = json.loads(response)
    assert payload["verdict"] == "supported"
    assert "normalized_qini_auc=0.19" in payload["key_evidence"]


def test_stub_xai_summarizes_top_features():
    llm = make_chat_llm("stub")

    response = llm(
        "xai reasoning",
        json.dumps(
            {
                "shap_result": {
                    "global_top_features": [
                        {"feature": "age_clean"},
                        {"feature": "purchase_sum_90d"},
                    ]
                },
                "leakage_auto_flag": False,
            }
        ),
    )

    payload = json.loads(response)
    assert payload["top_features"] == ["age_clean", "purchase_sum_90d"]
    assert payload["summary"] == "age_clean, purchase_sum_90d"


def test_stub_policy_uses_deterministic_elbow_when_roi_is_negative():
    llm = make_chat_llm("stub")

    response = llm(
        "policy simulation",
        json.dumps(
            {
                "elbow_threshold_pct": 20,
                "targeting_results": [
                    {"threshold_pct": 5, "roi": -0.5},
                    {"threshold_pct": 20, "roi": -0.3},
                ],
            }
        ),
    )

    payload = json.loads(response)
    assert payload["recommended_threshold"] == 20
    assert "elbow threshold 20%" in payload["recommendation_rationale"]


def test_stub_feature_semantics_probes_human_features_when_age_dominates():
    llm = make_chat_llm("stub")

    response = llm(
        "feature semantics",
        json.dumps(
            {
                "available_feature_recipes": ["rfm_baseline", "human_semantic_v1"],
                "context_summary": "age_clean dominates prior XAI",
                "prior_records": [],
            }
        ),
    )

    payload = json.loads(response)
    assert payload["feature_recipe"] == "human_semantic_v1"
    assert payload["temporal_policy"] == "post_issue_history"
