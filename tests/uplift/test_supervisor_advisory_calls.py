import pytest
from pydantic import ValidationError

from src.models.uplift import (
    UpliftAdvisoryReport,
    UpliftAdvisoryVerdict,
    UpliftDiagnosisResult,
    UpliftExperimentRecord,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftStopDecision,
    UpliftTableSchema,
)
from src.uplift.supervisor import (
    build_diagnosis_prompt,
    build_report_prompt,
    build_verdict_prompt,
    build_wave_planning_prompt,
    diagnosis_call,
    report_call,
    verdict_call,
    wave_planning_call,
)
from src.uplift.supervisor.advisory import _cap_prompt


def _contract() -> UpliftProjectContract:
    return UpliftProjectContract(
        task_name="retailhero-uplift",
        table_schema=UpliftTableSchema(
            clients_table="clients.csv",
            purchases_table="purchases.csv",
            train_table="uplift_train.csv",
            scoring_table="uplift_test.csv",
            products_table="products.csv",
        ),
    )


def _feature_artifact(recipe_id: str) -> UpliftFeatureArtifact:
    return UpliftFeatureArtifact(
        feature_recipe_id=recipe_id,
        feature_artifact_id=f"artifact-{recipe_id}",
        dataset_fingerprint="dataset123",
        builder_version="v1",
        artifact_path=f"features/{recipe_id}.csv",
        metadata_path=f"features/{recipe_id}.metadata.json",
        row_count=8,
        columns=["client_id", "feature"],
        generated_columns=["feature"],
        source_tables=["clients", "purchases"],
        feature_groups=["basket", "demographic", "points", "rfm"],
        windows_days=[30],
    )


def _record(run_id: str = "RUN-a") -> UpliftExperimentRecord:
    return UpliftExperimentRecord(
        run_id=run_id,
        hypothesis_id="UH-1",
        feature_recipe_id="recipe-a",
        feature_artifact_id="artifact-a",
        template_name="random_baseline",
        uplift_learner_family="random",
        base_estimator="none",
        params_hash="params-a",
        split_seed=42,
        status="success",
        qini_auc=0.12,
        uplift_auc=0.08,
        policy_gain={"top_50pct_zero_cost": 1.5},
        artifact_paths={"predictions": "runs/RUN-a/predictions.csv"},
    )


def _decision() -> UpliftStopDecision:
    return UpliftStopDecision(
        wave_id="UW-1",
        hypothesis_id="UH-1",
        action_type="recipe_comparison",
        stop_reason="business_decision_supportable",
        hypothesis_status="supported",
        should_stop=True,
        trial_ids=["RUN-a", "RUN-b"],
        champion_run_id="RUN-a",
        next_action="stop",
        evidence_summary={
            "champion_metric_name": "qini_auc",
            "champion_metric": 0.12,
            "best_policy_gain": 1.5,
            "cost_sensitivity": {"top_50pct_zero_cost": 1.5},
            "response_overlap": None,
        },
        artifact_paths={
            "ledger": "runs/uplift_ledger.jsonl",
            "RUN-a:predictions": "runs/RUN-a/predictions.csv",
        },
    )


def _diagnosis_payload(extra: str = "") -> str:
    return (
        "{"
        '"unresolved_questions":["Which recipe should we test next?"],'
        '"risks":["Tiny validation split may be unstable."],'
        '"candidate_hypotheses":[{'
        '"question":"Does the purchase-window recipe help?",'
        '"hypothesis_text":"Windowed purchase features improve uplift ranking.",'
        '"action_type":"recipe_comparison",'
        '"expected_signal":"Higher validation qini_auc.",'
        '"rationale":"Prior baselines leave feature uncertainty."'
        "}]"
        f"{extra}"
        "}"
    )


def _wave_payload(
    *,
    template_name: str = "random_baseline",
    split_seed: int = 42,
    primary_metric: str = "qini_auc",
    extra: str = "",
) -> str:
    return (
        "{"
        '"wave_id":"UW-llm-1",'
        '"hypothesis_id":"UH-1",'
        '"action_type":"recipe_comparison",'
        '"rationale":"Compare two approved cached recipes.",'
        '"trial_specs":['
        "{"
        '"spec_id":"UT-a",'
        '"hypothesis_id":"UH-1",'
        f'"template_name":"{template_name}",'
        '"learner_family":"random",'
        '"base_estimator":"none",'
        '"feature_recipe_id":"recipe-a",'
        '"params":{},'
        f'"split_seed":{split_seed},'
        f'"primary_metric":"{primary_metric}"'
        "},"
        "{"
        '"spec_id":"UT-b",'
        '"hypothesis_id":"UH-1",'
        f'"template_name":"{template_name}",'
        '"learner_family":"random",'
        '"base_estimator":"none",'
        '"feature_recipe_id":"recipe-b",'
        '"params":{},'
        f'"split_seed":{split_seed},'
        f'"primary_metric":"{primary_metric}"'
        "}"
        "],"
        '"expected_signal":"One cached recipe improves qini_auc.",'
        '"success_criterion":"Champion is selected from successful trial records.",'
        '"abort_on_first_failure":true,'
        '"required_feature_recipe_ids":["recipe-a","recipe-b"],'
        '"created_by":"llm"'
        f"{extra}"
        "}"
    )


def test_advisory_models_reject_extra_metric_or_contract_fields():
    with pytest.raises(ValidationError):
        UpliftDiagnosisResult.model_validate_json(
            _diagnosis_payload(extra=',"target_column":"bad_target"')
        )

    with pytest.raises(ValidationError):
        UpliftAdvisoryVerdict(
            stop_reason="business_decision_supportable",
            hypothesis_status="supported",
            verdict_summary="Supported.",
            rationale="Grounded in deterministic stop decision.",
            next_action="stop",
            cited_artifact_paths=["runs/uplift_ledger.jsonl"],
            qini_auc=999,  # type: ignore[call-arg]
        )

    with pytest.raises(ValidationError):
        UpliftAdvisoryReport(
            title="Uplift summary",
            executive_summary="Supported by deterministic evidence.",
            validation_summary="Validation metric is cited.",
            held_out_summary="Held-out status is cited.",
            scoring_summary="Scoring output is separate.",
            limitations=["Tiny fixture."],
            cited_artifact_paths=["runs/uplift_ledger.jsonl"],
            uplift_auc=999,  # type: ignore[call-arg]
        )


def test_diagnosis_prompt_marks_contract_fields_immutable():
    prompt = build_diagnosis_prompt(
        _contract(),
        records=[_record()],
        valid_actions=["recipe_comparison", "window_sweep"],
    )

    assert "Treatment column: treatment_flg" in prompt
    assert "Target column: target" in prompt
    assert "Primary metric: qini_auc" in prompt
    assert "immutable" in prompt
    assert "Allowed actions" in prompt


def test_diagnosis_call_retries_invalid_json_and_returns_strict_result():
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        if len(calls) == 1:
            return "not json"
        return _diagnosis_payload()

    result = diagnosis_call(
        fake_llm,
        _contract(),
        records=[_record()],
        valid_actions=["recipe_comparison"],
        max_retries=1,
    )

    assert len(calls) == 2
    assert result.candidate_hypotheses[0].action_type == "recipe_comparison"


def test_diagnosis_call_rejects_contract_mutation_payload():
    with pytest.raises(ValueError, match="invalid diagnosis JSON"):
        diagnosis_call(
            lambda prompt: _diagnosis_payload(extra=',"target_column":"bad_target"'),
            _contract(),
            records=[_record()],
            valid_actions=["recipe_comparison"],
            max_retries=0,
        )


def test_wave_planning_prompt_lists_allowed_artifacts_and_immutable_contract():
    artifacts = {
        "recipe-a": _feature_artifact("recipe-a"),
        "recipe-b": _feature_artifact("recipe-b"),
    }

    prompt = build_wave_planning_prompt(
        _contract(),
        diagnosis=UpliftDiagnosisResult.model_validate_json(_diagnosis_payload()),
        feature_artifacts=artifacts,
        known_hypothesis_ids=["UH-1"],
    )

    assert "Allowed feature recipe IDs" in prompt
    assert "recipe-a" in prompt and "recipe-b" in prompt
    assert "Allowed templates" in prompt
    assert "Do not change target, treatment, split, or metric" in prompt


def test_wave_planning_call_returns_valid_recipe_comparison_wave():
    artifacts = {
        "recipe-a": _feature_artifact("recipe-a"),
        "recipe-b": _feature_artifact("recipe-b"),
    }

    wave = wave_planning_call(
        lambda prompt: _wave_payload(),
        _contract(),
        diagnosis=UpliftDiagnosisResult.model_validate_json(_diagnosis_payload()),
        feature_artifacts=artifacts,
        known_hypothesis_ids=["UH-1"],
        max_retries=0,
    )

    assert wave.wave_id == "UW-llm-1"
    assert wave.created_by == "llm"
    assert wave.trial_specs[0].split_seed == _contract().split_contract.random_seed
    assert wave.trial_specs[0].primary_metric == _contract().evaluation_policy.primary_metric


def test_wave_planning_call_rejects_contract_mutation_payload():
    artifacts = {
        "recipe-a": _feature_artifact("recipe-a"),
        "recipe-b": _feature_artifact("recipe-b"),
    }

    with pytest.raises(ValueError, match="contract mutation"):
        wave_planning_call(
            lambda prompt: _wave_payload(extra=',"target_column":"bad_target"'),
            _contract(),
            diagnosis=UpliftDiagnosisResult.model_validate_json(_diagnosis_payload()),
            feature_artifacts=artifacts,
            known_hypothesis_ids=["UH-1"],
            max_retries=0,
        )


def test_wave_planning_call_rejects_invented_template_and_metric_mutation():
    artifacts = {
        "recipe-a": _feature_artifact("recipe-a"),
        "recipe-b": _feature_artifact("recipe-b"),
    }

    with pytest.raises(ValueError, match="unknown uplift template"):
        wave_planning_call(
            lambda prompt: _wave_payload(template_name="invented_template"),
            _contract(),
            diagnosis=UpliftDiagnosisResult.model_validate_json(_diagnosis_payload()),
            feature_artifacts=artifacts,
            known_hypothesis_ids=["UH-1"],
            max_retries=0,
        )

    with pytest.raises(ValueError, match="metric"):
        wave_planning_call(
            lambda prompt: _wave_payload(primary_metric="made_up_metric"),
            _contract(),
            diagnosis=UpliftDiagnosisResult.model_validate_json(_diagnosis_payload()),
            feature_artifacts=artifacts,
            known_hypothesis_ids=["UH-1"],
            max_retries=0,
        )


def test_prompt_cap_uses_conservative_word_proxy_for_token_budget():
    prompt = " ".join(f"word{i}" for i in range(200))

    capped = _cap_prompt(prompt, max_estimated_tokens=100)

    assert len(capped.split()) <= 75


def test_verdict_prompt_is_bounded_and_grounded_in_stop_decision():
    prompt = build_verdict_prompt(_decision(), records=[_record()])

    assert len(prompt.split()) <= 8000
    assert "business_decision_supportable" in prompt
    assert "Numeric metrics are deterministic evidence" in prompt


def test_verdict_call_rejects_metric_mutation_and_returns_consistent_result():
    decision = _decision()

    with pytest.raises(ValueError, match="invalid verdict JSON"):
        verdict_call(
            lambda prompt: (
                "{"
                '"stop_reason":"business_decision_supportable",'
                '"hypothesis_status":"supported",'
                '"verdict_summary":"Supported.",'
                '"rationale":"Grounded.",'
                '"next_action":"stop",'
                '"cited_artifact_paths":["runs/uplift_ledger.jsonl"],'
                '"qini_auc":999'
                "}"
            ),
            decision,
            records=[_record()],
            max_retries=0,
        )

    verdict = verdict_call(
        lambda prompt: (
            "{"
            '"stop_reason":"business_decision_supportable",'
            '"hypothesis_status":"supported",'
            '"verdict_summary":"Supported.",'
            '"rationale":"Matches the deterministic stop decision.",'
            '"next_action":"stop",'
            '"cited_artifact_paths":["runs/uplift_ledger.jsonl"]'
            "}"
        ),
        decision,
        records=[_record()],
        max_retries=0,
    )

    assert verdict.stop_reason == decision.stop_reason
    assert verdict.hypothesis_status == decision.hypothesis_status


def test_report_prompt_is_bounded_and_requires_validation_held_out_scoring_sections():
    prompt = build_report_prompt(_decision(), records=[_record()])

    assert len(prompt.split()) <= 8000
    assert "validation" in prompt.lower()
    assert "held-out" in prompt.lower()
    assert "scoring-only" in prompt.lower()
    assert "artifact paths" in prompt.lower()


def test_report_call_requires_known_artifact_citations_and_sections():
    decision = _decision()

    with pytest.raises(ValueError, match="artifact citation"):
        report_call(
            lambda prompt: (
                "{"
                '"title":"Uplift report",'
                '"executive_summary":"Supported.",'
                '"validation_summary":"Validation qini_auc is positive.",'
                '"held_out_summary":"No held-out estimate here.",'
                '"scoring_summary":"Scoring remains separate.",'
                '"limitations":["Tiny fixture."],'
                '"cited_artifact_paths":[]'
                "}"
            ),
            decision,
            records=[_record()],
            max_retries=0,
        )

    report = report_call(
        lambda prompt: (
            "{"
            '"title":"Uplift report",'
            '"executive_summary":"Supported by deterministic evidence.",'
            '"validation_summary":"Validation qini_auc is positive.",'
            '"held_out_summary":"Held-out evidence is unavailable in this wave.",'
            '"scoring_summary":"Scoring-only submission remains separate from evaluation.",'
            '"limitations":["Tiny fixture."],'
            '"cited_artifact_paths":["runs/uplift_ledger.jsonl","runs/RUN-a/predictions.csv"]'
            "}"
        ),
        decision,
        records=[_record()],
        max_retries=0,
    )

    assert "Validation" in report.validation_summary
    assert "Held-out" in report.held_out_summary
    assert "Scoring-only" in report.scoring_summary
