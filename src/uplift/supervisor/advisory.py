"""Advisory LLM call boundaries for the uplift supervisor."""
from __future__ import annotations

import json
from typing import Callable, Mapping, Sequence, TypeVar

from pydantic import ValidationError

from src.models.uplift import (
    UpliftAdvisoryReport,
    UpliftAdvisoryVerdict,
    UpliftDiagnosisResult,
    UpliftExperimentRecord,
    UpliftExperimentWaveSpec,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftStopDecision,
)
from src.uplift.supervisor.waves import validate_wave_spec
from src.uplift.templates import REGISTERED_UPLIFT_TEMPLATES


LLMCall = Callable[[str], str]
T = TypeVar("T")

_ESTIMATED_TOKENS_PER_WORD = 1.33
_PROMPT_TOKEN_BUDGET = 8000
_SUPPORTED_ADVISORY_ACTIONS = [
    "cost_sensitivity",
    "recipe_comparison",
    "ranking_stability_check",
    "window_sweep",
    "feature_ablation",
    "feature_group_expansion",
]
_FORBIDDEN_CONTRACT_KEYS = {
    "entity_key",
    "evaluation_policy",
    "primary_metric",
    "scoring_uplift_column",
    "split_contract",
    "split_seed",
    "table_schema",
    "target_column",
    "treatment_column",
}


def build_diagnosis_prompt(
    contract: UpliftProjectContract,
    *,
    records: Sequence[UpliftExperimentRecord],
    valid_actions: Sequence[str] | None = None,
) -> str:
    """Build a bounded prompt for unresolved-question diagnosis."""
    actions = list(valid_actions or _SUPPORTED_ADVISORY_ACTIONS)
    prior_records = [_record_summary(record) for record in records[-10:]]
    prompt = (
        "You are the diagnosis_call for a deterministic uplift supervisor.\n"
        "Own only unresolved question framing, risks, and candidate hypotheses.\n"
        "Contract fields are immutable. Do not change target, treatment, split, "
        "metric, feature artifacts, templates, or scoring semantics.\n"
        f"Task: {contract.task_name}\n"
        f"Entity key: {contract.entity_key}\n"
        f"Treatment column: {contract.treatment_column}\n"
        f"Target column: {contract.target_column}\n"
        f"Primary metric: {contract.evaluation_policy.primary_metric}\n"
        f"Split seed: {contract.split_contract.random_seed}\n"
        f"Allowed actions: {json.dumps(actions)}\n"
        f"Recent ledger records: {json.dumps(prior_records, sort_keys=True)}\n"
        "Return strict JSON with unresolved_questions, risks, and "
        "candidate_hypotheses. Do not include contract or metric fields."
    )
    return _cap_prompt(prompt)


def diagnosis_call(
    llm_call: LLMCall,
    contract: UpliftProjectContract,
    *,
    records: Sequence[UpliftExperimentRecord],
    valid_actions: Sequence[str] | None = None,
    max_retries: int = 1,
) -> UpliftDiagnosisResult:
    """Run the advisory diagnosis call with strict JSON parsing."""
    actions = list(valid_actions or _SUPPORTED_ADVISORY_ACTIONS)
    prompt = build_diagnosis_prompt(contract, records=records, valid_actions=actions)

    def parse(raw: str) -> UpliftDiagnosisResult:
        payload = _strict_json_object(raw)
        result = UpliftDiagnosisResult.model_validate(payload)
        invalid_actions = sorted(
            {
                candidate.action_type
                for candidate in result.candidate_hypotheses
                if candidate.action_type not in actions
            }
        )
        if invalid_actions:
            raise ValueError(
                f"unsupported diagnosis action_type: {', '.join(invalid_actions)}"
            )
        return result

    return _call_with_retries(
        llm_call,
        prompt,
        parse=parse,
        max_retries=max_retries,
        error_label="invalid diagnosis JSON",
    )


def build_wave_planning_prompt(
    contract: UpliftProjectContract,
    *,
    diagnosis: UpliftDiagnosisResult,
    feature_artifacts: Mapping[str, UpliftFeatureArtifact],
    known_hypothesis_ids: Sequence[str],
) -> str:
    """Build a bounded prompt for one valid wave proposal."""
    artifacts = {
        recipe_id: {
            "feature_artifact_id": artifact.feature_artifact_id,
            "feature_groups": artifact.feature_groups,
            "windows_days": artifact.windows_days,
        }
        for recipe_id, artifact in feature_artifacts.items()
    }
    prompt = (
        "You are the wave_planning_call for a deterministic uplift supervisor.\n"
        "Own one wave proposal only. Validators own execution authority.\n"
        "Do not change target, treatment, split, or metric. Use the exact "
        "split_seed and primary_metric below in every trial spec.\n"
        f"Split seed: {contract.split_contract.random_seed}\n"
        f"Primary metric: {contract.evaluation_policy.primary_metric}\n"
        f"Allowed actions: {json.dumps(_SUPPORTED_ADVISORY_ACTIONS)}\n"
        f"Allowed templates: {json.dumps(sorted(REGISTERED_UPLIFT_TEMPLATES))}\n"
        f"Allowed feature recipe IDs: {json.dumps(sorted(feature_artifacts))}\n"
        f"Known hypothesis IDs: {json.dumps(list(known_hypothesis_ids))}\n"
        f"Feature artifact metadata: {json.dumps(artifacts, sort_keys=True)}\n"
        f"Diagnosis: {diagnosis.model_dump_json()}\n"
        "Return strict JSON matching UpliftExperimentWaveSpec with "
        "created_by='llm'. Do not include contract fields."
    )
    return _cap_prompt(prompt)


def wave_planning_call(
    llm_call: LLMCall,
    contract: UpliftProjectContract,
    *,
    diagnosis: UpliftDiagnosisResult,
    feature_artifacts: Mapping[str, UpliftFeatureArtifact],
    known_hypothesis_ids: Sequence[str],
    max_retries: int = 1,
) -> UpliftExperimentWaveSpec:
    """Run advisory wave planning and return a preflight-valid wave spec."""
    prompt = build_wave_planning_prompt(
        contract,
        diagnosis=diagnosis,
        feature_artifacts=feature_artifacts,
        known_hypothesis_ids=known_hypothesis_ids,
    )

    def parse(raw: str) -> UpliftExperimentWaveSpec:
        payload = _strict_json_object(raw)
        _reject_contract_mutation(payload)
        wave = UpliftExperimentWaveSpec.model_validate(payload)
        if wave.created_by != "llm":
            raise ValueError("wave_planning_call requires created_by='llm'")
        if wave.hypothesis_id not in set(known_hypothesis_ids):
            raise ValueError(f"unknown hypothesis_id: {wave.hypothesis_id}")
        for trial_spec in wave.trial_specs:
            if trial_spec.split_seed != contract.split_contract.random_seed:
                raise ValueError("split mutation is not allowed")
            if trial_spec.primary_metric != contract.evaluation_policy.primary_metric:
                raise ValueError("metric mutation is not allowed")
        validate_wave_spec(wave, feature_artifacts=feature_artifacts)
        return wave

    return _call_with_retries(
        llm_call,
        prompt,
        parse=parse,
        max_retries=max_retries,
        error_label="invalid wave planning JSON",
    )


def build_verdict_prompt(
    decision: UpliftStopDecision,
    *,
    records: Sequence[UpliftExperimentRecord],
) -> str:
    """Build a bounded prompt for narrative verdict interpretation."""
    prompt = (
        "You are the verdict_call for a deterministic uplift supervisor.\n"
        "Numeric metrics are deterministic evidence and must not be overwritten.\n"
        "Explain the stop decision without changing stop_reason, "
        "hypothesis_status, trial IDs, run IDs, or artifact paths.\n"
        f"Stop decision: {decision.model_dump_json()}\n"
        f"Ledger records: {json.dumps([_record_summary(r) for r in records[-10:]], sort_keys=True)}\n"
        "Return strict JSON with stop_reason, hypothesis_status, "
        "verdict_summary, rationale, next_action, and cited_artifact_paths."
    )
    return _cap_prompt(prompt)


def verdict_call(
    llm_call: LLMCall,
    decision: UpliftStopDecision,
    *,
    records: Sequence[UpliftExperimentRecord],
    max_retries: int = 1,
) -> UpliftAdvisoryVerdict:
    """Run advisory verdict narration over deterministic stop evidence."""
    prompt = build_verdict_prompt(decision, records=records)

    def parse(raw: str) -> UpliftAdvisoryVerdict:
        payload = _strict_json_object(raw)
        verdict = UpliftAdvisoryVerdict.model_validate(payload)
        if verdict.stop_reason != decision.stop_reason:
            raise ValueError("verdict stop_reason conflicts with deterministic decision")
        if verdict.hypothesis_status != decision.hypothesis_status:
            raise ValueError(
                "verdict hypothesis_status conflicts with deterministic decision"
            )
        _validate_citations(verdict.cited_artifact_paths, _allowed_artifact_paths(decision))
        return verdict

    return _call_with_retries(
        llm_call,
        prompt,
        parse=parse,
        max_retries=max_retries,
        error_label="invalid verdict JSON",
    )


def build_report_prompt(
    decision: UpliftStopDecision,
    *,
    records: Sequence[UpliftExperimentRecord],
) -> str:
    """Build a bounded prompt for grounded stakeholder report narration."""
    prompt = (
        "You are the report_call for a deterministic uplift supervisor.\n"
        "Use artifact paths as citations. Distinguish validation metrics, "
        "held-out metrics, and scoring-only outputs. Do not create unsupported "
        "metric values or business claims.\n"
        f"Stop decision: {decision.model_dump_json()}\n"
        f"Ledger records: {json.dumps([_record_summary(r) for r in records[-10:]], sort_keys=True)}\n"
        "Return strict JSON with title, executive_summary, validation_summary, "
        "held_out_summary, scoring_summary, limitations, and cited_artifact_paths."
    )
    return _cap_prompt(prompt)


def report_call(
    llm_call: LLMCall,
    decision: UpliftStopDecision,
    *,
    records: Sequence[UpliftExperimentRecord],
    max_retries: int = 1,
) -> UpliftAdvisoryReport:
    """Run advisory report narration with artifact citation enforcement."""
    prompt = build_report_prompt(decision, records=records)

    def parse(raw: str) -> UpliftAdvisoryReport:
        payload = _strict_json_object(raw)
        report = UpliftAdvisoryReport.model_validate(payload)
        _validate_citations(report.cited_artifact_paths, _allowed_artifact_paths(decision))
        return report

    return _call_with_retries(
        llm_call,
        prompt,
        parse=parse,
        max_retries=max_retries,
        error_label="invalid report JSON",
    )


def _call_with_retries(
    llm_call: LLMCall,
    prompt: str,
    *,
    parse: Callable[[str], T],
    max_retries: int,
    error_label: str,
) -> T:
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return parse(llm_call(prompt))
        except (ValueError, ValidationError) as exc:
            last_error = exc
    raise ValueError(f"{error_label}: {last_error}") from last_error


def _strict_json_object(raw: str) -> dict[str, object]:
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    decoder = json.JSONDecoder()
    try:
        payload, end = decoder.raw_decode(raw[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc
    if raw[start + end :].strip():
        raise ValueError("unexpected text after JSON object")
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def _reject_contract_mutation(payload: object) -> None:
    if isinstance(payload, dict):
        found = sorted(set(payload).intersection(_FORBIDDEN_CONTRACT_KEYS))
        if found:
            raise ValueError(f"contract mutation is not allowed: {', '.join(found)}")
        for key, value in payload.items():
            if key == "trial_specs":
                continue
            _reject_contract_mutation(value)
    elif isinstance(payload, list):
        for item in payload:
            _reject_contract_mutation(item)


def _validate_citations(
    cited_artifact_paths: Sequence[str],
    allowed_artifact_paths: set[str],
) -> None:
    if not cited_artifact_paths:
        raise ValueError("artifact citation is required")
    unknown = sorted(set(cited_artifact_paths) - allowed_artifact_paths)
    if unknown:
        raise ValueError(f"unknown artifact citation: {', '.join(unknown)}")


def _allowed_artifact_paths(decision: UpliftStopDecision) -> set[str]:
    return {path for path in decision.artifact_paths.values() if path}


def _record_summary(record: UpliftExperimentRecord) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "hypothesis_id": record.hypothesis_id,
        "template_name": record.template_name,
        "status": record.status,
        "qini_auc": record.qini_auc,
        "uplift_auc": record.uplift_auc,
        "policy_gain": record.policy_gain,
        "artifact_paths": record.artifact_paths,
    }


def _cap_prompt(
    prompt: str,
    max_estimated_tokens: int = _PROMPT_TOKEN_BUDGET,
) -> str:
    """Cap prompt context with a conservative word proxy for token budget."""
    words = prompt.split()
    max_words = max(1, int(max_estimated_tokens / _ESTIMATED_TOKENS_PER_WORD))
    if len(words) <= max_words:
        return prompt
    return " ".join(words[:max_words])
