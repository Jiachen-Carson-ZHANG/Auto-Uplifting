"""Advisory LLM planning guardrails for uplift experiments."""
from __future__ import annotations

import json
import re
from typing import Callable, List

from src.models.uplift import (
    UpliftExperimentRecord,
    UpliftFeatureArtifact,
    UpliftProjectContract,
    UpliftTrialSpec,
)
from src.uplift.templates import REGISTERED_UPLIFT_TEMPLATES


LLMCall = Callable[[str], str]

_FAMILY_TO_TEMPLATE = {
    "random": "random_baseline",
    "response_model": "response_model_sklearn",
    "two_model": "two_model_sklearn",
    "solo_model": "solo_model_sklearn",
}


def _extract_json(raw: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


class UpliftAdvisoryPlanner:
    """Single-call strategy advisor constrained to registered uplift templates."""

    def __init__(self, llm_call: LLMCall) -> None:
        self._llm_call = llm_call

    def _prompt(
        self,
        contract: UpliftProjectContract,
        feature_artifact: UpliftFeatureArtifact,
        prior_records: List[UpliftExperimentRecord],
    ) -> str:
        prior_summary = [
            {
                "template_name": record.template_name,
                "status": record.status,
                "qini_auc": record.qini_auc,
                "uplift_auc": record.uplift_auc,
            }
            for record in prior_records[-5:]
        ]
        return (
            "You are an advisory uplift strategy call. "
            "You must choose only from registered templates. "
            "Do not change target, treatment, split, metric, or scoring semantics.\n"
            f"Task: {contract.task_name}\n"
            f"Entity key: {contract.entity_key}\n"
            f"Treatment column: {contract.treatment_column}\n"
            f"Target column: {contract.target_column}\n"
            f"Feature recipe id: {feature_artifact.feature_recipe_id}\n"
            f"Allowed templates: {sorted(REGISTERED_UPLIFT_TEMPLATES)}\n"
            f"Prior records: {json.dumps(prior_summary, sort_keys=True)}\n"
            "Return JSON with hypothesis_id, template_name, learner_family, "
            "base_estimator, and optional params."
        )

    def propose_next_trial(
        self,
        contract: UpliftProjectContract,
        *,
        feature_artifact: UpliftFeatureArtifact,
        prior_records: List[UpliftExperimentRecord],
    ) -> UpliftTrialSpec:
        """Return a valid trial spec; invalid LLM choices are clamped safely."""
        payload = _extract_json(
            self._llm_call(self._prompt(contract, feature_artifact, prior_records))
        )
        requested_family = str(payload.get("learner_family", "response_model"))
        if requested_family not in _FAMILY_TO_TEMPLATE:
            requested_family = "response_model"

        requested_template = str(payload.get("template_name", ""))
        expected_family = REGISTERED_UPLIFT_TEMPLATES.get(requested_template)
        if expected_family != requested_family:
            requested_template = _FAMILY_TO_TEMPLATE[requested_family]

        return UpliftTrialSpec(
            hypothesis_id=str(payload.get("hypothesis_id", "advisory-next-trial")),
            template_name=requested_template,
            learner_family=requested_family,  # type: ignore[arg-type]
            base_estimator=str(payload.get("base_estimator", "logistic_regression")),
            feature_recipe_id=feature_artifact.feature_recipe_id,
            params=payload.get("params", {}) if isinstance(payload.get("params", {}), dict) else {},
            split_seed=contract.split_contract.random_seed,
            primary_metric=contract.evaluation_policy.primary_metric,
        )
