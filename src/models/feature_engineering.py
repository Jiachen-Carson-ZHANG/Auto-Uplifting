"""
Feature engineering models for Phase 5.

FeatureDecision      — structured output from the agent's decision call.
FeatureSpec          — discriminated union of spec types (template, transform, composite, codegen).
FeatureAuditVerdict  — leakage audit or codegen guardrail verdict.
FeatureExecutionResult — outcome of executing a feature proposal.
FeatureHistoryEntry  — append-only telemetry for empirical experiment memory.
"""
from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union
from typing import Annotated
from pydantic import BaseModel, Field


# ── Feature Spec variants (discriminated on spec_type) ──────────────

class TemplateFeatureSpec(BaseModel):
    """Spec for a registry-backed template feature (RFM, temporal, etc.)."""
    spec_type: Literal["template"] = "template"
    template_name: str
    params: Dict[str, Any] = Field(default_factory=dict)


class TransformFeatureSpec(BaseModel):
    """Spec for a single-column transform (log1p, clip, bucketize, etc.)."""
    spec_type: Literal["transform"] = "transform"
    input_col: str
    op: str
    params: Dict[str, Any] = Field(default_factory=dict)
    output_col: str


class CompositeFeatureSpec(BaseModel):
    """Spec for a multi-input composite operation (safe_divide, ratios, etc.)."""
    spec_type: Literal["composite"] = "composite"
    name: str
    op: str
    inputs: List[Dict[str, Any]]  # [{"ref": "col_name"} | {"literal": value}]
    post: List[str] = Field(default_factory=list)


class CodegenEscalationSpec(BaseModel):
    """Spec for Phase 2 codegen escape hatch — agent must explain why bounded path is insufficient."""
    spec_type: Literal["codegen"] = "codegen"
    description: str
    reason_bounded_insufficient: str
    code: Optional[str] = None


FeatureSpec = Annotated[
    Union[TemplateFeatureSpec, TransformFeatureSpec, CompositeFeatureSpec, CodegenEscalationSpec],
    Field(discriminator="spec_type"),
]


# ── Decision contract ───────────────────────────────────────────────

class FeatureDecision(BaseModel):
    """Structured output from the FeatureEngineeringAgent's decision call."""
    status: Literal["proposed", "blocked", "skip"]
    action: Literal[
        "add", "drop", "transform", "composite",
        "request_context", "blocked", "escalate_codegen",
    ]
    reasoning: str
    feature_spec: Optional[FeatureSpec] = None
    expected_impact: str = ""
    risk_flags: List[str] = Field(default_factory=list)
    observations: List[str] = Field(default_factory=list)
    facts_to_save: List[str] = Field(default_factory=list)


# ── Audit verdict ───────────────────────────────────────────────────

class FeatureAuditVerdict(BaseModel):
    """Leakage audit or codegen guardrail verdict."""
    verdict: Literal["pass", "block", "warn"]
    reasons: List[str] = Field(default_factory=list)
    required_fixes: List[str] = Field(default_factory=list)


# ── Execution result ────────────────────────────────────────────────

class FeatureExecutionResult(BaseModel):
    """Outcome of executing a feature proposal (bounded or codegen)."""
    status: Literal["success", "failed", "blocked"]
    output_path: Optional[str] = None
    produced_columns: List[str] = Field(default_factory=list)
    dropped_columns: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    failure_reason: Optional[str] = None


# ── History entry (empirical experiment memory) ─────────────────────

class FeatureHistoryEntry(BaseModel):
    """
    Append-only telemetry for the feature engineering loop.

    This is empirical experiment memory — what we tried, what happened,
    and what to remember. Static external knowledge belongs in references/.
    """
    entry_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    action: str
    feature_spec_json: str  # serialized spec for replay
    dataset_name: str
    task_type: str
    metric_before: Optional[float] = None
    metric_after: Optional[float] = None
    observed_outcome: str = ""
    distilled_takeaway: str = ""
    audit_verdict: str = ""
