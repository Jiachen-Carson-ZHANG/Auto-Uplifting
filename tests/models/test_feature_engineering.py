"""Tests for src/models/feature_engineering.py — Pydantic contracts."""
import pytest
from pydantic import ValidationError

from src.models.feature_engineering import (
    CodegenEscalationSpec,
    CompositeFeatureSpec,
    FeatureAuditVerdict,
    FeatureDecision,
    FeatureExecutionResult,
    FeatureHistoryEntry,
    TemplateFeatureSpec,
    TransformFeatureSpec,
)


class TestFeatureDecision:
    def test_round_trip(self):
        d = FeatureDecision(
            status="proposed",
            action="add",
            reasoning="test",
            feature_spec=TemplateFeatureSpec(template_name="rfm_recency"),
        )
        json_str = d.model_dump_json()
        restored = FeatureDecision.model_validate_json(json_str)
        assert restored.action == "add"
        assert restored.feature_spec.spec_type == "template"

    def test_invalid_action_rejected(self):
        with pytest.raises(ValidationError):
            FeatureDecision(
                status="proposed", action="invalid_action", reasoning="test"
            )

    def test_invalid_status_rejected(self):
        with pytest.raises(ValidationError):
            FeatureDecision(
                status="invalid_status", action="add", reasoning="test"
            )

    def test_defaults(self):
        d = FeatureDecision(
            status="blocked", action="blocked", reasoning="no action"
        )
        assert d.feature_spec is None
        assert d.risk_flags == []
        assert d.observations == []
        assert d.facts_to_save == []
        assert d.expected_impact == ""


class TestDiscriminatedUnion:
    def test_template_spec(self):
        d = FeatureDecision(
            status="proposed",
            action="add",
            reasoning="test",
            feature_spec=TemplateFeatureSpec(
                template_name="rfm_recency", params={"entity_key": "cid"}
            ),
        )
        assert d.feature_spec.spec_type == "template"
        assert d.feature_spec.template_name == "rfm_recency"

    def test_transform_spec(self):
        d = FeatureDecision(
            status="proposed",
            action="transform",
            reasoning="test",
            feature_spec=TransformFeatureSpec(
                input_col="amount", op="log1p", output_col="log_amount"
            ),
        )
        assert d.feature_spec.spec_type == "transform"

    def test_composite_spec(self):
        d = FeatureDecision(
            status="proposed",
            action="composite",
            reasoning="test",
            feature_spec=CompositeFeatureSpec(
                name="ratio",
                op="safe_divide",
                inputs=[{"ref": "a"}, {"ref": "b"}],
            ),
        )
        assert d.feature_spec.spec_type == "composite"

    def test_codegen_spec(self):
        d = FeatureDecision(
            status="proposed",
            action="escalate_codegen",
            reasoning="test",
            feature_spec=CodegenEscalationSpec(
                description="custom logic",
                reason_bounded_insufficient="need sequence",
            ),
        )
        assert d.feature_spec.spec_type == "codegen"

    def test_discriminated_union_from_json(self):
        """Parse JSON with spec_type discriminator."""
        json_str = '{"status":"proposed","action":"add","reasoning":"test","feature_spec":{"spec_type":"template","template_name":"rfm_recency","params":{}}}'
        d = FeatureDecision.model_validate_json(json_str)
        assert isinstance(d.feature_spec, TemplateFeatureSpec)


class TestFeatureAuditVerdict:
    def test_defaults(self):
        v = FeatureAuditVerdict(verdict="pass")
        assert v.reasons == []
        assert v.required_fixes == []

    def test_round_trip(self):
        v = FeatureAuditVerdict(
            verdict="block", reasons=["target leak"], required_fixes=["remove target"]
        )
        restored = FeatureAuditVerdict.model_validate_json(v.model_dump_json())
        assert restored.verdict == "block"
        assert len(restored.reasons) == 1


class TestFeatureExecutionResult:
    def test_success(self):
        r = FeatureExecutionResult(
            status="success", produced_columns=["new_col"]
        )
        assert r.failure_reason is None

    def test_failed(self):
        r = FeatureExecutionResult(
            status="failed", failure_reason="column not found"
        )
        assert r.produced_columns == []


class TestFeatureHistoryEntry:
    def test_defaults(self):
        e = FeatureHistoryEntry(
            entry_id="test-001",
            action="add",
            feature_spec_json="{}",
            dataset_name="uci",
            task_type="binary",
        )
        assert e.metric_before is None
        assert e.metric_after is None
        assert e.observed_outcome == ""
        assert e.timestamp is not None

    def test_round_trip(self):
        e = FeatureHistoryEntry(
            entry_id="test-002",
            action="transform",
            feature_spec_json='{"op":"log1p"}',
            dataset_name="olist",
            task_type="regression",
            metric_before=0.5,
            metric_after=0.55,
        )
        restored = FeatureHistoryEntry.model_validate_json(e.model_dump_json())
        assert restored.metric_after == 0.55
