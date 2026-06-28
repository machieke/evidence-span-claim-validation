from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import SourceModality, StrictModel


class GraphEdgeRecord(StrictModel):
    edge_id: str
    normalized_claim_id: str
    claim_id: str
    source_id: str
    evidence_id: str
    subject: Any
    predicate: str
    object: Any
    truth_status: Optional[str] = None
    attribution: Optional[Dict[str, Any]] = None
    qualifiers: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "graph.edge.v1"

    @model_validator(mode="after")
    def validate_required_identifiers(self) -> "GraphEdgeRecord":
        for field_name in (
            "edge_id",
            "normalized_claim_id",
            "claim_id",
            "source_id",
            "evidence_id",
            "predicate",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        return self


class ModelRoutingRecord(StrictModel):
    routing_id: str
    stage: str
    record_type: str
    record_id: str
    source_id: str
    source_modality: SourceModality
    model_role: Literal["extraction", "validation"]
    selected_tier: Literal["default", "strong"]
    selected_model: str
    reasons: List[str] = Field(default_factory=list)
    score: Optional[float] = None
    schema_version: str = "model.routing.v1"

    @model_validator(mode="after")
    def validate_route(self) -> "ModelRoutingRecord":
        for field_name in (
            "routing_id",
            "stage",
            "record_type",
            "record_id",
            "source_id",
            "selected_model",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.score is not None and not (0 <= self.score <= 1):
            raise ValueError("score must be between 0 and 1")
        return self


class PrivacyPolicyViolationRecord(StrictModel):
    violation_id: str
    source_id: str
    claim_id: str
    evidence_id: str
    provider: str
    model: str
    policy: str
    reason_code: str
    sensitive_metadata_keys: List[str] = Field(default_factory=list)
    schema_version: str = "privacy.violation.v1"

    @model_validator(mode="after")
    def validate_violation(self) -> "PrivacyPolicyViolationRecord":
        for field_name in (
            "violation_id",
            "source_id",
            "claim_id",
            "evidence_id",
            "provider",
            "model",
            "policy",
            "reason_code",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        return self


class RetentionPlanRecord(StrictModel):
    retention_id: str
    action: Literal["delete_raw_source"]
    source_id: str
    source_modality: SourceModality
    source_file: str
    source_uri: Optional[str] = None
    ingested_at: datetime
    age_days: int
    retention_days: int
    reason_code: str
    dry_run: bool = True
    schema_version: str = "retention.plan.v1"

    @model_validator(mode="after")
    def validate_retention_plan(self) -> "RetentionPlanRecord":
        for field_name in ("retention_id", "source_id", "source_file", "reason_code"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.age_days < 0:
            raise ValueError("age_days must be non-negative")
        if self.retention_days < 1:
            raise ValueError("retention_days must be positive")
        return self


class PIIFindingRecord(StrictModel):
    finding_id: str
    artifact: str
    record_id: str
    source_id: Optional[str] = None
    evidence_id: Optional[str] = None
    claim_id: Optional[str] = None
    field: str
    pii_type: Literal["email", "phone", "ssn"]
    match_hash: str
    redacted_preview: str
    char_start: int
    char_end: int
    schema_version: str = "pii.finding.v1"

    @model_validator(mode="after")
    def validate_finding(self) -> "PIIFindingRecord":
        for field_name in (
            "finding_id",
            "artifact",
            "record_id",
            "field",
            "match_hash",
            "redacted_preview",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if self.char_start < 0 or self.char_end <= self.char_start:
            raise ValueError("PII finding character offsets are invalid")
        return self


class PIIRedactionRecord(StrictModel):
    redaction_id: str
    artifact: str
    record_id: str
    source_id: Optional[str] = None
    evidence_id: Optional[str] = None
    claim_id: Optional[str] = None
    fields: List[str] = Field(default_factory=list)
    replacement_count: int
    output_path: str
    schema_version: str = "pii.redaction.v1"

    @model_validator(mode="after")
    def validate_redaction(self) -> "PIIRedactionRecord":
        for field_name in ("redaction_id", "artifact", "record_id", "output_path"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.fields:
            raise ValueError("PII redaction records require at least one field")
        if self.replacement_count < 1:
            raise ValueError("replacement_count must be positive")
        return self


class ClaimRepairSuggestionRecord(StrictModel):
    repair_id: str
    claim_id: str
    source_id: str
    evidence_id: str
    span_id: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    original_evidence_text: str
    suggested_evidence_text: str
    support_scope: Literal["span", "evidence"]
    schema_version: str = "claim.repair_suggestion.v1"

    @model_validator(mode="after")
    def validate_repair(self) -> "ClaimRepairSuggestionRecord":
        for field_name in (
            "repair_id",
            "claim_id",
            "source_id",
            "evidence_id",
            "original_evidence_text",
            "suggested_evidence_text",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must not be empty")
        if not self.reason_codes:
            raise ValueError("repair suggestions require at least one reason code")
        return self


class ClaimDuplicateGroupRecord(StrictModel):
    dedupe_id: str
    normalized_proposition: Dict[str, Any]
    normalized_claim: Dict[str, Any]
    member_count: int
    member_claim_ids: List[str]
    member_normalized_claim_ids: List[str]
    source_ids: List[str]
    evidence_ids: List[str]
    duplicate_level: str = "same_normalized_proposition"
    cross_source: bool = False
    source_count: int = 0
    evidence_count: int = 0
    omitted_qualifier_keys: List[str] = Field(default_factory=list)
    schema_version: str = "claim.dedupe.v1"

    @model_validator(mode="after")
    def validate_duplicate_group(self) -> "ClaimDuplicateGroupRecord":
        if not self.dedupe_id.strip():
            raise ValueError("dedupe_id must not be empty")
        if self.member_count < 1:
            raise ValueError("member_count must be positive")
        if len(self.member_claim_ids) != self.member_count:
            raise ValueError("member_count must match member_claim_ids")
        if len(self.member_normalized_claim_ids) != self.member_count:
            raise ValueError("member_count must match member_normalized_claim_ids")
        if len(self.evidence_ids) != self.member_count:
            raise ValueError("member_count must match evidence_ids")
        if not self.source_ids:
            raise ValueError("duplicate groups require at least one source_id")
        source_count = len(set(self.source_ids))
        evidence_count = len(set(self.evidence_ids))
        if self.source_count not in {0, source_count}:
            raise ValueError("source_count must match source_ids")
        if self.evidence_count not in {0, evidence_count}:
            raise ValueError("evidence_count must match evidence_ids")
        self.source_count = source_count
        self.evidence_count = evidence_count
        self.cross_source = source_count > 1
        return self


class GoldEvaluationRecord(StrictModel):
    evaluation_id: str
    gold_path: str
    gold_claims: int
    expected_accepted: int
    produced_accepted: int
    accepted_matches: int
    accepted_false_positives: int
    accepted_missing: int
    accepted_precision: Optional[float] = None
    accepted_recall: Optional[float] = None
    expected_quarantined: int
    produced_quarantined: int
    quarantine_matches: int
    quarantine_false_positives: int
    quarantine_missing: int
    quarantine_precision: Optional[float] = None
    quarantine_recall: Optional[float] = None
    missing_keys: List[Dict[str, str]] = Field(default_factory=list)
    false_positive_keys: List[Dict[str, str]] = Field(default_factory=list)
    missing_quarantine_keys: List[Dict[str, str]] = Field(default_factory=list)
    false_positive_quarantine_keys: List[Dict[str, str]] = Field(default_factory=list)
    schema_version: str = "gold.eval.v1"

    @model_validator(mode="after")
    def validate_gold_eval(self) -> "GoldEvaluationRecord":
        if not self.evaluation_id.strip():
            raise ValueError("evaluation_id must not be empty")
        if not self.gold_path.strip():
            raise ValueError("gold_path must not be empty")
        count_fields = (
            "gold_claims",
            "expected_accepted",
            "produced_accepted",
            "accepted_matches",
            "accepted_false_positives",
            "accepted_missing",
            "expected_quarantined",
            "produced_quarantined",
            "quarantine_matches",
            "quarantine_false_positives",
            "quarantine_missing",
        )
        for field_name in count_fields:
            if getattr(self, field_name) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        for field_name in (
            "accepted_precision",
            "accepted_recall",
            "quarantine_precision",
            "quarantine_recall",
        ):
            value = getattr(self, field_name)
            if value is not None and not (0 <= value <= 1):
                raise ValueError(f"{field_name} must be between 0 and 1")
        return self
