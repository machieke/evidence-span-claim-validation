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
