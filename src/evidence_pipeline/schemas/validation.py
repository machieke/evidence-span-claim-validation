from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel, utc_now

ValidationStatus = Literal[
    "schema_valid",
    "schema_invalid",
    "deterministic_valid",
    "semantic_valid",
    "accepted_extracted",
    "needs_review",
    "quarantined",
    "repair_attempted",
    "repaired",
]


class ValidationRecord(StrictModel):
    validation_id: str
    claim_id: Optional[str] = None
    record_id: Optional[str] = None
    stage: str
    status: ValidationStatus
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validator_version: str = "deterministic.v8"
    created_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "validation.v1"

    @model_validator(mode="after")
    def validate_rejected_status_errors(self) -> "ValidationRecord":
        if self.status in {"schema_invalid", "quarantined"} and not self.errors:
            raise ValueError("rejected validation records require at least one error code")
        return self


class ErrorRecord(StrictModel):
    error_id: str
    stage: str
    message: str
    source_id: Optional[str] = None
    record_id: Optional[str] = None
    reason_code: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "error.v1"


class QuarantineRecord(StrictModel):
    quarantine_id: str
    record_type: str
    record_id: str
    source_id: Optional[str] = None
    evidence_id: Optional[str] = None
    claim_id: Optional[str] = None
    stage: str
    reason_codes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    payload: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "quarantine.v1"

    @model_validator(mode="after")
    def validate_reason_codes(self) -> "QuarantineRecord":
        if not self.reason_codes:
            raise ValueError("quarantine records require at least one reason code")
        return self
