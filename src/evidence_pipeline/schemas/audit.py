from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel, utc_now

AuditEventStatus = Literal["created", "skipped", "failed"]


class AuditEventRecord(StrictModel):
    audit_event_id: str
    action: str
    actor_id: Optional[str] = None
    target_type: str
    target_id: str
    source_id: Optional[str] = None
    evidence_id: Optional[str] = None
    claim_id: Optional[str] = None
    status: AuditEventStatus
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    schema_version: str = "audit.event.v1"

    @model_validator(mode="after")
    def validate_required_labels(self) -> "AuditEventRecord":
        if not self.action.strip():
            raise ValueError("action must not be empty")
        if not self.target_type.strip():
            raise ValueError("target_type must not be empty")
        if not self.target_id.strip():
            raise ValueError("target_id must not be empty")
        return self
