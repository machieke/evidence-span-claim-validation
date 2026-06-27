from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel, utc_now

ReviewDecision = Literal["accept", "reject", "needs_review"]


class ReviewDecisionRecord(StrictModel):
    review_id: str
    claim_id: str
    source_id: Optional[str] = None
    evidence_id: Optional[str] = None
    reviewer_id: str
    decision: ReviewDecision
    reason_codes: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    reviewed_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "review.decision.v1"

    @model_validator(mode="after")
    def validate_reviewer(self) -> "ReviewDecisionRecord":
        if not self.reviewer_id.strip():
            raise ValueError("reviewer_id must not be empty")
        return self
