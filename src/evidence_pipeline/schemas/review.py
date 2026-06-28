from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import SourceModality, StrictModel, utc_now

ReviewDecision = Literal["accept", "reject", "needs_review"]
ReviewQueueState = Literal["unreviewed", "accept", "reject", "needs_review"]


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


class ReviewQueueEvidence(StrictModel):
    evidence_type: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)


class ReviewQueueRecord(StrictModel):
    review_queue_id: str
    claim_id: str
    source_id: Optional[str] = None
    evidence_id: Optional[str] = None
    source_file: Optional[str] = None
    source_modality: Optional[SourceModality] = None
    claim_type: Optional[str] = None
    source_faithful_claim: Optional[str] = None
    evidence_text: Optional[str] = None
    evidence: ReviewQueueEvidence = Field(default_factory=ReviewQueueEvidence)
    evidence_anchor: Dict[str, Any] = Field(default_factory=dict)
    normalized_claims: List[Dict[str, Any]] = Field(default_factory=list)
    validation_status: str
    reason_codes: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    review_state: ReviewQueueState
    review_commands: Dict[ReviewDecision, str] = Field(default_factory=dict)
    latest_review: Optional[Dict[str, Any]] = None
    schema_version: str = "review.queue.v1"

    @model_validator(mode="after")
    def validate_required_identifiers(self) -> "ReviewQueueRecord":
        if not self.review_queue_id.strip():
            raise ValueError("review_queue_id must not be empty")
        if not self.claim_id.strip():
            raise ValueError("claim_id must not be empty")
        return self
