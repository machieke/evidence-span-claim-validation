from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import SourceModality, StrictModel

SpanLabel = Literal["claim_bearing", "not_claim_bearing", "visual_region_candidate", "visual_cluster_candidate"]


class SpanRecord(StrictModel):
    span_id: str
    chunk_id: Optional[str] = None
    source_id: str
    source_modality: SourceModality
    evidence_id: str
    text: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    context_text: Optional[str] = None
    label: SpanLabel
    score: Optional[float] = None
    detector: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "span.v1"

    @model_validator(mode="after")
    def validate_text_offsets(self) -> "SpanRecord":
        if self.source_modality != "image":
            if self.text is None or not self.text.strip():
                raise ValueError("non-image spans require text")
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must both be present or absent")
        if self.char_start is not None:
            if self.char_start < 0 or self.char_end < self.char_start:
                raise ValueError("invalid character offsets")
        if self.score is not None and not (0 <= self.score <= 1):
            raise ValueError("score must be between 0 and 1")
        return self
