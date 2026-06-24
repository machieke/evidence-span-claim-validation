from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import SourceModality, StrictModel

EvidenceType = Literal[
    "text_span",
    "utterance_span",
    "message_span",
    "visual_region",
    "visual_cluster",
    "ocr_text_span",
]


class EvidenceRecord(StrictModel):
    evidence_id: str
    source_id: str
    source_modality: SourceModality
    evidence_type: EvidenceType
    text: Optional[str] = None
    provenance: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "evidence.v1"

    @model_validator(mode="after")
    def validate_text_for_text_like_evidence(self) -> "EvidenceRecord":
        if self.evidence_type in {"text_span", "utterance_span", "message_span", "ocr_text_span"}:
            if self.text is None or not self.text.strip():
                raise ValueError("text-like evidence requires non-empty text")
        return self
