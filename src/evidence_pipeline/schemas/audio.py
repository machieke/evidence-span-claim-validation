from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel


class AudioUtteranceRecord(StrictModel):
    utterance_id: str
    source_id: str
    speaker: str
    start: float
    end: float
    text: str
    asr_segment_ids: List[str] = Field(default_factory=list)
    turn_ids: List[str] = Field(default_factory=list)
    asr_confidence: Optional[float] = None
    diarization_confidence: Optional[float] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "audio.utterance.v1"

    @model_validator(mode="after")
    def validate_utterance(self) -> "AudioUtteranceRecord":
        if self.start < 0:
            raise ValueError("start must be non-negative")
        if self.end < self.start:
            raise ValueError("end must be greater than or equal to start")
        if not self.speaker.strip():
            raise ValueError("speaker must not be empty")
        if not self.text.strip():
            raise ValueError("utterance text must not be empty")
        for field_name in ("asr_confidence", "diarization_confidence"):
            value = getattr(self, field_name)
            if value is not None and not (0 <= value <= 1):
                raise ValueError(f"{field_name} must be between 0 and 1")
        return self
