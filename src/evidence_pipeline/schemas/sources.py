from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import Field, field_validator

from evidence_pipeline.schemas.base import SourceModality, StrictModel, utc_now


class SourceRecord(StrictModel):
    source_id: str
    source_modality: SourceModality
    source_file: Optional[str] = None
    source_uri: Optional[str] = None
    sha256: Optional[str] = None
    created_at: Optional[datetime] = None
    ingested_at: datetime = Field(default_factory=utc_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "source.v1"

    @field_validator("source_file", "source_uri")
    @classmethod
    def non_empty_optional_string(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("must not be empty")
        return value
