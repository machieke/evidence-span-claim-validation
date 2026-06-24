from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import Field

from evidence_pipeline.schemas.base import SourceModality, StrictModel


class ChunkRecord(StrictModel):
    chunk_id: str
    source_id: str
    source_modality: SourceModality
    evidence_ids: List[str] = Field(default_factory=list)
    primary_evidence_ids: List[str] = Field(default_factory=list)
    overlap_evidence_ids: List[str] = Field(default_factory=list)
    text: Optional[str] = None
    provenance_summary: Dict[str, Any] = Field(default_factory=dict)
    chunking_policy: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "chunk.v1"
