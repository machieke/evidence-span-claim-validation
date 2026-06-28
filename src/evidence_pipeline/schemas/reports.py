from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel


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
