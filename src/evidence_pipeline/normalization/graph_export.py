from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord


@dataclass
class GraphExportResult:
    output_path: Path
    edge_count: int


def _edge_from_normalized_claim(record: NormalizedClaimRecord) -> Dict[str, object]:
    normalized = record.normalized_claim
    subject = normalized.get("subject")
    predicate = normalized.get("predicate")
    object_value = normalized.get("object")
    qualifiers = normalized.get("qualifiers") or {}
    edge_id = stable_id(
        "edge",
        {
            "normalized_claim_id": record.normalized_claim_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
        },
    )
    return {
        "edge_id": edge_id,
        "normalized_claim_id": record.normalized_claim_id,
        "claim_id": record.claim_id,
        "source_id": record.source_id,
        "evidence_id": record.evidence_id,
        "subject": subject,
        "predicate": predicate,
        "object": object_value,
        "qualifiers": qualifiers,
        "schema_version": "graph.edge.v1",
    }


def export_graph_jsonl(config: PipelineConfig, output_path: Optional[Path] = None) -> GraphExportResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "claim_graph.jsonl"
    paths = config.jsonl_paths()
    edges = [
        _edge_from_normalized_claim(record)
        for _, record in read_jsonl_records(paths["claims_normalized"], NormalizedClaimRecord)
    ]
    write_jsonl(output_path, edges)
    return GraphExportResult(output_path=output_path, edge_count=len(edges))
