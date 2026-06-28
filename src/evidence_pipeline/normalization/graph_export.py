from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord
from evidence_pipeline.schemas.reports import GraphEdgeRecord

GRAPH_EXPORT_VERSION = "graph.export.v1"


@dataclass
class GraphExportResult:
    output_path: Path
    edge_count: int


def _edge_from_normalized_claim(record: NormalizedClaimRecord) -> GraphEdgeRecord:
    normalized = record.normalized_claim
    subject = normalized.get("subject")
    predicate = normalized.get("predicate")
    object_value = normalized.get("object")
    raw_qualifiers = normalized.get("qualifiers") or {}
    qualifiers = raw_qualifiers if isinstance(raw_qualifiers, dict) else {}
    modality = qualifiers.get("modality")
    source_faithful_claim = qualifiers.get("source_faithful_claim")
    edge_id = stable_id(
        "edge",
        {
            "normalized_claim_id": record.normalized_claim_id,
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
        },
    )
    return GraphEdgeRecord(
        edge_id=edge_id,
        normalized_claim_id=record.normalized_claim_id,
        claim_id=record.claim_id,
        source_id=record.source_id,
        evidence_id=record.evidence_id,
        subject=subject,
        predicate=predicate if isinstance(predicate, str) else "",
        object=object_value,
        modality=modality if isinstance(modality, str) else None,
        source_faithful_claim=source_faithful_claim if isinstance(source_faithful_claim, str) else None,
        truth_status=qualifiers.get("truth_status"),
        attribution=qualifiers.get("attribution"),
        qualifiers=qualifiers,
    )


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
