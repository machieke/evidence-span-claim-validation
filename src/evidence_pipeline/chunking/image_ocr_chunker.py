from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


@dataclass
class ImageOCRChunkResult:
    created: int
    skipped: int


def chunk_image_ocr(config: PipelineConfig, source_id: Optional[str] = None) -> ImageOCRChunkResult:
    paths = config.jsonl_paths()
    existing_chunk_ids = existing_values(paths["chunks"], "chunk_id")
    created = 0
    skipped = 0

    for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord):
        if evidence.source_modality != "image" or evidence.evidence_type != "ocr_text_span":
            continue
        if source_id is not None and evidence.source_id != source_id:
            continue
        chunk_id = stable_id("chunk_ocr", {"evidence_id": evidence.evidence_id})
        if chunk_id in existing_chunk_ids:
            skipped += 1
            continue
        append_jsonl(
            paths["chunks"],
            ChunkRecord(
                chunk_id=chunk_id,
                source_id=evidence.source_id,
                source_modality="image",
                evidence_ids=[evidence.evidence_id],
                primary_evidence_ids=[evidence.evidence_id],
                text=evidence.text or "",
                provenance_summary={
                    "image_id": evidence.provenance.get("image_id"),
                    "ocr_evidence_id": evidence.evidence_id,
                    "bbox": evidence.provenance.get("bbox"),
                },
                chunking_policy={"strategy": "image_ocr_single_evidence"},
            ),
        )
        existing_chunk_ids.add(chunk_id)
        created += 1

    return ImageOCRChunkResult(created=created, skipped=skipped)
