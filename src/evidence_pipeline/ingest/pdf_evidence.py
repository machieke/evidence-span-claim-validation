from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.pdf import PDFBlockRecord


@dataclass
class PDFEvidenceResult:
    created: int
    skipped: int


def _pdf_evidence_id(block: PDFBlockRecord) -> str:
    return stable_id("ev_pdf", {"source_id": block.source_id, "block_id": block.block_id})


def build_pdf_evidence(config: PipelineConfig, source_id: Optional[str] = None) -> PDFEvidenceResult:
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["evidence"], "evidence_id")
    created = 0
    skipped = 0

    for _, block in read_jsonl_records(paths["pdf_blocks"], PDFBlockRecord):
        if source_id is not None and block.source_id != source_id:
            continue
        if block.block_type in {"header", "footer"}:
            skipped += 1
            continue
        evidence_id = _pdf_evidence_id(block)
        if evidence_id in existing_ids:
            skipped += 1
            continue
        provenance: Dict[str, object] = {
            "page": block.page,
            "block_id": block.block_id,
            "block_no": block.block_no,
            "bbox": block.bbox,
            "char_start": block.char_start_document,
            "char_end": block.char_end_document,
            "section_path": block.section_path,
            "extractor": block.extractor,
        }
        append_jsonl(
            paths["evidence"],
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=block.source_id,
                source_modality="pdf",
                evidence_type="text_span",
                text=block.cleaned_text or block.text,
                provenance={key: value for key, value in provenance.items() if value is not None},
                risk_flags=block.risk_flags,
            ),
        )
        existing_ids.add(evidence_id)
        created += 1

    return PDFEvidenceResult(created=created, skipped=skipped)
