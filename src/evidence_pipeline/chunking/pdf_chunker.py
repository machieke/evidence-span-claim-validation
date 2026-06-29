from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


@dataclass
class PDFChunkResult:
    created: int
    skipped: int


def _sort_key(evidence: EvidenceRecord) -> Tuple[int, int, str]:
    return (
        int(evidence.provenance.get("page") or 0),
        int(evidence.provenance.get("block_no") or 0),
        evidence.evidence_id,
    )


def _section_path(evidence: EvidenceRecord) -> Tuple[str, ...]:
    value = evidence.provenance.get("section_path") or []
    if isinstance(value, list):
        return tuple(str(part) for part in value)
    return (str(value),)


def _section_paths(evidence_batch: List[EvidenceRecord]) -> List[List[str]]:
    paths: List[List[str]] = []
    seen = set()
    for evidence in evidence_batch:
        path = _section_path(evidence)
        if not path or path in seen:
            continue
        paths.append(list(path))
        seen.add(path)
    return paths


def _flush_chunk(
    config: PipelineConfig,
    evidence_batch: List[EvidenceRecord],
    overlap_ids: List[str],
    target_tokens: int,
    overlap_tokens: int,
    existing_chunk_ids: set,
    carry_overlap: bool = True,
) -> Tuple[int, List[str]]:
    if not evidence_batch:
        return 0, overlap_ids
    paths = config.jsonl_paths()
    primary_ids = [record.evidence_id for record in evidence_batch]
    evidence_ids = overlap_ids + primary_ids
    chunk_id = stable_id(
        "chunk_pdf",
        {
            "primary_evidence_ids": primary_ids,
            "target_tokens": target_tokens,
            "overlap_tokens": overlap_tokens,
        },
    )
    if chunk_id in existing_chunk_ids:
        return 0, primary_ids[-1:] if carry_overlap else []
    pages = sorted({record.provenance.get("page") for record in evidence_batch if record.provenance.get("page") is not None})
    section_paths = _section_paths(evidence_batch)
    append_jsonl(
        paths["chunks"],
        ChunkRecord(
            chunk_id=chunk_id,
            source_id=evidence_batch[0].source_id,
            source_modality="pdf",
            evidence_ids=evidence_ids,
            primary_evidence_ids=primary_ids,
            overlap_evidence_ids=overlap_ids,
            text="\n\n".join(record.text or "" for record in evidence_batch),
            provenance_summary={
                "pages": pages,
                "block_ids": [record.provenance.get("block_id") for record in evidence_batch],
                "section_paths": section_paths,
            },
            chunking_policy={
                "strategy": "section_page_block_token_fallback",
                "target_tokens": target_tokens,
                "overlap_tokens": overlap_tokens,
            },
        ),
    )
    existing_chunk_ids.add(chunk_id)
    return 1, primary_ids[-1:] if carry_overlap else []


def chunk_pdf(config: PipelineConfig, source_id: Optional[str] = None, target_tokens: int = 1200, overlap_tokens: int = 150) -> PDFChunkResult:
    paths = config.jsonl_paths()
    evidence_records = [
        record
        for _, record in read_jsonl_records(paths["evidence"], EvidenceRecord)
        if record.source_modality == "pdf" and (source_id is None or record.source_id == source_id)
    ]
    grouped: Dict[str, List[EvidenceRecord]] = {}
    for evidence in evidence_records:
        grouped.setdefault(evidence.source_id, []).append(evidence)
    existing_chunk_ids = existing_values(paths["chunks"], "chunk_id")

    created = 0
    skipped = 0
    char_budget = max(1, target_tokens * 4)
    for source_records in grouped.values():
        source_records.sort(key=_sort_key)
        batch: List[EvidenceRecord] = []
        batch_chars = 0
        overlap_ids: List[str] = []
        current_section: Tuple[str, ...] = ()
        for evidence in source_records:
            text_len = len(evidence.text or "")
            section_path = _section_path(evidence)
            section_changed = bool(batch) and section_path != current_section
            budget_exceeded = bool(batch) and batch_chars + text_len > char_budget
            if section_changed or budget_exceeded:
                count, overlap_ids = _flush_chunk(
                    config,
                    batch,
                    overlap_ids,
                    target_tokens,
                    overlap_tokens,
                    existing_chunk_ids,
                    carry_overlap=not section_changed,
                )
                if count:
                    created += count
                else:
                    skipped += 1
                batch = []
                batch_chars = 0
            current_section = section_path
            batch.append(evidence)
            batch_chars += text_len
        if batch:
            count, overlap_ids = _flush_chunk(config, batch, overlap_ids, target_tokens, overlap_tokens, existing_chunk_ids)
            if count:
                created += count
            else:
                skipped += 1

    return PDFChunkResult(created=created, skipped=skipped)
