from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values
from evidence_pipeline.schemas.pdf import PDFBlockRecord
from evidence_pipeline.schemas.sources import SourceRecord


@dataclass
class PDFIngestResult:
    source_id: str
    source_created: bool
    blocks_created: int
    blocks_skipped: int
    extractor: str


@dataclass
class _ExtractedPDFBlock:
    page: int
    block_no: int
    text: str
    bbox: Optional[List[float]]
    extractor: str
    risk_flags: List[str]


def clean_pdf_text(text: str) -> Tuple[str, List[str]]:
    cleaned = text.strip()
    actions: List[str] = []
    repaired = re.sub(r"(?<=\w)-\n(?=\w)", "", cleaned)
    if repaired != cleaned:
        cleaned = repaired
        actions.append("repair_hyphenation")
    joined = re.sub(r"(?<!\n)\n(?!\n)", " ", cleaned)
    if joined != cleaned:
        cleaned = joined
        actions.append("join_wrapped_lines")
    collapsed = re.sub(r"[ \t]+", " ", cleaned).strip()
    if collapsed != cleaned:
        cleaned = collapsed
        actions.append("collapse_whitespace")
    return cleaned, actions


def _extract_with_pymupdf(path: Path) -> Optional[List[_ExtractedPDFBlock]]:
    try:
        import fitz  # type: ignore
    except Exception:
        return None

    blocks: List[_ExtractedPDFBlock] = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document, start=1):
            for block_index, block in enumerate(page.get_text("blocks")):
                if len(block) < 5:
                    continue
                x0, y0, x1, y1, text = block[:5]
                if not str(text).strip():
                    continue
                blocks.append(
                    _ExtractedPDFBlock(
                        page=page_index,
                        block_no=block_index,
                        text=str(text),
                        bbox=[float(x0), float(y0), float(x1), float(y1)],
                        extractor="pymupdf",
                        risk_flags=[],
                    )
                )
    return blocks


def _extract_with_pypdf(path: Path) -> List[_ExtractedPDFBlock]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF ingestion requires PyMuPDF or pypdf to be installed") from exc

    reader = PdfReader(str(path))
    blocks: List[_ExtractedPDFBlock] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        blocks.append(
            _ExtractedPDFBlock(
                page=page_index,
                block_no=0,
                text=text,
                bbox=None,
                extractor="pypdf",
                risk_flags=["missing_bbox"],
            )
        )
    return blocks


def _extract_blocks(path: Path) -> List[_ExtractedPDFBlock]:
    blocks = _extract_with_pymupdf(path)
    if blocks is not None:
        return blocks
    return _extract_with_pypdf(path)


def _record_from_block(source_id: str, source_file: Path, block: _ExtractedPDFBlock, char_start: int) -> PDFBlockRecord:
    cleaned, cleanup_actions = clean_pdf_text(block.text)
    block_id = stable_id(
        "pdf_blk",
        {
            "source_id": source_id,
            "page": block.page,
            "block_no": block.block_no,
            "text": cleaned,
        },
    )
    return PDFBlockRecord(
        block_id=block_id,
        source_id=source_id,
        source_file=str(source_file),
        page=block.page,
        block_no=block.block_no,
        block_type="text",
        text=cleaned,
        original_text=block.text,
        cleaned_text=cleaned,
        cleanup_actions=cleanup_actions,
        bbox=block.bbox,
        char_start_document=char_start,
        char_end_document=char_start + len(cleaned),
        extractor=block.extractor,
        risk_flags=block.risk_flags,
    )


def ingest_pdf(path: Path, config: PipelineConfig, metadata: Optional[Dict[str, Any]] = None) -> PDFIngestResult:
    metadata = dict(metadata or {})
    sha256 = sha256_file(path)
    source_id = stable_id("src", {"modality": "pdf", "sha256": sha256})
    paths = config.jsonl_paths()

    blocks = _extract_blocks(path)
    extractor = blocks[0].extractor if blocks else "none"
    existing_sources = existing_values(paths["sources"], "source_id")
    source_created = source_id not in existing_sources
    if source_created:
        append_jsonl(
            paths["sources"],
            SourceRecord(
                source_id=source_id,
                source_modality="pdf",
                source_file=str(path),
                sha256=sha256,
                metadata={**metadata, "block_count": len(blocks), "extractor": extractor},
            ),
        )

    existing_block_ids = existing_values(paths["pdf_blocks"], "block_id")
    created = 0
    skipped = 0
    char_start = 0
    for block in blocks:
        record = _record_from_block(source_id, path, block, char_start)
        char_start = (record.char_end_document or char_start) + 1
        if record.block_id in existing_block_ids:
            skipped += 1
            continue
        append_jsonl(paths["pdf_blocks"], record)
        existing_block_ids.add(record.block_id)
        created += 1

    return PDFIngestResult(
        source_id=source_id,
        source_created=source_created,
        blocks_created=created,
        blocks_skipped=skipped,
        extractor=extractor,
    )
