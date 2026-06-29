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


PAGE_FURNITURE_MAX_CHARS = 120
PAGE_FURNITURE_MIN_PAGES = 2
SECTION_HEADING_MAX_CHARS = 120
SECTION_HEADING_MAX_WORDS = 14
_NUMBERED_HEADING_RE = re.compile(r"^(?P<number>\d+(?:\.\d+)+|\d+[.)])\s+(?P<title>.+)$")
_APPENDIX_HEADING_RE = re.compile(
    r"^(?:appendix|attachment|exhibit)\s+[A-Z0-9]+(?:[.: -]+.+)?$",
    re.IGNORECASE,
)
_TITLE_HEADING_VERBS = {
    "appears",
    "are",
    "caused",
    "causes",
    "contains",
    "decreased",
    "found",
    "had",
    "has",
    "have",
    "increased",
    "indicates",
    "is",
    "measured",
    "observed",
    "recommended",
    "reports",
    "reported",
    "requires",
    "replaced",
    "says",
    "shows",
    "uses",
    "was",
    "were",
}


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


def _page_furniture_signature(text: str) -> str:
    cleaned, _ = clean_pdf_text(text)
    normalized = re.sub(r"\b\d+\b", "#", cleaned.lower())
    return re.sub(r"\s+", " ", normalized).strip()


def classify_repeated_pdf_furniture(blocks: List[_ExtractedPDFBlock]) -> Dict[Tuple[int, int], str]:
    by_page: Dict[int, List[_ExtractedPDFBlock]] = {}
    by_signature: Dict[str, List[_ExtractedPDFBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
        signature = _page_furniture_signature(block.text)
        if not signature or len(signature) > PAGE_FURNITURE_MAX_CHARS:
            continue
        by_signature.setdefault(signature, []).append(block)

    first_block_by_page = {
        page: min(page_blocks, key=lambda block: block.block_no).block_no
        for page, page_blocks in by_page.items()
    }
    last_block_by_page = {
        page: max(page_blocks, key=lambda block: block.block_no).block_no
        for page, page_blocks in by_page.items()
    }

    furniture: Dict[Tuple[int, int], str] = {}
    for occurrences in by_signature.values():
        pages = {block.page for block in occurrences}
        if len(pages) < PAGE_FURNITURE_MIN_PAGES:
            continue
        for block in occurrences:
            if block.block_no == first_block_by_page.get(block.page):
                furniture[(block.page, block.block_no)] = "header"
            elif block.block_no == last_block_by_page.get(block.page):
                furniture[(block.page, block.block_no)] = "footer"
    return furniture


def _numbered_heading_level(number: str) -> int:
    normalized = number.rstrip(".)")
    return max(1, normalized.count(".") + 1)


def _is_all_caps_heading(text: str) -> bool:
    if text.endswith((".", "?", "!")):
        return False
    letters = [char for char in text if char.isalpha()]
    if len(letters) < 3:
        return False
    upper_ratio = sum(char.isupper() for char in letters) / len(letters)
    return upper_ratio >= 0.85


def _looks_like_title_heading(text: str) -> bool:
    if text.endswith((".", "?", "!")):
        return False
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]*", text)
    if not 1 <= len(words) <= SECTION_HEADING_MAX_WORDS:
        return False
    lowered = {word.lower() for word in words}
    if lowered & _TITLE_HEADING_VERBS:
        return False
    cased_words = [word for word in words if word[0].isalpha()]
    uppercase_starts = sum(word[0].isupper() for word in cased_words)
    return uppercase_starts >= max(1, len(cased_words) - 1)


def _pdf_heading_candidate(text: str) -> Optional[Tuple[int, str]]:
    cleaned, _ = clean_pdf_text(text)
    if not cleaned or len(cleaned) > SECTION_HEADING_MAX_CHARS:
        return None
    if "\n\n" in cleaned:
        return None

    numbered = _NUMBERED_HEADING_RE.match(cleaned)
    if numbered:
        number = numbered.group("number")
        title = numbered.group("title").strip()
        if title and len(title.split()) <= SECTION_HEADING_MAX_WORDS:
            return _numbered_heading_level(number), cleaned

    if _APPENDIX_HEADING_RE.match(cleaned):
        return 1, cleaned

    if _is_all_caps_heading(cleaned):
        return 1, cleaned

    if _looks_like_title_heading(cleaned):
        return 2, cleaned

    return None


def infer_pdf_section_paths(
    blocks: List[_ExtractedPDFBlock],
    furniture_blocks: Optional[Dict[Tuple[int, int], str]] = None,
) -> Dict[Tuple[int, int], List[str]]:
    furniture_blocks = furniture_blocks or {}
    paths: Dict[Tuple[int, int], List[str]] = {}
    section_stack: List[str] = []

    for block in sorted(blocks, key=lambda item: (item.page, item.block_no)):
        key = (block.page, block.block_no)
        if furniture_blocks.get(key) in {"header", "footer"}:
            paths[key] = []
            continue

        heading = _pdf_heading_candidate(block.text)
        if heading is not None:
            level, title = heading
            if not section_stack and level > 1:
                level = 1
            section_stack = section_stack[: level - 1]
            section_stack.append(title)
        paths[key] = list(section_stack)

    return paths


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


def _record_from_block(
    source_id: str,
    source_file: Path,
    block: _ExtractedPDFBlock,
    char_start: int,
    block_type: str = "text",
    section_path: Optional[List[str]] = None,
) -> PDFBlockRecord:
    cleaned, cleanup_actions = clean_pdf_text(block.text)
    risk_flags = list(block.risk_flags)
    if block_type in {"header", "footer"}:
        cleanup_actions.append("classify_repeated_page_furniture")
        risk_flags.append("page_furniture")
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
        block_type=block_type,
        text=cleaned,
        original_text=block.text,
        cleaned_text=cleaned,
        cleanup_actions=cleanup_actions,
        bbox=block.bbox,
        char_start_document=char_start,
        char_end_document=char_start + len(cleaned),
        section_path=section_path or [],
        extractor=block.extractor,
        risk_flags=sorted(set(risk_flags)),
    )


def ingest_pdf(path: Path, config: PipelineConfig, metadata: Optional[Dict[str, Any]] = None) -> PDFIngestResult:
    metadata = dict(metadata or {})
    sha256 = sha256_file(path)
    source_id = stable_id("src", {"modality": "pdf", "sha256": sha256})
    paths = config.jsonl_paths()

    blocks = _extract_blocks(path)
    furniture_blocks = classify_repeated_pdf_furniture(blocks)
    section_paths = infer_pdf_section_paths(blocks, furniture_blocks)
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
        record = _record_from_block(
            source_id,
            path,
            block,
            char_start,
            block_type=furniture_blocks.get((block.page, block.block_no), "text"),
            section_path=section_paths.get((block.page, block.block_no), []),
        )
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
