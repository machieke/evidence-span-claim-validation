from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel

PDFBlockType = Literal["text", "image", "table", "caption", "header", "footer", "unknown"]


class PDFBlockRecord(StrictModel):
    block_id: str
    source_id: str
    source_file: str
    page: int
    block_no: int
    block_type: PDFBlockType = "text"
    text: str
    original_text: Optional[str] = None
    cleaned_text: Optional[str] = None
    cleanup_actions: List[str] = Field(default_factory=list)
    bbox: Optional[List[float]] = None
    char_start_document: Optional[int] = None
    char_end_document: Optional[int] = None
    section_path: List[str] = Field(default_factory=list)
    extractor: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "pdf.block.v1"

    @model_validator(mode="after")
    def validate_block(self) -> "PDFBlockRecord":
        if self.page < 1:
            raise ValueError("page must be 1-based")
        if self.block_no < 0:
            raise ValueError("block_no must be non-negative")
        if not self.text.strip():
            raise ValueError("PDF text block must not be empty")
        if self.cleaned_text is None:
            self.cleaned_text = self.text
        if self.original_text is None:
            self.original_text = self.text
        if self.bbox is not None and len(self.bbox) != 4:
            raise ValueError("bbox must contain four coordinates")
        return self
