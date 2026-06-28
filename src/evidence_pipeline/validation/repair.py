from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.reports import ClaimRepairSuggestionRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.spans.sentence_splitter import split_sentences
from evidence_pipeline.validation.text_support import normalize_text_for_matching

REPAIR_SUGGESTION_VERSION = "claim.repair_suggestion.v1"


@dataclass
class RepairSuggestionResult:
    output_path: Path
    suggestion_count: int


REPAIR_REASON_CODES = {"evidence_not_exact_substring"}


def _candidate_support_texts(
    claim: RawClaimRecord,
    evidence_by_id: Dict[str, EvidenceRecord],
    spans_by_id: Dict[str, SpanRecord],
) -> List[Tuple[str, str]]:
    candidates: List[Tuple[str, str]] = []
    if claim.span_id and claim.span_id in spans_by_id and spans_by_id[claim.span_id].text:
        candidates.append(("span", spans_by_id[claim.span_id].text or ""))
    evidence = evidence_by_id.get(claim.evidence_id)
    if evidence is not None and evidence.text:
        candidates.append(("evidence", evidence.text))
    return candidates


def _find_repair(original: str, support_text: str) -> Optional[str]:
    if original in support_text:
        return None
    normalized_original = normalize_text_for_matching(original)
    if normalized_original == normalize_text_for_matching(support_text):
        return support_text
    for segment in split_sentences(support_text):
        if normalized_original == normalize_text_for_matching(segment.text):
            return segment.text
    return None


def suggest_evidence_repairs(
    config: PipelineConfig,
    output_path: Optional[Path] = None,
    only_reason_codes: Optional[Sequence[str]] = None,
) -> RepairSuggestionResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "claim_repairs.jsonl"
    requested_reasons = set(only_reason_codes or [])
    paths = config.jsonl_paths()
    evidence_by_id = {record.evidence_id: record for _, record in read_jsonl_records(paths["evidence"], EvidenceRecord)}
    spans_by_id = {record.span_id: record for _, record in read_jsonl_records(paths["spans"], SpanRecord)}
    suggestions = []

    for _, claim in read_jsonl_records(paths["claims_raw"], RawClaimRecord):
        if requested_reasons and "evidence_not_exact_substring" not in requested_reasons:
            continue
        if not claim.evidence_text:
            continue
        for support_scope, support_text in _candidate_support_texts(claim, evidence_by_id, spans_by_id):
            repaired = _find_repair(claim.evidence_text, support_text)
            if repaired is None:
                continue
            suggestions.append(
                ClaimRepairSuggestionRecord(
                    repair_id=stable_id(
                        "repair",
                        {
                            "claim_id": claim.claim_id,
                            "original_evidence_text": claim.evidence_text,
                            "suggested_evidence_text": repaired,
                        },
                    ),
                    claim_id=claim.claim_id,
                    source_id=claim.source_id,
                    evidence_id=claim.evidence_id,
                    span_id=claim.span_id,
                    reason_codes=["evidence_not_exact_substring"],
                    original_evidence_text=claim.evidence_text,
                    suggested_evidence_text=repaired,
                    support_scope=support_scope,
                ).model_dump(mode="json", exclude_none=True)
            )
            break

    write_jsonl(output_path, suggestions)
    return RepairSuggestionResult(output_path=output_path, suggestion_count=len(suggestions))
