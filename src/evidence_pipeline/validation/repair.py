from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.audit import AuditEventRecord
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.reports import ClaimRepairSuggestionRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.schemas.validation import ValidationRecord
from evidence_pipeline.spans.sentence_splitter import split_sentences
from evidence_pipeline.validation.text_support import normalize_text_for_matching

REPAIR_SUGGESTION_VERSION = "claim.repair_suggestion.v1"
REPAIR_APPLICATION_VERSION = "claim.repair_application.v1"
_NORMALIZED_CHAR_REPLACEMENTS = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
}


@dataclass
class RepairSuggestionResult:
    output_path: Path
    suggestion_count: int


@dataclass
class RepairApplicationResult:
    applied: int
    skipped: int
    failed: int
    source_ids: List[str]
    claim_ids: List[str]


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


def _normalized_with_offsets(value: str) -> Tuple[str, List[Tuple[int, int]]]:
    chars: List[str] = []
    offsets: List[Tuple[int, int]] = []
    pending_whitespace: Optional[Tuple[int, int]] = None

    for index, char in enumerate(value):
        normalized_chars = unicodedata.normalize("NFKC", char)
        for source, target in _NORMALIZED_CHAR_REPLACEMENTS.items():
            normalized_chars = normalized_chars.replace(source, target)
        for normalized_char in normalized_chars:
            if normalized_char.isspace():
                if chars:
                    if pending_whitespace is None:
                        pending_whitespace = (index, index + 1)
                    else:
                        pending_whitespace = (pending_whitespace[0], index + 1)
                continue
            if pending_whitespace is not None:
                chars.append(" ")
                offsets.append(pending_whitespace)
                pending_whitespace = None
            chars.append(normalized_char)
            offsets.append((index, index + 1))

    return "".join(chars), offsets


def _find_normalized_substring_repair(original: str, support_text: str) -> Optional[str]:
    normalized_original = normalize_text_for_matching(original)
    if not normalized_original:
        return None
    normalized_support, offsets = _normalized_with_offsets(support_text)
    match_start = normalized_support.find(normalized_original)
    if match_start < 0:
        return None
    match_end = match_start + len(normalized_original)
    source_start = offsets[match_start][0]
    source_end = offsets[match_end - 1][1]
    return support_text[source_start:source_end]


def _find_repair(original: str, support_text: str) -> Optional[str]:
    if original in support_text:
        return None
    normalized_original = normalize_text_for_matching(original)
    if normalized_original == normalize_text_for_matching(support_text):
        return support_text
    for segment in split_sentences(support_text):
        if normalized_original == normalize_text_for_matching(segment.text):
            return segment.text
    return _find_normalized_substring_repair(original, support_text)


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


def _repaired_claim_id(suggestion: ClaimRepairSuggestionRecord) -> str:
    return stable_id(
        "claim_repair",
        {
            "claim_id": suggestion.claim_id,
            "repair_id": suggestion.repair_id,
            "suggested_evidence_text": suggestion.suggested_evidence_text,
        },
    )


def _repair_validation_id(repaired_claim_id: str) -> str:
    return stable_id(
        "val",
        {
            "record_id": repaired_claim_id,
            "status": "repaired",
            "version": REPAIR_APPLICATION_VERSION,
        },
    )


def _repair_audit_event_id(repair_id: str, repaired_claim_id: str, status: str) -> str:
    return stable_id(
        "audit",
        {
            "action": "apply_repair",
            "repair_id": repair_id,
            "repaired_claim_id": repaired_claim_id,
            "status": status,
            "version": REPAIR_APPLICATION_VERSION,
        },
    )


def _has_exact_support(
    suggestion: ClaimRepairSuggestionRecord,
    claim: RawClaimRecord,
    evidence_by_id: Dict[str, EvidenceRecord],
    spans_by_id: Dict[str, SpanRecord],
) -> bool:
    for _, support_text in _candidate_support_texts(claim, evidence_by_id, spans_by_id):
        if suggestion.suggested_evidence_text in support_text:
            return True
    return False


def _repaired_claim(claim: RawClaimRecord, suggestion: ClaimRepairSuggestionRecord) -> RawClaimRecord:
    repair_metadata = {
        "repair_id": suggestion.repair_id,
        "original_claim_id": claim.claim_id,
        "reason_codes": suggestion.reason_codes,
        "original_evidence_text": suggestion.original_evidence_text,
    }
    attributes = dict(claim.attributes)
    attributes["repair"] = repair_metadata
    risk_flags = sorted(set(claim.risk_flags) | {"evidence_text_repaired"})
    return claim.model_copy(
        update={
            "claim_id": _repaired_claim_id(suggestion),
            "evidence_text": suggestion.suggested_evidence_text,
            "attributes": attributes,
            "support_status": "raw_extracted",
            "risk_flags": risk_flags,
        }
    )


def _append_repair_audit_event(
    config: PipelineConfig,
    suggestion: ClaimRepairSuggestionRecord,
    repaired_claim_id: str,
    status: str,
    actor_id: Optional[str],
    details: Dict[str, object],
    existing_audit_ids: Set[str],
) -> None:
    audit_event_id = _repair_audit_event_id(suggestion.repair_id, repaired_claim_id, status)
    if audit_event_id in existing_audit_ids:
        return
    append_jsonl(
        config.jsonl_paths()["audit_events"],
        AuditEventRecord(
            audit_event_id=audit_event_id,
            action="apply_repair",
            actor_id=actor_id,
            target_type="claim",
            target_id=repaired_claim_id,
            source_id=suggestion.source_id,
            evidence_id=suggestion.evidence_id,
            claim_id=repaired_claim_id,
            status=status,
            details=details,
        ),
    )
    existing_audit_ids.add(audit_event_id)


def apply_evidence_repairs(
    config: PipelineConfig,
    input_path: Optional[Path] = None,
    repair_ids: Optional[Sequence[str]] = None,
    actor_id: Optional[str] = None,
) -> RepairApplicationResult:
    if input_path is None:
        input_path = config.paths.reports_dir / "claim_repairs.jsonl"
    selected_repair_ids = set(repair_ids or [])
    paths = config.jsonl_paths()
    claims_by_id = {record.claim_id: record for _, record in read_jsonl_records(paths["claims_raw"], RawClaimRecord)}
    evidence_by_id = {record.evidence_id: record for _, record in read_jsonl_records(paths["evidence"], EvidenceRecord)}
    spans_by_id = {record.span_id: record for _, record in read_jsonl_records(paths["spans"], SpanRecord)}
    existing_claim_ids = existing_values(paths["claims_raw"], "claim_id")
    existing_validation_ids = existing_values(paths["validations"], "validation_id")
    existing_audit_ids = existing_values(paths["audit_events"], "audit_event_id")

    applied = 0
    skipped = 0
    failed = 0
    source_ids: Set[str] = set()
    claim_ids: Set[str] = set()
    for _, suggestion in read_jsonl_records(input_path, ClaimRepairSuggestionRecord):
        if selected_repair_ids and suggestion.repair_id not in selected_repair_ids:
            continue
        source_ids.add(suggestion.source_id)
        claim_ids.add(suggestion.claim_id)
        claim = claims_by_id.get(suggestion.claim_id)
        repaired_claim_id = _repaired_claim_id(suggestion)
        claim_ids.add(repaired_claim_id)
        if claim is None:
            failed += 1
            _append_repair_audit_event(
                config,
                suggestion,
                repaired_claim_id,
                "failed",
                actor_id,
                {"reason": "missing_claim", "original_claim_id": suggestion.claim_id},
                existing_audit_ids,
            )
            continue
        if claim.evidence_text != suggestion.original_evidence_text:
            failed += 1
            _append_repair_audit_event(
                config,
                suggestion,
                repaired_claim_id,
                "failed",
                actor_id,
                {"reason": "claim_evidence_text_changed", "original_claim_id": suggestion.claim_id},
                existing_audit_ids,
            )
            continue
        if not _has_exact_support(suggestion, claim, evidence_by_id, spans_by_id):
            failed += 1
            _append_repair_audit_event(
                config,
                suggestion,
                repaired_claim_id,
                "failed",
                actor_id,
                {"reason": "suggested_text_not_exact_support", "original_claim_id": suggestion.claim_id},
                existing_audit_ids,
            )
            continue
        if repaired_claim_id in existing_claim_ids:
            skipped += 1
            _append_repair_audit_event(
                config,
                suggestion,
                repaired_claim_id,
                "skipped",
                actor_id,
                {"reason": "repaired_claim_exists", "original_claim_id": suggestion.claim_id},
                existing_audit_ids,
            )
            continue

        repaired_claim = _repaired_claim(claim, suggestion)
        append_jsonl(paths["claims_raw"], repaired_claim)
        existing_claim_ids.add(repaired_claim.claim_id)
        validation_id = _repair_validation_id(repaired_claim.claim_id)
        if validation_id not in existing_validation_ids:
            append_jsonl(
                paths["validations"],
                ValidationRecord(
                    validation_id=validation_id,
                    claim_id=repaired_claim.claim_id,
                    record_id=repaired_claim.claim_id,
                    stage="apply_repairs",
                    status="repaired",
                    warnings=suggestion.reason_codes,
                    validator_version=REPAIR_APPLICATION_VERSION,
                    metadata={
                        "repair_id": suggestion.repair_id,
                        "original_claim_id": suggestion.claim_id,
                        "original_evidence_text": suggestion.original_evidence_text,
                        "suggested_evidence_text": suggestion.suggested_evidence_text,
                        "support_scope": suggestion.support_scope,
                    },
                ),
            )
            existing_validation_ids.add(validation_id)
        _append_repair_audit_event(
            config,
            suggestion,
            repaired_claim.claim_id,
            "created",
            actor_id,
            {
                "repair_id": suggestion.repair_id,
                "original_claim_id": suggestion.claim_id,
                "reason_codes": suggestion.reason_codes,
            },
            existing_audit_ids,
        )
        claims_by_id[repaired_claim.claim_id] = repaired_claim
        applied += 1

    return RepairApplicationResult(
        applied=applied,
        skipped=skipped,
        failed=failed,
        source_ids=sorted(source_ids),
        claim_ids=sorted(claim_ids),
    )
