from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.validation.text_support import contains_negation, contains_uncertainty

RULE_EXTRACTOR_VERSION = "rules.v1"


@dataclass
class ClaimExtractionResult:
    created: int
    skipped: int


def _claim_id(span: SpanRecord, extractor: str) -> str:
    return stable_id(
        "claim",
        {
            "span_id": span.span_id,
            "evidence_id": span.evidence_id,
            "text": span.text,
            "extractor": extractor,
        },
    )


def _modality_for_text(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "question_asked"
    if contains_negation(stripped):
        return "negated"
    if contains_uncertainty(stripped):
        return "uncertain_observation"
    return "asserted"


def _chat_claim_text(span: SpanRecord) -> str:
    if (span.text or "").strip().endswith("?"):
        return f"The speaker asked: {span.text}"
    return f"The speaker asserted: {span.text}"


def _pdf_claim_text(span: SpanRecord) -> str:
    return f"The document states: {span.text}"


def _attribution(span: SpanRecord, evidence: EvidenceRecord) -> Dict[str, Optional[str]]:
    if span.source_modality == "chat":
        agent = evidence.provenance.get("sender_id") or evidence.provenance.get("sender_display_name")
        return {"type": "speaker", "agent": str(agent) if agent is not None else None}
    if span.source_modality == "audio":
        agent = evidence.provenance.get("speaker")
        return {"type": "speaker", "agent": str(agent) if agent is not None else None}
    if span.source_modality == "pdf":
        return {"type": "document", "agent": evidence.source_id}
    return {"type": "unknown", "agent": None}


def _truth_status(span: SpanRecord) -> str:
    if span.source_modality in {"chat", "audio"}:
        return "speaker_asserted_unverified"
    if span.source_modality == "image":
        return "model_observation_unverified"
    return "source_asserted_unverified"


def _source_faithful_claim(span: SpanRecord) -> str:
    if span.source_modality == "chat":
        return _chat_claim_text(span)
    if span.source_modality == "pdf":
        return _pdf_claim_text(span)
    return f"The source states: {span.text}"


def _risk_flags(span: SpanRecord, evidence: EvidenceRecord) -> List[str]:
    return sorted(set(span.risk_flags) | set(evidence.risk_flags))


def _raw_claim_from_span(span: SpanRecord, evidence: EvidenceRecord) -> RawClaimRecord:
    if not span.text:
        raise ValueError("text span is required for deterministic extraction")
    risk_flags = _risk_flags(span, evidence)
    context_dependent = "context_dependent_coreference" in risk_flags and bool(span.context_text)
    return RawClaimRecord(
        claim_id=_claim_id(span, RULE_EXTRACTOR_VERSION),
        source_id=span.source_id,
        source_modality=span.source_modality,
        span_id=span.span_id,
        evidence_id=span.evidence_id,
        claim_type="attributed_text_claim",
        source_faithful_claim=_source_faithful_claim(span),
        subject=None,
        predicate=None,
        object=None,
        quantity=None,
        attributes={"extractor": RULE_EXTRACTOR_VERSION},
        modality=_modality_for_text(span.text),
        evidence_text=span.text,
        context_dependent=context_dependent,
        context_used=span.context_text if context_dependent else None,
        attribution=_attribution(span, evidence),
        truth_status=_truth_status(span),
        confidence=span.score if span.score is not None else 0.5,
        model={
            "provider": "deterministic",
            "model": RULE_EXTRACTOR_VERSION,
            "prompt_version": None,
        },
        support_status="raw_extracted",
        risk_flags=risk_flags,
    )


def extract_claims_from_spans(
    config: PipelineConfig,
    modality: str = "all",
    source_id: Optional[str] = None,
) -> ClaimExtractionResult:
    if modality not in {"all", "chat", "pdf"}:
        raise ValueError("baseline extractor currently supports all, chat, or pdf")

    paths = config.jsonl_paths()
    evidence_by_id = {
        evidence.evidence_id: evidence
        for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord)
    }
    existing_claim_ids = existing_values(paths["claims_raw"], "claim_id")
    created = 0
    skipped = 0

    for _, span in read_jsonl_records(paths["spans"], SpanRecord):
        if span.label != "claim_bearing":
            continue
        if modality != "all" and span.source_modality != modality:
            continue
        if source_id is not None and span.source_id != source_id:
            continue
        if span.source_modality not in {"chat", "pdf"}:
            continue
        evidence = evidence_by_id.get(span.evidence_id)
        if evidence is None:
            skipped += 1
            continue
        record = _raw_claim_from_span(span, evidence)
        if record.claim_id in existing_claim_ids:
            skipped += 1
            continue
        append_jsonl(paths["claims_raw"], record)
        existing_claim_ids.add(record.claim_id)
        created += 1

    return ClaimExtractionResult(created=created, skipped=skipped)
