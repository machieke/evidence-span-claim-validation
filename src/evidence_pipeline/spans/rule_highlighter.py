from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.spans.sentence_splitter import split_sentences


@dataclass
class SpanDetectionResult:
    created: int
    skipped: int


_CLAIM_VERBS = re.compile(
    r"\b(is|are|was|were|has|had|have|contains|uses|requires|causes|indicates|reports|reported|"
    r"found|observed|measured|increased|decreased|saw|heard|replaced|will|plans?|committed|"
    r"should|must|can|cannot|did|does|failed|lacks)\b",
    re.IGNORECASE,
)
_NUMBER_OR_DATE = re.compile(r"\b(\d+|\d{4}-\d{2}-\d{2}|yesterday|today|tomorrow|last week|next week)\b", re.IGNORECASE)
_NEGATION = re.compile(r"\b(not|no|never|without|neither|failed to|lacks|cannot|can't|won't|didn't|doesn't)\b", re.IGNORECASE)
_UNCERTAINTY = re.compile(r"\b(appears|seems|may|might|likely|allegedly|reportedly|possibly|suggests|maybe)\b", re.IGNORECASE)
_QUESTION = re.compile(r"\?$")
_COREFERENCE = re.compile(r"\b(it|they|them|this|that|these|those|he|she|him|her|its|their)\b", re.IGNORECASE)
_LOW_VALUE = {
    "ok",
    "okay",
    "thanks",
    "thank you",
    "hi",
    "hello",
    "yes",
    "no",
    "lol",
    "great",
    "sure",
}


def _score_segment(text: str) -> Optional[float]:
    normalized = text.strip().lower()
    if not normalized or normalized in _LOW_VALUE:
        return None
    score = 0.0
    if _CLAIM_VERBS.search(text):
        score += 0.45
    if _NUMBER_OR_DATE.search(text):
        score += 0.2
    if _NEGATION.search(text):
        score += 0.15
    if _UNCERTAINTY.search(text):
        score += 0.15
    if _QUESTION.search(text.strip()):
        score += 0.4
    if score <= 0:
        return None
    return min(score, 0.95)


def _risk_flags(text: str, provenance: Dict[str, object], evidence_risk_flags: List[str]) -> List[str]:
    flags: Set[str] = set(evidence_risk_flags)
    if _COREFERENCE.search(text):
        flags.add("context_dependent_coreference")
    if provenance.get("sender_role") == "assistant":
        flags.add("assistant_generated_text")
    if text.strip().endswith("?"):
        flags.add("question_speech_act")
    return sorted(flags)


def detect_chat_spans(config: PipelineConfig, source_id: Optional[str] = None) -> SpanDetectionResult:
    paths = config.jsonl_paths()
    evidence_by_id = {
        evidence.evidence_id: evidence
        for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord)
        if evidence.source_modality == "chat"
    }
    existing_span_ids = existing_values(paths["spans"], "span_id")
    created = 0
    skipped = 0

    for _, chunk in read_jsonl_records(paths["chunks"], ChunkRecord):
        if chunk.source_modality != "chat":
            continue
        if source_id is not None and chunk.source_id != source_id:
            continue
        for evidence_id in chunk.primary_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence.text is None:
                continue
            for segment in split_sentences(evidence.text):
                score = _score_segment(segment.text)
                if score is None:
                    continue
                span_id = stable_id(
                    "span_chat",
                    {
                        "chunk_id": chunk.chunk_id,
                        "evidence_id": evidence.evidence_id,
                        "char_start": segment.char_start,
                        "char_end": segment.char_end,
                        "detector": "chat_rules_v1",
                    },
                )
                if span_id in existing_span_ids:
                    skipped += 1
                    continue
                append_jsonl(
                    paths["spans"],
                    SpanRecord(
                        span_id=span_id,
                        chunk_id=chunk.chunk_id,
                        source_id=evidence.source_id,
                        source_modality="chat",
                        evidence_id=evidence.evidence_id,
                        text=segment.text,
                        char_start=segment.char_start,
                        char_end=segment.char_end,
                        context_text=chunk.text,
                        label="claim_bearing",
                        score=score,
                        detector={"name": "chat_rules_v1", "version": "0.1.0"},
                        risk_flags=_risk_flags(segment.text, evidence.provenance, evidence.risk_flags),
                    ),
                )
                existing_span_ids.add(span_id)
                created += 1
    return SpanDetectionResult(created=created, skipped=skipped)
