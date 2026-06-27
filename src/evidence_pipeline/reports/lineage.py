from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import ensure_parent, read_jsonl


def _rows(path: Path) -> List[dict]:
    return [payload for _, payload in read_jsonl(path)]


def _first_by(rows: List[dict], key: str, value: Any) -> Optional[dict]:
    for row in rows:
        if row.get(key) == value:
            return row
    return None


def _all_by(rows: List[dict], key: str, value: Any) -> List[dict]:
    return [row for row in rows if row.get(key) == value]


def trace_claim(config: PipelineConfig, claim_id: str) -> Dict[str, Any]:
    paths = config.jsonl_paths()
    raw_claims = _rows(paths["claims_raw"])
    validated_claims = _rows(paths["claims_validated"])
    normalized_claims = _rows(paths["claims_normalized"])
    validations = _rows(paths["validations"])
    review_decisions = _rows(paths["review_decisions"])
    audit_events = _rows(paths["audit_events"])
    quarantine = _rows(paths["quarantine"])
    evidence_rows = _rows(paths["evidence"])
    span_rows = _rows(paths["spans"])
    chunk_rows = _rows(paths["chunks"])
    source_rows = _rows(paths["sources"])

    raw_claim = _first_by(raw_claims, "claim_id", claim_id)
    validated_claim = _first_by(validated_claims, "claim_id", claim_id)
    normalized = _all_by(normalized_claims, "claim_id", claim_id)
    claim_validations = _all_by(validations, "claim_id", claim_id)
    claim_reviews = _all_by(review_decisions, "claim_id", claim_id)
    claim_audit_events = _all_by(audit_events, "claim_id", claim_id)
    quarantined = _all_by(quarantine, "claim_id", claim_id)

    anchor = raw_claim or validated_claim or (normalized[0] if normalized else None)
    evidence = None
    span = None
    chunk = None
    source = None
    if anchor:
        evidence = _first_by(evidence_rows, "evidence_id", anchor.get("evidence_id"))
        span_id = anchor.get("span_id")
        if span_id:
            span = _first_by(span_rows, "span_id", span_id)
        if span:
            chunk = _first_by(chunk_rows, "chunk_id", span.get("chunk_id"))
        source = _first_by(source_rows, "source_id", anchor.get("source_id"))

    return {
        "claim_id": claim_id,
        "found": anchor is not None,
        "source": source,
        "evidence": evidence,
        "chunk": chunk,
        "span": span,
        "raw_claim": raw_claim,
        "validations": claim_validations,
        "review_decisions": claim_reviews,
        "audit_events": claim_audit_events,
        "validated_claim": validated_claim,
        "normalized_claims": normalized,
        "quarantine": quarantined,
    }


def write_claim_trace(config: PipelineConfig, claim_id: str, output_path: Path) -> Dict[str, Any]:
    trace = trace_claim(config, claim_id)
    ensure_parent(output_path)
    output_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
    return trace
