from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl, write_jsonl

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE),
    "phone": re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}

ARTIFACT_TEXT_FIELDS = {
    "chat_messages": ("text",),
    "pdf_blocks": ("text",),
    "audio_utterances": ("text",),
    "evidence": ("text",),
    "spans": ("text", "context_text"),
    "claims_raw": ("source_faithful_claim", "evidence_text", "context_used"),
}

ARTIFACT_RECORD_ID_FIELDS = {
    "chat_messages": "message_id",
    "pdf_blocks": "block_id",
    "audio_utterances": "utterance_id",
    "evidence": "evidence_id",
    "spans": "span_id",
    "claims_raw": "claim_id",
}


@dataclass
class PIIDetectionResult:
    output_path: Path
    finding_count: int


def _match_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _redacted_preview(pii_type: str, value: str) -> str:
    if pii_type == "email" and "@" in value:
        local, domain = value.split("@", 1)
        return f"{local[:1]}***@{domain}"
    digits = "".join(character for character in value if character.isdigit())
    if pii_type == "phone" and len(digits) >= 4:
        return f"***-***-{digits[-4:]}"
    if pii_type == "ssn" and len(digits) >= 4:
        return f"***-**-{digits[-4:]}"
    return "[redacted]"


def _artifact_names(artifact: str) -> List[str]:
    if artifact == "all":
        return list(ARTIFACT_TEXT_FIELDS)
    if artifact not in ARTIFACT_TEXT_FIELDS:
        expected = ", ".join(["all"] + sorted(ARTIFACT_TEXT_FIELDS))
        raise ValueError(f"PII detection supports artifacts: {expected}")
    return [artifact]


def _findings_for_text(
    artifact_name: str,
    line_number: int,
    payload: dict,
    field_name: str,
    text: str,
) -> Iterable[dict]:
    record_id_field = ARTIFACT_RECORD_ID_FIELDS[artifact_name]
    record_id = str(payload.get(record_id_field) or f"{artifact_name}:{line_number}")
    for pii_type, pattern in PII_PATTERNS.items():
        for match in pattern.finditer(text):
            raw_match = match.group(0)
            match_hash = _match_hash(raw_match)
            yield {
                "finding_id": stable_id(
                    "pii",
                    {
                        "artifact": artifact_name,
                        "record_id": record_id,
                        "field": field_name,
                        "pii_type": pii_type,
                        "match_hash": match_hash,
                    },
                ),
                "artifact": artifact_name,
                "record_id": record_id,
                "source_id": payload.get("source_id"),
                "evidence_id": payload.get("evidence_id"),
                "claim_id": payload.get("claim_id"),
                "field": field_name,
                "pii_type": pii_type,
                "match_hash": match_hash,
                "redacted_preview": _redacted_preview(pii_type, raw_match),
                "char_start": match.start(),
                "char_end": match.end(),
                "schema_version": "pii.finding.v1",
            }


def _scan_artifact(config: PipelineConfig, artifact_name: str) -> List[dict]:
    paths = config.jsonl_paths()
    findings = []
    for line_number, payload in read_jsonl(paths[artifact_name]):
        for field_name in ARTIFACT_TEXT_FIELDS[artifact_name]:
            value = payload.get(field_name)
            if not isinstance(value, str) or not value:
                continue
            findings.extend(_findings_for_text(artifact_name, line_number, payload, field_name, value))
    return findings


def detect_pii(
    config: PipelineConfig,
    artifact: str = "all",
    output_path: Optional[Path] = None,
) -> PIIDetectionResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "pii_findings.jsonl"
    findings: List[dict] = []
    seen_ids = set()
    for artifact_name in _artifact_names(artifact):
        for finding in _scan_artifact(config, artifact_name):
            if finding["finding_id"] in seen_ids:
                continue
            seen_ids.add(finding["finding_id"])
            findings.append(finding)
    write_jsonl(output_path, findings)
    return PIIDetectionResult(output_path=output_path, finding_count=len(findings))
