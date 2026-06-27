from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from pydantic import ValidationError

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.validation import QuarantineRecord, ValidationRecord

SCHEMA_REPAIR_VERSION = "schema_repair.v1"

FIELD_ALIASES = {
    "claim": "source_faithful_claim",
    "claim_text": "source_faithful_claim",
    "text": "source_faithful_claim",
    "evidence": "evidence_text",
    "supporting_text": "evidence_text",
}


@dataclass
class RawClaimImportResult:
    imported: int
    repaired: int
    quarantined: int
    skipped: int


def _candidate_payloads(path: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return
    if text[0] in "[{":
        decoded = json.loads(text)
        if isinstance(decoded, dict) and isinstance(decoded.get("claims"), list):
            decoded = decoded["claims"]
        if isinstance(decoded, dict):
            yield "1", decoded
            return
        if not isinstance(decoded, list):
            raise ValueError("raw claim import JSON must be an object, a list, or an object with a claims list")
        for index, item in enumerate(decoded, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"raw claim candidate {index} must be a JSON object")
            yield str(index), item
        return
    for line_number, payload in read_jsonl(path):
        yield str(line_number), payload


def _repair_attribution(value: object, source_modality: object) -> Tuple[object, List[str]]:
    if not isinstance(value, str):
        return value, []
    attribution_type = "unknown"
    if source_modality in {"chat", "audio"}:
        attribution_type = "speaker"
    elif source_modality == "pdf":
        attribution_type = "document"
    elif source_modality == "image":
        attribution_type = "model"
    return {"type": attribution_type, "agent": value}, ["coerce_attribution_string"]


def _repair_confidence(value: object) -> Tuple[object, List[str]]:
    actions = []
    if isinstance(value, str):
        try:
            value = float(value.strip().rstrip("%"))
            actions.append("coerce_confidence_number")
        except ValueError:
            return value, actions
    if isinstance(value, (int, float)) and not isinstance(value, bool) and 1 < value <= 100:
        value = float(value) / 100
        actions.append("scale_confidence_percent")
    return value, actions


def repair_raw_claim_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    repaired = dict(payload)
    actions: List[str] = []
    for alias, canonical in FIELD_ALIASES.items():
        if canonical not in repaired and alias in repaired:
            repaired[canonical] = repaired.pop(alias)
            actions.append(f"rename_{alias}_to_{canonical}")

    if "attribution" in repaired:
        repaired["attribution"], attribution_actions = _repair_attribution(
            repaired["attribution"],
            repaired.get("source_modality"),
        )
        actions.extend(attribution_actions)

    if "confidence" in repaired:
        repaired["confidence"], confidence_actions = _repair_confidence(repaired["confidence"])
        actions.extend(confidence_actions)

    if isinstance(repaired.get("risk_flags"), str):
        repaired["risk_flags"] = [repaired["risk_flags"]]
        actions.append("coerce_risk_flags_string")

    if not repaired.get("claim_id") and repaired.get("source_id") and repaired.get("evidence_id") and repaired.get("source_faithful_claim"):
        repaired["claim_id"] = stable_id(
            "claim_import",
            {
                "source_id": repaired["source_id"],
                "evidence_id": repaired["evidence_id"],
                "source_faithful_claim": repaired["source_faithful_claim"],
            },
        )
        actions.append("generate_claim_id")

    return repaired, actions


def _validation_id(record_id: str, status: str) -> str:
    return stable_id("schema_val", {"record_id": record_id, "status": status, "version": SCHEMA_REPAIR_VERSION})


def _quarantine_id(candidate_id: str, repaired_payload: Dict[str, Any]) -> str:
    return stable_id(
        "q_schema",
        {
            "candidate_id": candidate_id,
            "payload": repaired_payload,
            "version": SCHEMA_REPAIR_VERSION,
        },
    )


def import_raw_claim_candidates(config: PipelineConfig, input_path: Path) -> RawClaimImportResult:
    paths = config.jsonl_paths()
    existing_claim_ids = existing_values(paths["claims_raw"], "claim_id")
    existing_validation_ids = existing_values(paths["validations"], "validation_id")
    existing_quarantine_ids = existing_values(paths["quarantine"], "quarantine_id")
    imported = 0
    repaired_count = 0
    quarantined = 0
    skipped = 0

    for candidate_id, payload in _candidate_payloads(input_path):
        repaired_payload, repair_actions = repair_raw_claim_payload(payload)
        record_id = str(repaired_payload.get("claim_id") or f"{input_path}:{candidate_id}")
        try:
            claim = RawClaimRecord.model_validate(repaired_payload)
        except ValidationError as exc:
            quarantine_id = _quarantine_id(candidate_id, repaired_payload)
            if quarantine_id in existing_quarantine_ids:
                skipped += 1
                continue
            append_jsonl(
                paths["quarantine"],
                QuarantineRecord(
                    quarantine_id=quarantine_id,
                    record_type="claim_candidate",
                    record_id=record_id,
                    source_id=repaired_payload.get("source_id"),
                    evidence_id=repaired_payload.get("evidence_id"),
                    claim_id=repaired_payload.get("claim_id"),
                    stage="import_raw_claims",
                    reason_codes=["schema_invalid_after_repair"],
                    warnings=repair_actions,
                    payload={
                        "candidate_id": candidate_id,
                        "errors": json.loads(exc.json()),
                        "repaired_payload": repaired_payload,
                    },
                ),
            )
            existing_quarantine_ids.add(quarantine_id)
            quarantined += 1
            validation_id = _validation_id(record_id, "schema_invalid")
            if validation_id not in existing_validation_ids:
                append_jsonl(
                    paths["validations"],
                    ValidationRecord(
                        validation_id=validation_id,
                        claim_id=repaired_payload.get("claim_id"),
                        record_id=record_id,
                        stage="import_raw_claims",
                        status="schema_invalid",
                        errors=["schema_invalid_after_repair"],
                        warnings=repair_actions,
                        validator_version=SCHEMA_REPAIR_VERSION,
                    ),
                )
                existing_validation_ids.add(validation_id)
            continue

        if claim.claim_id in existing_claim_ids:
            skipped += 1
            continue
        append_jsonl(paths["claims_raw"], claim)
        existing_claim_ids.add(claim.claim_id)
        imported += 1
        if repair_actions:
            repaired_count += 1

        status = "repaired" if repair_actions else "schema_valid"
        validation_id = _validation_id(claim.claim_id, status)
        if validation_id in existing_validation_ids:
            continue
        append_jsonl(
            paths["validations"],
            ValidationRecord(
                validation_id=validation_id,
                claim_id=claim.claim_id,
                record_id=claim.claim_id,
                stage="import_raw_claims",
                status=status,
                warnings=repair_actions,
                validator_version=SCHEMA_REPAIR_VERSION,
            ),
        )
        existing_validation_ids.add(validation_id)

    return RawClaimImportResult(imported=imported, repaired=repaired_count, quarantined=quarantined, skipped=skipped)
