from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import ensure_parent, read_jsonl, write_jsonl
from evidence_pipeline.schemas.reports import AcceptanceCheckRecord

ACCEPTANCE_CHECK_VERSION = "acceptance.check.v1"

MAX_DETAIL_RECORDS = 20

CORE_ID_FIELDS = {
    "sources": "source_id",
    "chat_messages": "message_id",
    "pdf_blocks": "block_id",
    "audio_utterances": "utterance_id",
    "images": "image_id",
    "image_regions": "region_id",
    "image_region_embeddings": "embedding_id",
    "image_feature_clusters": "feature_cluster_id",
    "evidence": "evidence_id",
    "chunks": "chunk_id",
    "spans": "span_id",
    "claims_raw": "claim_id",
    "validations": "validation_id",
    "claims_validated": "claim_id",
    "claims_normalized": "normalized_claim_id",
    "jobs": "job_id",
    "review_decisions": "review_id",
    "audit_events": "audit_event_id",
    "errors": "error_id",
    "quarantine": "quarantine_id",
}

IMAGE_SAFE_MODALITIES = {"model_observation", "uncertain_observation", "hypothetical"}
IMAGE_SAFE_TRUTH_STATUSES = {"model_observation_unverified", "human_confirmed"}
IMAGE_SAFE_ATTRIBUTIONS = {"model", "human_reviewer"}


@dataclass
class AcceptanceCheckResult:
    output_path: Path
    checks: List[dict]
    passed: bool


def _rows(path: Path) -> List[dict]:
    return [payload for _, payload in read_jsonl(path)]


def _optional_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return _rows(path)


def _index_by(rows: Iterable[dict], key: str) -> Dict[str, dict]:
    indexed = {}
    for row in rows:
        value = row.get(key)
        if isinstance(value, str) and value:
            indexed[value] = row
    return indexed


def _present(value: object) -> bool:
    return value is not None and value != "" and value != [] and value != {}


def _details(failures: List[dict]) -> List[dict]:
    if len(failures) <= MAX_DETAIL_RECORDS:
        return failures
    return failures[:MAX_DETAIL_RECORDS] + [{"omitted_failures": len(failures) - MAX_DETAIL_RECORDS}]


def _check_record(
    check_id: str,
    description: str,
    total: int,
    failures: List[dict],
) -> AcceptanceCheckRecord:
    return AcceptanceCheckRecord(
        check_id=check_id,
        description=description,
        status="failed" if failures else "passed",
        total=total,
        failed=len(failures),
        details=_details(failures),
        schema_version=ACCEPTANCE_CHECK_VERSION,
    )


def _presence_check(check_id: str, description: str, artifact_name: str, rows: List[dict]) -> AcceptanceCheckRecord:
    failures = []
    if not rows:
        failures.append({"artifact": artifact_name, "reason": "missing_required_records"})
    return _check_record(check_id, description, 1, failures)


def _accepted_claims(artifacts: Dict[str, List[dict]]) -> List[dict]:
    return [
        claim
        for claim in artifacts["claims_validated"]
        if claim.get("support_status") == "accepted_extracted"
    ]


def _text_like_claim_requires_exact_match(claim: dict, evidence: Optional[dict]) -> bool:
    if claim.get("source_modality") != "image":
        return True
    if evidence is not None and evidence.get("evidence_type") == "ocr_text_span":
        return True
    return bool(claim.get("evidence_text")) and bool(evidence and evidence.get("text"))


def _artifact_ids_unique(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    failures: List[dict] = []
    total = 0
    for artifact_name, id_field in CORE_ID_FIELDS.items():
        seen = set()
        for row in artifacts.get(artifact_name, []):
            total += 1
            record_id = row.get(id_field)
            if not isinstance(record_id, str) or not record_id:
                failures.append(
                    {
                        "artifact": artifact_name,
                        "id_field": id_field,
                        "reason": "missing_record_id",
                    }
                )
                continue
            if record_id in seen:
                failures.append(
                    {
                        "artifact": artifact_name,
                        "id_field": id_field,
                        "record_id": record_id,
                        "reason": "duplicate_record_id",
                    }
                )
            seen.add(record_id)
    return _check_record(
        "artifact_ids_unique",
        "Configured JSONL artifacts do not contain duplicate or missing record identifiers.",
        total,
        failures,
    )


def _evidence_records_created(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    sources_by_id = _index_by(artifacts["sources"], "source_id")
    failures = []
    for evidence in artifacts["evidence"]:
        source_id = evidence.get("source_id")
        if source_id not in sources_by_id:
            failures.append(
                {
                    "evidence_id": evidence.get("evidence_id"),
                    "source_id": source_id,
                    "reason": "missing_source",
                }
            )
    if not artifacts["evidence"]:
        failures.append({"artifact": "evidence", "reason": "missing_required_records"})
        total = 1
    else:
        total = len(artifacts["evidence"])
    return _check_record(
        "evidence_records_created",
        "Evidence records exist and refer to registered sources.",
        total,
        failures,
    )


def _chunks_link_evidence(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    evidence_ids = {row.get("evidence_id") for row in artifacts["evidence"]}
    text_evidence = [
        row
        for row in artifacts["evidence"]
        if row.get("evidence_type") in {"text_span", "utterance_span", "message_span", "ocr_text_span"}
    ]
    failures = []
    if text_evidence and not artifacts["chunks"]:
        return _check_record(
            "chunks_link_evidence",
            "Text-like evidence is represented by chunks linked to evidence IDs.",
            1,
            [{"artifact": "chunks", "reason": "missing_chunks_for_text_evidence"}],
        )
    for chunk in artifacts["chunks"]:
        chunk_evidence_ids = chunk.get("evidence_ids") or []
        if not isinstance(chunk_evidence_ids, list) or not chunk_evidence_ids:
            failures.append({"chunk_id": chunk.get("chunk_id"), "reason": "missing_evidence_ids"})
            continue
        missing = sorted(str(evidence_id) for evidence_id in chunk_evidence_ids if evidence_id not in evidence_ids)
        if missing:
            failures.append(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "missing_evidence_ids": missing,
                    "reason": "unknown_evidence_id",
                }
            )
    return _check_record(
        "chunks_link_evidence",
        "Text-like evidence is represented by chunks linked to evidence IDs.",
        len(artifacts["chunks"]),
        failures,
    )


def _spans_or_regions_detected(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    failures = []
    if not artifacts["spans"] and not artifacts["image_regions"]:
        failures.append({"artifacts": ["spans", "image_regions"], "reason": "missing_detected_units"})
    return _check_record(
        "spans_or_regions_detected",
        "Claim-bearing text spans or image regions have been detected.",
        1,
        failures,
    )


def _spans_link_evidence(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    evidence_ids = {row.get("evidence_id") for row in artifacts["evidence"]}
    chunk_ids = {row.get("chunk_id") for row in artifacts["chunks"]}
    failures = []
    for span in artifacts["spans"]:
        reasons = []
        if span.get("evidence_id") not in evidence_ids:
            reasons.append("unknown_evidence_id")
        chunk_id = span.get("chunk_id")
        if chunk_id is not None and chunk_id not in chunk_ids:
            reasons.append("unknown_chunk_id")
        if reasons:
            failures.append(
                {
                    "span_id": span.get("span_id"),
                    "evidence_id": span.get("evidence_id"),
                    "chunk_id": chunk_id,
                    "reasons": reasons,
                }
            )
    return _check_record(
        "spans_link_evidence",
        "Detected spans refer to known evidence and chunks.",
        len(artifacts["spans"]),
        failures,
    )


def _validation_run_completed(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    failures = []
    if not artifacts["validations"]:
        failures.append({"artifact": "validations", "reason": "missing_validation_records"})
    return _check_record(
        "validation_run_completed",
        "Validation produced validation-stage records.",
        1,
        failures,
    )


def _accepted_claims_complete_provenance(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    sources_by_id = _index_by(artifacts["sources"], "source_id")
    evidence_by_id = _index_by(artifacts["evidence"], "evidence_id")
    failures = []
    accepted = _accepted_claims(artifacts)
    for claim in accepted:
        reasons = []
        evidence = evidence_by_id.get(str(claim.get("evidence_id")))
        source = sources_by_id.get(str(claim.get("source_id")))
        if evidence is None:
            reasons.append("missing_evidence")
        if source is None:
            reasons.append("missing_source")
        if evidence is not None and evidence.get("source_id") != claim.get("source_id"):
            reasons.append("evidence_source_mismatch")
        if evidence is not None and evidence.get("source_modality") != claim.get("source_modality"):
            reasons.append("evidence_modality_mismatch")
        if evidence is not None and not isinstance(evidence.get("provenance"), dict):
            reasons.append("missing_evidence_provenance")
        elif evidence is not None and not evidence.get("provenance"):
            reasons.append("empty_evidence_provenance")
        if reasons:
            failures.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "evidence_id": claim.get("evidence_id"),
                    "source_id": claim.get("source_id"),
                    "reasons": reasons,
                }
            )
    return _check_record(
        "accepted_claims_complete_provenance",
        "Accepted claims link to source and evidence records with provenance.",
        len(accepted),
        failures,
    )


def _accepted_text_claims_exact_evidence(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    evidence_by_id = _index_by(artifacts["evidence"], "evidence_id")
    failures = []
    total = 0
    for claim in _accepted_claims(artifacts):
        evidence = evidence_by_id.get(str(claim.get("evidence_id")))
        if not _text_like_claim_requires_exact_match(claim, evidence):
            continue
        total += 1
        validation = claim.get("validation") or {}
        if not isinstance(validation, dict) or validation.get("evidence_exact_match") is not True:
            failures.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "evidence_id": claim.get("evidence_id"),
                    "reason": "evidence_exact_match_not_true",
                }
            )
    return _check_record(
        "accepted_text_claims_exact_evidence",
        "Accepted text-like claims have exact evidence substring validation.",
        total,
        failures,
    )


def _accepted_chat_audio_claims_attributed(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    raw_by_claim_id = _index_by(artifacts["claims_raw"], "claim_id")
    failures = []
    total = 0
    for claim in _accepted_claims(artifacts):
        if claim.get("source_modality") not in {"chat", "audio"}:
            continue
        total += 1
        raw = raw_by_claim_id.get(str(claim.get("claim_id")))
        attribution = raw.get("attribution") if raw else None
        reasons = []
        if raw is None:
            reasons.append("missing_raw_claim")
        if not isinstance(attribution, dict):
            reasons.append("missing_attribution")
        elif attribution.get("type") != "speaker":
            reasons.append("attribution_not_speaker")
        elif not _present(attribution.get("agent")):
            reasons.append("missing_attribution_agent")
        validation = claim.get("validation") or {}
        if isinstance(validation, dict) and validation.get("attribution_preserved") is False:
            reasons.append("attribution_not_preserved")
        if reasons:
            failures.append({"claim_id": claim.get("claim_id"), "reasons": reasons})
    return _check_record(
        "accepted_chat_audio_claims_attributed",
        "Accepted chat and audio claims keep non-empty speaker attribution.",
        total,
        failures,
    )


def _accepted_pdf_claims_page_provenance(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    evidence_by_id = _index_by(artifacts["evidence"], "evidence_id")
    failures = []
    total = 0
    for claim in _accepted_claims(artifacts):
        if claim.get("source_modality") != "pdf":
            continue
        total += 1
        evidence = evidence_by_id.get(str(claim.get("evidence_id")))
        provenance = evidence.get("provenance") if evidence else None
        if not isinstance(provenance, dict) or not (
            _present(provenance.get("page")) or _present(provenance.get("page_number"))
        ):
            failures.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "evidence_id": claim.get("evidence_id"),
                    "reason": "missing_page_provenance",
                }
            )
    return _check_record(
        "accepted_pdf_claims_page_provenance",
        "Accepted PDF claims preserve page provenance.",
        total,
        failures,
    )


def _accepted_claims_preserve_semantics(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    failures = []
    accepted = _accepted_claims(artifacts)
    for claim in accepted:
        validation = claim.get("validation") or {}
        if not isinstance(validation, dict):
            failures.append({"claim_id": claim.get("claim_id"), "reason": "missing_validation_summary"})
            continue
        failed_flags = [
            flag
            for flag in (
                "attribution_preserved",
                "negation_preserved",
                "uncertainty_preserved",
                "quantities_preserved",
            )
            if validation.get(flag) is False
        ]
        if failed_flags:
            failures.append({"claim_id": claim.get("claim_id"), "failed_flags": failed_flags})
    return _check_record(
        "accepted_claims_preserve_semantics",
        "Accepted claims preserve attribution, negation, uncertainty, and quantities when applicable.",
        len(accepted),
        failures,
    )


def _image_claim_truth_status_policy(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    failures = []
    image_claims = [claim for claim in artifacts["claims_raw"] if claim.get("source_modality") == "image"]
    for claim in image_claims:
        attribution = claim.get("attribution") or {}
        attribution_type = attribution.get("type") if isinstance(attribution, dict) else None
        reasons = []
        if claim.get("modality") not in IMAGE_SAFE_MODALITIES:
            reasons.append("unsafe_image_modality")
        if claim.get("truth_status") not in IMAGE_SAFE_TRUTH_STATUSES:
            reasons.append("unsafe_image_truth_status")
        if attribution_type not in IMAGE_SAFE_ATTRIBUTIONS:
            reasons.append("unsafe_image_attribution")
        if reasons:
            failures.append(
                {
                    "claim_id": claim.get("claim_id"),
                    "modality": claim.get("modality"),
                    "truth_status": claim.get("truth_status"),
                    "attribution_type": attribution_type,
                    "reasons": reasons,
                }
            )
    return _check_record(
        "image_claim_truth_status_policy",
        "Image claims remain model observations, hypotheses, or human-reviewed labels.",
        len(image_claims),
        failures,
    )


def _invalid_claims_quarantined_with_reasons(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    quarantines_by_claim_id: Dict[str, List[dict]] = {}
    for row in artifacts["quarantine"]:
        claim_id = row.get("claim_id")
        if isinstance(claim_id, str) and claim_id:
            quarantines_by_claim_id.setdefault(claim_id, []).append(row)

    invalid_validations = [row for row in artifacts["validations"] if row.get("status") == "quarantined"]
    failures = []
    for validation in invalid_validations:
        claim_id = validation.get("claim_id")
        matching_quarantines = quarantines_by_claim_id.get(str(claim_id), [])
        reasons = []
        if not validation.get("errors"):
            reasons.append("validation_missing_errors")
        if not matching_quarantines:
            reasons.append("missing_quarantine_record")
        elif not any(row.get("reason_codes") for row in matching_quarantines):
            reasons.append("quarantine_missing_reason_codes")
        if reasons:
            failures.append(
                {
                    "claim_id": claim_id,
                    "validation_id": validation.get("validation_id"),
                    "reasons": reasons,
                }
            )
    return _check_record(
        "invalid_claims_quarantined_with_reasons",
        "Quarantined validations have matching quarantine records and machine-readable reasons.",
        len(invalid_validations),
        failures,
    )


def _quarantine_reason_codes_present(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    failures = []
    for row in artifacts["quarantine"]:
        reason_codes = row.get("reason_codes")
        if (
            not isinstance(reason_codes, list)
            or not reason_codes
            or any(not isinstance(reason, str) or not reason for reason in reason_codes)
        ):
            failures.append(
                {
                    "quarantine_id": row.get("quarantine_id"),
                    "claim_id": row.get("claim_id"),
                    "reason": "missing_machine_readable_reason_code",
                }
            )
    return _check_record(
        "quarantine_reason_codes_present",
        "Every quarantined record has at least one machine-readable reason code.",
        len(artifacts["quarantine"]),
        failures,
    )


def _normalized_claims_from_accepted_claims(artifacts: Dict[str, List[dict]]) -> AcceptanceCheckRecord:
    accepted_claim_ids = {claim.get("claim_id") for claim in _accepted_claims(artifacts)}
    failures = []
    for normalized in artifacts["claims_normalized"]:
        claim_id = normalized.get("claim_id")
        if claim_id not in accepted_claim_ids:
            failures.append(
                {
                    "normalized_claim_id": normalized.get("normalized_claim_id"),
                    "claim_id": claim_id,
                    "reason": "normalized_from_non_accepted_claim",
                }
            )
    return _check_record(
        "normalized_claims_from_accepted_claims",
        "Normalized claims are derived only from accepted validated claims.",
        len(artifacts["claims_normalized"]),
        failures,
    )


def _summary_report_exists(config: PipelineConfig) -> AcceptanceCheckRecord:
    candidates = [
        config.paths.reports_dir / "extraction_summary.md",
        config.paths.reports_dir / "extraction_summary.html",
    ]
    if any(path.exists() and path.stat().st_size > 0 for path in candidates):
        failures: List[dict] = []
    else:
        failures = [
            {
                "paths": [str(path) for path in candidates],
                "reason": "missing_non_empty_summary_report",
            }
        ]
    return _check_record(
        "summary_report_exists",
        "A non-empty extraction summary report exists.",
        1,
        failures,
    )


def _load_artifacts(config: PipelineConfig) -> Dict[str, List[dict]]:
    paths = config.jsonl_paths()
    return {artifact_name: _optional_rows(path) for artifact_name, path in paths.items()}


def build_acceptance_checks(config: PipelineConfig) -> List[AcceptanceCheckRecord]:
    artifacts = _load_artifacts(config)
    checks = [
        _presence_check(
            "source_records_registered",
            "At least one source has been registered.",
            "sources",
            artifacts["sources"],
        ),
        _evidence_records_created(artifacts),
        _chunks_link_evidence(artifacts),
        _spans_or_regions_detected(artifacts),
        _spans_link_evidence(artifacts),
        _presence_check(
            "raw_claims_extracted",
            "At least one raw claim has been extracted.",
            "claims_raw",
            artifacts["claims_raw"],
        ),
        _validation_run_completed(artifacts),
        _invalid_claims_quarantined_with_reasons(artifacts),
        _accepted_claims_complete_provenance(artifacts),
        _accepted_text_claims_exact_evidence(artifacts),
        _accepted_chat_audio_claims_attributed(artifacts),
        _accepted_pdf_claims_page_provenance(artifacts),
        _accepted_claims_preserve_semantics(artifacts),
        _image_claim_truth_status_policy(artifacts),
        _normalized_claims_from_accepted_claims(artifacts),
        _quarantine_reason_codes_present(artifacts),
        _summary_report_exists(config),
        _artifact_ids_unique(artifacts),
    ]
    return checks


def write_acceptance_report(config: PipelineConfig, output_path: Optional[Path] = None) -> AcceptanceCheckResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "acceptance_check.jsonl"
    ensure_parent(output_path)
    records = build_acceptance_checks(config)
    write_jsonl(output_path, records)
    checks = [record.model_dump(mode="json") for record in records]
    return AcceptanceCheckResult(
        output_path=output_path,
        checks=checks,
        passed=all(check["status"] == "passed" for check in checks),
    )
