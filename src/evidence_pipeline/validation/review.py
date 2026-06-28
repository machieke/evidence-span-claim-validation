from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl, write_jsonl
from evidence_pipeline.schemas.audit import AuditEventRecord
from evidence_pipeline.schemas.base import utc_now
from evidence_pipeline.schemas.review import ReviewDecisionRecord

REVIEW_DECISIONS = {"accept", "reject", "needs_review"}


@dataclass
class ClaimReviewResult:
    review_id: str
    created: bool


@dataclass
class ReviewQueueResult:
    output_path: Path
    item_count: int


def _normalize_decision(decision: str) -> str:
    normalized = decision.strip().lower().replace("-", "_")
    if normalized not in REVIEW_DECISIONS:
        expected = ", ".join(sorted(REVIEW_DECISIONS))
        raise ValueError(f"review decision must be one of: {expected}")
    return normalized


def _claim_payload(config: PipelineConfig, claim_id: str) -> Optional[dict]:
    paths = config.jsonl_paths()
    for key in ("claims_raw", "claims_validated"):
        for _, payload in read_jsonl(paths[key]):
            if payload.get("claim_id") == claim_id:
                return payload
    return None


def _latest_reviews(config: PipelineConfig) -> Dict[str, ReviewDecisionRecord]:
    latest_by_claim_id: Dict[str, ReviewDecisionRecord] = {}
    for _, review_payload in read_jsonl(config.jsonl_paths()["review_decisions"]):
        review = ReviewDecisionRecord.model_validate(review_payload)
        latest = latest_by_claim_id.get(review.claim_id)
        if latest is None or review.reviewed_at >= latest.reviewed_at:
            latest_by_claim_id[review.claim_id] = review
    return latest_by_claim_id


def _validation_by_claim_id(config: PipelineConfig) -> Dict[str, dict]:
    validations = {}
    for _, validation in read_jsonl(config.jsonl_paths()["validations"]):
        claim_id = validation.get("claim_id")
        if isinstance(claim_id, str):
            validations[claim_id] = validation
    return validations


def _payloads_by_id(config: PipelineConfig, artifact: str, id_field: str) -> Dict[str, dict]:
    return {
        str(payload[id_field]): payload
        for _, payload in read_jsonl(config.jsonl_paths()[artifact])
        if id_field in payload
    }


def _reviewable_without_validation(claim: dict) -> bool:
    return claim.get("support_status") == "needs_review"


def _is_reviewable_claim(claim: dict, validation: Optional[dict]) -> bool:
    if validation is not None and validation.get("status") in {"needs_review", "quarantined"}:
        return True
    return _reviewable_without_validation(claim)


def _review_queue_item(
    claim: dict,
    validation: Optional[dict],
    evidence: Optional[dict],
    source: Optional[dict],
    latest_review: Optional[ReviewDecisionRecord],
) -> dict:
    evidence_risk_flags = evidence.get("risk_flags", []) if evidence else []
    claim_risk_flags = claim.get("risk_flags", [])
    validation_errors = validation.get("errors", []) if validation else []
    validation_warnings = validation.get("warnings", []) if validation else []
    return {
        "review_queue_id": stable_id(
            "reviewq",
            {
                "claim_id": claim.get("claim_id"),
                "validation_id": validation.get("validation_id") if validation else None,
                "review_id": latest_review.review_id if latest_review else None,
            },
        ),
        "claim_id": claim.get("claim_id"),
        "source_id": claim.get("source_id"),
        "evidence_id": claim.get("evidence_id"),
        "source_file": source.get("source_file") if source else None,
        "source_modality": claim.get("source_modality"),
        "claim_type": claim.get("claim_type"),
        "source_faithful_claim": claim.get("source_faithful_claim"),
        "evidence_text": claim.get("evidence_text") or (evidence or {}).get("text"),
        "evidence": {
            "evidence_type": evidence.get("evidence_type") if evidence else None,
            "provenance": evidence.get("provenance", {}) if evidence else {},
            "risk_flags": evidence_risk_flags,
        },
        "validation_status": validation.get("status") if validation else "unvalidated",
        "reason_codes": validation_errors,
        "warnings": validation_warnings,
        "risk_flags": sorted(set(claim_risk_flags) | set(evidence_risk_flags)),
        "review_state": latest_review.decision if latest_review else "unreviewed",
        "latest_review": latest_review.model_dump(mode="json") if latest_review else None,
        "schema_version": "review.queue.v1",
    }


def _review_id(
    claim_id: str,
    reviewer_id: str,
    decision: str,
    reason_codes: List[str],
    notes: Optional[str],
) -> str:
    return stable_id(
        "review",
        {
            "claim_id": claim_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "reason_codes": reason_codes,
            "notes": notes or "",
        },
    )


def write_review_queue(
    config: PipelineConfig,
    output_path: Optional[Path] = None,
    include_reviewed: bool = False,
) -> ReviewQueueResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "review_queue.jsonl"
    paths = config.jsonl_paths()
    validations = _validation_by_claim_id(config)
    latest_reviews = _latest_reviews(config)
    evidence_by_id = _payloads_by_id(config, "evidence", "evidence_id")
    source_by_id = _payloads_by_id(config, "sources", "source_id")

    queue_items = []
    for _, claim in read_jsonl(paths["claims_raw"]):
        claim_id = claim.get("claim_id")
        if not isinstance(claim_id, str):
            continue
        validation = validations.get(claim_id)
        latest_review = latest_reviews.get(claim_id)
        if not _is_reviewable_claim(claim, validation):
            continue
        if (
            latest_review is not None
            and latest_review.decision in {"accept", "reject"}
            and not include_reviewed
        ):
            continue
        evidence = evidence_by_id.get(str(claim.get("evidence_id")))
        source = source_by_id.get(str(claim.get("source_id")))
        queue_items.append(_review_queue_item(claim, validation, evidence, source, latest_review))

    write_jsonl(output_path, queue_items)
    return ReviewQueueResult(output_path=output_path, item_count=len(queue_items))


def _audit_event_id(
    action: str,
    reviewer_id: str,
    claim_id: str,
    review_id: str,
    status: str,
    created_at: datetime,
) -> str:
    return stable_id(
        "audit",
        {
            "action": action,
            "reviewer_id": reviewer_id,
            "claim_id": claim_id,
            "review_id": review_id,
            "status": status,
            "created_at": created_at.isoformat(),
        },
    )


def _append_review_audit(
    config: PipelineConfig,
    claim: dict,
    review_id: str,
    reviewer_id: str,
    decision: str,
    reason_codes: List[str],
    status: str,
    skip_reason: Optional[str] = None,
) -> None:
    created_at = utc_now()
    details = {
        "decision": decision,
        "reason_codes": reason_codes,
        "review_id": review_id,
    }
    if skip_reason:
        details["skip_reason"] = skip_reason
    append_jsonl(
        config.jsonl_paths()["audit_events"],
        AuditEventRecord(
            audit_event_id=_audit_event_id(
                "review_claim",
                reviewer_id,
                str(claim["claim_id"]),
                review_id,
                status,
                created_at,
            ),
            action="review_claim",
            actor_id=reviewer_id,
            target_type="claim",
            target_id=str(claim["claim_id"]),
            source_id=claim.get("source_id"),
            evidence_id=claim.get("evidence_id"),
            claim_id=str(claim["claim_id"]),
            status=status,
            details=details,
            created_at=created_at,
        ),
    )


def record_claim_review(
    config: PipelineConfig,
    claim_id: str,
    decision: str,
    reviewer_id: str,
    reason_codes: Optional[List[str]] = None,
    notes: Optional[str] = None,
) -> ClaimReviewResult:
    normalized_decision = _normalize_decision(decision)
    normalized_reason_codes = sorted(set(reason_codes or []))
    claim = _claim_payload(config, claim_id)
    if claim is None:
        raise ValueError(f"claim_id not found: {claim_id}")

    paths = config.jsonl_paths()
    review_id = _review_id(claim_id, reviewer_id, normalized_decision, normalized_reason_codes, notes)
    if review_id in existing_values(paths["review_decisions"], "review_id"):
        _append_review_audit(
            config,
            claim,
            review_id,
            reviewer_id,
            normalized_decision,
            normalized_reason_codes,
            status="skipped",
            skip_reason="duplicate_review",
        )
        return ClaimReviewResult(review_id=review_id, created=False)

    append_jsonl(
        paths["review_decisions"],
        ReviewDecisionRecord(
            review_id=review_id,
            claim_id=claim_id,
            source_id=claim.get("source_id"),
            evidence_id=claim.get("evidence_id"),
            reviewer_id=reviewer_id,
            decision=normalized_decision,
            reason_codes=normalized_reason_codes,
            notes=notes,
            metadata={"claim_type": claim.get("claim_type"), "source_modality": claim.get("source_modality")},
        ),
    )
    _append_review_audit(
        config,
        claim,
        review_id,
        reviewer_id,
        normalized_decision,
        normalized_reason_codes,
        status="created",
    )
    return ClaimReviewResult(review_id=review_id, created=True)
