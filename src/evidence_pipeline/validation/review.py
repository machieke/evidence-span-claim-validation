from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl
from evidence_pipeline.schemas.review import ReviewDecisionRecord

REVIEW_DECISIONS = {"accept", "reject", "needs_review"}


@dataclass
class ClaimReviewResult:
    review_id: str
    created: bool


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
    return ClaimReviewResult(review_id=review_id, created=True)
