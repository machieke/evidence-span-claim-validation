from __future__ import annotations

import html
import json
import shlex
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import (
    append_jsonl,
    ensure_parent,
    existing_values,
    read_jsonl,
    write_jsonl,
)
from evidence_pipeline.schemas.audit import AuditEventRecord
from evidence_pipeline.schemas.base import utc_now
from evidence_pipeline.schemas.review import ReviewDecisionRecord, ReviewQueueRecord

REVIEW_DECISIONS = {"accept", "reject", "needs_review"}
REVIEW_QUEUE_FORMATS = {"jsonl", "html"}
REVIEW_QUEUE_VERSION = "review.queue.v1"

ANCHOR_KEYS_BY_MODALITY = {
    "chat": (
        "conversation_id",
        "message_id",
        "sender_id",
        "sender_role",
        "timestamp",
        "start_char",
        "end_char",
    ),
    "pdf": (
        "page",
        "page_number",
        "block_id",
        "bbox",
        "section_path",
        "start_char",
        "end_char",
    ),
    "audio": (
        "utterance_id",
        "speaker",
        "speaker_label",
        "start_seconds",
        "end_seconds",
        "start_ms",
        "end_ms",
        "asr_confidence",
        "diarization_confidence",
        "overlap",
    ),
    "image": (
        "image_id",
        "region_id",
        "bbox",
        "crop_path",
        "mask_path",
        "feature_cluster_id",
        "member_region_ids",
        "source_ids",
        "cluster_size",
        "cohesion_score",
    ),
}


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


def _normalized_by_claim_id(config: PipelineConfig) -> Dict[str, List[dict]]:
    normalized_claims: Dict[str, List[dict]] = {}
    for _, payload in read_jsonl(config.jsonl_paths()["claims_normalized"]):
        claim_id = payload.get("claim_id")
        if isinstance(claim_id, str):
            normalized_claims.setdefault(claim_id, []).append(payload)
    return normalized_claims


def _reviewable_without_validation(claim: dict) -> bool:
    return claim.get("support_status") == "needs_review"


def _is_reviewable_claim(claim: dict, validation: Optional[dict]) -> bool:
    if validation is not None and validation.get("status") in {"needs_review", "quarantined"}:
        return True
    return _reviewable_without_validation(claim)


def _review_command(claim_id: str, decision: str, reason_codes: List[str]) -> str:
    args = [
        "python3",
        "-m",
        "evidence_pipeline",
        "review-claim",
        claim_id,
        "--decision",
        decision,
        "--reviewer-id",
        "human_reviewer",
    ]
    for reason_code in reason_codes:
        args.extend(["--reason-code", reason_code])
    return " ".join(shlex.quote(str(arg)) for arg in args)


def _review_commands(claim_id: str, reason_codes: List[str]) -> Dict[str, str]:
    return {
        decision: _review_command(claim_id, decision, reason_codes)
        for decision in ("accept", "reject", "needs_review")
    }


def _add_anchor_value(anchor: Dict[str, Any], key: str, value: Any) -> None:
    if value is None or value == "" or value == [] or value == {}:
        return
    anchor[key] = value


def _evidence_anchor(claim: dict, evidence: Optional[dict], source: Optional[dict]) -> Dict[str, Any]:
    source_modality = claim.get("source_modality")
    provenance = evidence.get("provenance", {}) if evidence else {}
    if not isinstance(provenance, dict):
        provenance = {}

    anchor: Dict[str, Any] = {}
    for key, value in (
        ("source_modality", source_modality),
        ("source_id", claim.get("source_id")),
        ("source_file", source.get("source_file") if source else None),
        ("evidence_id", claim.get("evidence_id")),
        ("evidence_type", evidence.get("evidence_type") if evidence else None),
    ):
        _add_anchor_value(anchor, key, value)

    for key in ANCHOR_KEYS_BY_MODALITY.get(str(source_modality), ()):
        _add_anchor_value(anchor, key, provenance.get(key))

    return anchor


def _review_queue_item(
    claim: dict,
    validation: Optional[dict],
    evidence: Optional[dict],
    source: Optional[dict],
    normalized_claims: List[dict],
    latest_review: Optional[ReviewDecisionRecord],
) -> dict:
    evidence_risk_flags = evidence.get("risk_flags", []) if evidence else []
    claim_risk_flags = claim.get("risk_flags", [])
    validation_errors = validation.get("errors", []) if validation else []
    validation_warnings = validation.get("warnings", []) if validation else []
    claim_id = str(claim.get("claim_id"))
    record = ReviewQueueRecord(
        review_queue_id=stable_id(
            "reviewq",
            {
                "claim_id": claim.get("claim_id"),
                "validation_id": validation.get("validation_id") if validation else None,
                "review_id": latest_review.review_id if latest_review else None,
            },
        ),
        claim_id=claim_id,
        source_id=claim.get("source_id"),
        evidence_id=claim.get("evidence_id"),
        source_file=source.get("source_file") if source else None,
        source_modality=claim.get("source_modality"),
        claim_type=claim.get("claim_type"),
        source_faithful_claim=claim.get("source_faithful_claim"),
        evidence_text=claim.get("evidence_text") or (evidence or {}).get("text"),
        evidence={
            "evidence_type": evidence.get("evidence_type") if evidence else None,
            "provenance": evidence.get("provenance", {}) if evidence else {},
            "risk_flags": evidence_risk_flags,
        },
        evidence_anchor=_evidence_anchor(claim, evidence, source),
        normalized_claims=normalized_claims,
        validation_status=validation.get("status") if validation else "unvalidated",
        reason_codes=validation_errors,
        warnings=validation_warnings,
        risk_flags=sorted(set(claim_risk_flags) | set(evidence_risk_flags)),
        review_state=latest_review.decision if latest_review else "unreviewed",
        review_commands=_review_commands(claim_id, validation_errors),
        latest_review=latest_review.model_dump(mode="json") if latest_review else None,
    )
    return record.model_dump(mode="json", exclude_none=True)


def _default_review_queue_path(config: PipelineConfig, output_format: str) -> Path:
    suffix = "html" if output_format == "html" else "jsonl"
    return config.paths.reports_dir / f"review_queue.{suffix}"


def _format_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def _format_json_cell(value: object) -> str:
    if not value:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _review_controls(item: dict) -> str:
    decisions = ("accept", "reject", "needs_review")
    selected_decision = (
        item.get("review_state") if item.get("review_state") in decisions else "needs_review"
    )
    reason_codes = item.get("reason_codes") or []
    reason_code = reason_codes[0] if reason_codes else ""
    options = "".join(
        f'<option value="{decision}"{" selected" if decision == selected_decision else ""}>{decision}</option>'
        for decision in decisions
    )
    return "".join(
        [
            f'<form class="review-actions" data-claim-id="{html.escape(_format_cell(item.get("claim_id")))}">',
            f'<select name="decision" aria-label="Decision">{options}</select>',
            (
                '<input name="reason_code" aria-label="Reason code" '
                f'value="{html.escape(_format_cell(reason_code))}">'
            ),
            '<textarea name="notes" aria-label="Reviewer notes" rows="2"></textarea>',
            '<div class="action-buttons">',
            '<button type="button" data-decision="accept">Accept</button>',
            '<button type="button" data-decision="reject">Reject</button>',
            '<button type="button" data-decision="needs_review">Needs review</button>',
            "</div>",
            "</form>",
        ]
    )


def _review_command_block(item: dict) -> str:
    commands = item.get("review_commands") or {}
    lines = [
        f"{decision}: {command}"
        for decision, command in commands.items()
        if command
    ]
    if not lines:
        return ""
    return f'<pre class="review-commands">{html.escape(chr(10).join(lines))}</pre>'


def _anchor_value(value: object) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return _format_cell(value)


def _evidence_anchor_block(item: dict) -> str:
    anchor = item.get("evidence_anchor") or {}
    if not isinstance(anchor, dict) or not anchor:
        return ""
    rows = []
    for key, value in anchor.items():
        rows.append(
            "<dt>"
            f"{html.escape(str(key))}"
            "</dt>"
            "<dd>"
            f"{html.escape(_anchor_value(value))}"
            "</dd>"
        )
    return '<dl class="evidence-anchor">' + "".join(rows) + "</dl>"


def render_review_queue_html(queue_items: List[dict]) -> str:
    rows = []
    for item in queue_items:
        details = json.dumps(item, indent=2, sort_keys=True)
        normalized = _format_json_cell(item.get("normalized_claims"))
        controls = _review_controls(item)
        command_block = _review_command_block(item)
        anchor_block = _evidence_anchor_block(item)
        rows.append(
            "<tr>"
            f"<td>{html.escape(_format_cell(item.get('review_state')))}</td>"
            f"<td>{controls}</td>"
            f"<td>{command_block}</td>"
            f"<td>{html.escape(_format_cell(item.get('validation_status')))}</td>"
            f"<td>{html.escape(_format_cell(item.get('reason_codes')))}</td>"
            f"<td>{html.escape(_format_cell(item.get('warnings')))}</td>"
            f"<td>{html.escape(_format_cell(item.get('risk_flags')))}</td>"
            f"<td>{html.escape(_format_cell(item.get('source_file')))}</td>"
            f"<td>{anchor_block}</td>"
            f"<td>{html.escape(_format_cell(item.get('claim_id')))}</td>"
            f"<td>{html.escape(_format_cell(item.get('source_faithful_claim')))}</td>"
            f"<td>{html.escape(_format_cell(item.get('evidence_text')))}</td>"
            f"<td><pre>{html.escape(normalized)}</pre></td>"
            "<td><details><summary>JSON</summary><pre>"
            f"{html.escape(details)}"
            "</pre></details></td>"
            "</tr>"
        )
    body = [
        "<h1>Claim Review Queue</h1>",
        f"<p>Items: {len(queue_items)}</p>",
        "<table>",
        "<thead><tr>"
        "<th>Review</th>"
        "<th>Action</th>"
        "<th>Commands</th>"
        "<th>Validation</th>"
        "<th>Reasons</th>"
        "<th>Warnings</th>"
        "<th>Risk Flags</th>"
        "<th>Source</th>"
        "<th>Anchor</th>"
        "<th>Claim ID</th>"
        "<th>Claim</th>"
        "<th>Evidence</th>"
        "<th>Normalized</th>"
        "<th>Packet</th>"
        "</tr></thead>",
        "<tbody>",
        *rows,
        "</tbody></table>",
    ]
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Claim Review Queue</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;line-height:1.45;margin:2rem;max-width:1400px;}",
            "table{border-collapse:collapse;margin:1rem 0;width:100%;}",
            "th,td{border:1px solid #d0d7de;padding:0.45rem 0.6rem;text-align:left;vertical-align:top;}",
            "th{background:#f6f8fa;}",
            "pre{white-space:pre-wrap;max-width:44rem;}",
            ".review-actions{display:grid;gap:0.35rem;min-width:12rem;}",
            ".review-actions select,.review-actions input,.review-actions textarea{font:inherit;max-width:100%;}",
            ".action-buttons{display:flex;gap:0.35rem;flex-wrap:wrap;}",
            ".action-buttons button{font:inherit;padding:0.25rem 0.45rem;}",
            ".review-commands{max-width:32rem;}",
            ".evidence-anchor{display:grid;grid-template-columns:max-content minmax(8rem,1fr);gap:0.15rem 0.5rem;margin:0;}",
            ".evidence-anchor dt{font-weight:700;}",
            ".evidence-anchor dd{margin:0;word-break:break-word;}",
            "</style>",
            "</head>",
            "<body>",
            *body,
            "</body>",
            "</html>",
            "",
        ]
    )


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
    output_format: str = "jsonl",
) -> ReviewQueueResult:
    normalized_format = output_format.strip().lower()
    if normalized_format not in REVIEW_QUEUE_FORMATS:
        raise ValueError("review queue format must be jsonl or html")
    if output_path is None:
        output_path = _default_review_queue_path(config, normalized_format)
    paths = config.jsonl_paths()
    validations = _validation_by_claim_id(config)
    latest_reviews = _latest_reviews(config)
    evidence_by_id = _payloads_by_id(config, "evidence", "evidence_id")
    source_by_id = _payloads_by_id(config, "sources", "source_id")
    normalized_by_claim_id = _normalized_by_claim_id(config)

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
        normalized_claims = normalized_by_claim_id.get(claim_id, [])
        queue_items.append(
            _review_queue_item(claim, validation, evidence, source, normalized_claims, latest_review)
        )

    if normalized_format == "html":
        ensure_parent(output_path)
        output_path.write_text(render_review_queue_html(queue_items), encoding="utf-8")
    else:
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
