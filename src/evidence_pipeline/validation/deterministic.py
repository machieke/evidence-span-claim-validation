from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.claims import ClaimValidationSummary, RawClaimRecord, ValidatedClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.schemas.validation import QuarantineRecord, ValidationRecord
from evidence_pipeline.validation.text_support import (
    NEGATION_MARKERS,
    UNCERTAINTY_MARKERS,
    contains_negation,
    contains_uncertainty,
    evidence_substring_match,
    extract_quantities,
    unsupported_entities,
)

VALIDATOR_VERSION = "deterministic.v1"


@dataclass
class ClaimValidationRunResult:
    accepted: int
    quarantined: int
    skipped: int


@dataclass
class _ClaimValidationDecision:
    status: str
    errors: List[str]
    warnings: List[str]
    summary: ClaimValidationSummary


def _support_texts(claim: RawClaimRecord, evidence: Optional[EvidenceRecord], span: Optional[SpanRecord]) -> List[str]:
    texts = []
    if span is not None and span.text:
        texts.append(span.text)
    if evidence is not None and evidence.text:
        texts.append(evidence.text)
    if claim.context_used:
        texts.append(claim.context_used)
    return texts


def _claim_text_for_checks(claim: RawClaimRecord) -> str:
    parts = [claim.source_faithful_claim, claim.modality]
    if claim.subject:
        parts.append(str(claim.subject))
    if claim.predicate:
        parts.append(str(claim.predicate))
    if claim.object is not None:
        parts.append(str(claim.object))
    if claim.attributes:
        parts.append(str(claim.attributes))
    return " ".join(parts)


def _validate_attribution(claim: RawClaimRecord, evidence: Optional[EvidenceRecord]) -> List[str]:
    if claim.source_modality in {"chat", "audio"}:
        if claim.attribution.type != "speaker":
            return ["attribution_not_speaker"]
        if evidence is None:
            return []
        provenance = evidence.provenance
        allowed_agents = {
            str(value)
            for value in (
                provenance.get("sender_id"),
                provenance.get("sender_display_name"),
                provenance.get("speaker"),
            )
            if value is not None
        }
        if allowed_agents and claim.attribution.agent not in allowed_agents:
            return ["attribution_agent_mismatch"]
    if claim.source_modality == "pdf" and claim.attribution.type not in {"document", "speaker"}:
        return ["attribution_not_document"]
    if claim.source_modality == "image" and claim.attribution.type not in {"model", "human_reviewer"}:
        return ["attribution_not_model_or_reviewer"]
    return []


def _quantities_preserved(claim: RawClaimRecord, support_text: str) -> bool:
    evidence_quantities = extract_quantities(claim.evidence_text or support_text)
    claim_quantities = extract_quantities(_claim_text_for_checks(claim))
    if not evidence_quantities:
        return True
    return bool(claim_quantities) and claim_quantities.issubset(evidence_quantities)


def validate_claim_deterministically(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
    span: Optional[SpanRecord],
) -> _ClaimValidationDecision:
    errors: List[str] = []
    warnings: List[str] = []

    if evidence is None:
        errors.append("missing_evidence")
    elif claim.source_id != evidence.source_id:
        errors.append("source_id_mismatch")

    if claim.span_id and span is None:
        errors.append("missing_span")
    elif span is not None and span.evidence_id != claim.evidence_id:
        errors.append("span_evidence_mismatch")

    support_texts = _support_texts(claim, evidence, span)
    support_text = support_texts[0] if support_texts else ""
    evidence_exact_match = None
    if claim.source_modality != "image":
        if not claim.evidence_text:
            errors.append("missing_evidence_text")
            evidence_exact_match = False
        else:
            match = evidence_substring_match(claim.evidence_text, support_texts)
            evidence_exact_match = match.exact
            if not match.exact:
                errors.append("evidence_not_exact_substring")
                if match.normalized:
                    warnings.append("evidence_matches_after_normalization")

    attribution_errors = _validate_attribution(claim, evidence)
    errors.extend(attribution_errors)
    attribution_preserved = not attribution_errors

    claim_text = _claim_text_for_checks(claim)
    support_for_semantics = claim.evidence_text or support_text

    negation_preserved = True
    if contains_negation(support_for_semantics):
        negation_preserved = contains_negation(claim_text) or claim.modality == "negated"
        if not negation_preserved:
            errors.append("negation_dropped")

    uncertainty_preserved = True
    if contains_uncertainty(support_for_semantics):
        uncertainty_preserved = contains_uncertainty(claim_text) or claim.modality in {
            "uncertain_observation",
            "reported_uncertain",
            "hypothetical",
        }
        if not uncertainty_preserved:
            errors.append("uncertainty_dropped")

    quantities_preserved = _quantities_preserved(claim, support_for_semantics)
    if not quantities_preserved:
        errors.append("quantity_mismatch")

    introduced_entities = unsupported_entities(
        claim_text,
        support_texts=support_texts,
        allowed_texts=[claim.attribution.agent or ""],
    )
    if introduced_entities:
        warnings.append("unsupported_entities_introduced")

    summary = ClaimValidationSummary(
        deterministic_valid=not errors,
        evidence_exact_match=evidence_exact_match,
        negation_preserved=negation_preserved,
        uncertainty_preserved=uncertainty_preserved,
        attribution_preserved=attribution_preserved,
        quantities_preserved=quantities_preserved,
        introduced_entities=introduced_entities,
        validator_version=VALIDATOR_VERSION,
    )
    return _ClaimValidationDecision(
        status="accepted_extracted" if not errors else "quarantined",
        errors=errors,
        warnings=warnings,
        summary=summary,
    )


def _normalized_claim_from_raw(claim: RawClaimRecord) -> Dict[str, object]:
    return {
        "subject": claim.subject,
        "predicate": claim.predicate,
        "object": claim.object,
        "quantity": claim.quantity,
        "attributes": claim.attributes,
    }


def _combined_risk_flags(claim: RawClaimRecord, evidence: Optional[EvidenceRecord], span: Optional[SpanRecord]) -> List[str]:
    flags = set(claim.risk_flags)
    if evidence is not None:
        flags.update(evidence.risk_flags)
    if span is not None:
        flags.update(span.risk_flags)
    return sorted(flags)


def _validation_id(claim_id: str) -> str:
    return stable_id("val", {"claim_id": claim_id, "validator": VALIDATOR_VERSION})


def _quarantine_id(claim_id: str) -> str:
    return stable_id("q", {"claim_id": claim_id, "validator": VALIDATOR_VERSION})


def validate_raw_claims(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    claim_ids: Optional[Sequence[str]] = None,
) -> ClaimValidationRunResult:
    paths = config.jsonl_paths()
    requested_claim_ids = set(claim_ids or [])
    evidence_by_id = {record.evidence_id: record for _, record in read_jsonl_records(paths["evidence"], EvidenceRecord)}
    spans_by_id = {record.span_id: record for _, record in read_jsonl_records(paths["spans"], SpanRecord)}
    existing_validation_ids = existing_values(paths["validations"], "validation_id")
    accepted = 0
    quarantined = 0
    skipped = 0

    for _, claim in read_jsonl_records(paths["claims_raw"], RawClaimRecord):
        if source_id is not None and claim.source_id != source_id:
            continue
        if requested_claim_ids and claim.claim_id not in requested_claim_ids:
            continue
        validation_id = _validation_id(claim.claim_id)
        if validation_id in existing_validation_ids:
            skipped += 1
            continue

        evidence = evidence_by_id.get(claim.evidence_id)
        span = spans_by_id.get(claim.span_id) if claim.span_id else None
        decision = validate_claim_deterministically(claim, evidence, span)

        append_jsonl(
            paths["validations"],
            ValidationRecord(
                validation_id=validation_id,
                claim_id=claim.claim_id,
                record_id=claim.claim_id,
                stage="validate_claims",
                status=decision.status,
                errors=decision.errors,
                warnings=decision.warnings,
                validator_version=VALIDATOR_VERSION,
                metadata={"source_modality": claim.source_modality},
            ),
        )
        existing_validation_ids.add(validation_id)

        if decision.status == "accepted_extracted":
            append_jsonl(
                paths["claims_validated"],
                ValidatedClaimRecord(
                    claim_id=claim.claim_id,
                    source_id=claim.source_id,
                    source_modality=claim.source_modality,
                    span_id=claim.span_id,
                    evidence_id=claim.evidence_id,
                    source_faithful_claim=claim.source_faithful_claim,
                    evidence_text=claim.evidence_text,
                    normalized_claim=_normalized_claim_from_raw(claim),
                    modality=claim.modality,
                    truth_status=claim.truth_status,
                    support_status="accepted_extracted",
                    validation=decision.summary,
                    risk_flags=_combined_risk_flags(claim, evidence, span),
                ),
            )
            accepted += 1
        else:
            append_jsonl(
                paths["quarantine"],
                QuarantineRecord(
                    quarantine_id=_quarantine_id(claim.claim_id),
                    record_type="claim",
                    record_id=claim.claim_id,
                    source_id=claim.source_id,
                    evidence_id=claim.evidence_id,
                    claim_id=claim.claim_id,
                    stage="validate_claims",
                    reason_codes=decision.errors,
                    warnings=decision.warnings,
                    payload=claim.model_dump(mode="json", exclude_none=True),
                ),
            )
            quarantined += 1

    return ClaimValidationRunResult(accepted=accepted, quarantined=quarantined, skipped=skipped)
