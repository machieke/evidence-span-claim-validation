from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.claims import ClaimValidationSummary, RawClaimRecord, ValidatedClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.review import ReviewDecisionRecord
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

VALIDATOR_VERSION = "deterministic.v9"
IMAGE_CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.85
IMAGE_CLUSTER_MIN_COHESION = 0.75
IMAGE_CLUSTER_MIN_SIZE = 5
IMAGE_CLUSTER_MIN_SOURCE_COUNT = 3
ENTITY_ALLOWED_PROVENANCE_KEYS = (
    "sender_id",
    "sender_display_name",
    "speaker",
    "speaker_label",
    "image_id",
    "region_id",
    "feature_cluster_id",
    "embedding_model",
    "clustering_method",
    "ocr_model",
)


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


def _valid_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    return all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)


def _pdf_provenance_errors(claim: RawClaimRecord, evidence: Optional[EvidenceRecord]) -> List[str]:
    if claim.source_modality != "pdf" or evidence is None:
        return []

    provenance = evidence.provenance
    errors: List[str] = []
    page = provenance.get("page") or provenance.get("page_number")
    if not isinstance(page, int) or isinstance(page, bool) or page < 1:
        errors.append("missing_page_provenance")
    if not provenance.get("block_id"):
        errors.append("missing_block_provenance")

    extractor = str(provenance.get("extractor") or "").lower()
    if extractor == "pymupdf" and not _valid_bbox(provenance.get("bbox")):
        errors.append("missing_bbox_provenance")

    return errors


def _has_text_provenance(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _chat_provenance_errors(claim: RawClaimRecord, evidence: Optional[EvidenceRecord]) -> List[str]:
    if claim.source_modality != "chat" or evidence is None:
        return []

    provenance = evidence.provenance
    errors: List[str] = []
    if not _has_text_provenance(provenance.get("conversation_id")):
        errors.append("missing_conversation_provenance")
    if not _has_text_provenance(provenance.get("message_id")):
        errors.append("missing_message_provenance")
    if not _has_text_provenance(provenance.get("sender_id")):
        errors.append("missing_sender_provenance")
    return errors


def _numeric_provenance(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _confidence_provenance_finding(provenance: Dict[str, object], key: str) -> Optional[str]:
    if key not in provenance:
        return f"missing_{key}_provenance"
    value = _numeric_provenance(provenance.get(key))
    if value is None or not 0 <= value <= 1:
        return f"invalid_{key}_provenance"
    return None


def _audio_provenance_findings(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
) -> Tuple[List[str], List[str]]:
    if claim.source_modality != "audio" or evidence is None:
        return [], []

    provenance = evidence.provenance
    errors: List[str] = []
    warnings: List[str] = []
    if not _has_text_provenance(provenance.get("utterance_id")):
        errors.append("missing_utterance_provenance")
    if not _has_text_provenance(provenance.get("speaker")):
        errors.append("missing_speaker_provenance")

    start = _numeric_provenance(provenance.get("start", provenance.get("start_seconds")))
    end = _numeric_provenance(provenance.get("end", provenance.get("end_seconds")))
    if start is None or end is None:
        errors.append("missing_audio_timestamp_provenance")
    elif start < 0 or end < start:
        errors.append("invalid_audio_timestamp_bounds")
    duration = _numeric_provenance(provenance.get("source_duration", provenance.get("duration_seconds")))
    if duration is None:
        warnings.append("missing_audio_source_duration_provenance")
    elif duration < 0:
        errors.append("invalid_audio_source_duration_provenance")
    elif start is not None and end is not None and end > duration:
        errors.append("audio_timestamp_out_of_source_bounds")

    for key in ("asr_confidence", "diarization_confidence"):
        finding = _confidence_provenance_finding(provenance, key)
        if finding is None:
            continue
        if finding.startswith("missing_"):
            warnings.append(finding)
        else:
            errors.append(finding)

    return errors, warnings


def _non_empty_list(value: object) -> bool:
    return isinstance(value, list) and bool(value)


def _valid_xywh_bbox(value: object) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    values = [_numeric_provenance(item) for item in value]
    if any(item is None for item in values):
        return False
    x, y, width, height = values
    return x >= 0 and y >= 0 and width > 0 and height > 0


def _image_provenance_findings(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
) -> Tuple[List[str], List[str]]:
    if claim.source_modality != "image" or evidence is None:
        return [], []

    provenance = evidence.provenance
    errors: List[str] = []
    warnings: List[str] = []

    if evidence.evidence_type == "visual_region":
        if not _has_text_provenance(provenance.get("image_id")):
            errors.append("missing_image_provenance")
        if not _has_text_provenance(provenance.get("region_id")):
            errors.append("missing_region_provenance")
        if not _valid_xywh_bbox(provenance.get("bbox")):
            errors.append("missing_image_bbox_provenance")
        if not _has_text_provenance(provenance.get("proposal_method")):
            errors.append("missing_region_proposal_provenance")
        if not _has_text_provenance(provenance.get("crop_path")):
            warnings.append("missing_region_crop_provenance")
    elif evidence.evidence_type == "visual_cluster":
        if not _has_text_provenance(provenance.get("feature_cluster_id")):
            errors.append("missing_feature_cluster_provenance")
        if not _has_text_provenance(provenance.get("embedding_model")):
            errors.append("missing_embedding_model_provenance")
        if not _has_text_provenance(provenance.get("clustering_method")):
            errors.append("missing_clustering_method_provenance")
        if not _non_empty_list(provenance.get("member_region_ids")):
            errors.append("missing_cluster_member_provenance")
        if not _non_empty_list(provenance.get("representative_region_ids")):
            warnings.append("missing_cluster_representative_provenance")
    elif evidence.evidence_type == "ocr_text_span":
        if not _has_text_provenance(provenance.get("image_id")):
            errors.append("missing_image_provenance")
        if not _valid_xywh_bbox(provenance.get("bbox")):
            errors.append("missing_image_bbox_provenance")
        if not _has_text_provenance(provenance.get("ocr_model")):
            errors.append("missing_ocr_model_provenance")
        finding = _confidence_provenance_finding(provenance, "ocr_confidence")
        if finding is not None:
            if finding.startswith("missing_"):
                warnings.append(finding)
            else:
                errors.append(finding)

    return errors, warnings


def _context_dependency_errors(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
    span: Optional[SpanRecord],
) -> List[str]:
    if claim.source_modality not in {"chat", "audio"}:
        return []
    risk_flags = set(_combined_risk_flags(claim, evidence, span))
    if "context_dependent_coreference" not in risk_flags:
        return []
    errors: List[str] = []
    if not claim.context_dependent:
        errors.append("context_dependency_not_flagged")
    if not claim.context_used:
        errors.append("missing_context_used")
    return errors


def _quantities_preserved(claim: RawClaimRecord, support_text: str) -> bool:
    evidence_quantities = extract_quantities(claim.evidence_text or support_text)
    claim_quantities = extract_quantities(_claim_text_for_checks(claim))
    if not evidence_quantities:
        return True
    return bool(claim_quantities) and claim_quantities.issubset(evidence_quantities)


def _allowed_entity_texts(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
) -> List[str]:
    allowed = [claim.attribution.agent or ""]
    if evidence is None:
        return allowed

    for key in ENTITY_ALLOWED_PROVENANCE_KEYS:
        value = evidence.provenance.get(key)
        if isinstance(value, str):
            allowed.append(value)
    return allowed


def validate_claim_deterministically(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
    span: Optional[SpanRecord],
    review_decisions: Optional[Sequence[ReviewDecisionRecord]] = None,
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
    requires_exact_evidence = claim.source_modality != "image" or (
        evidence is not None and evidence.evidence_type == "ocr_text_span"
    )
    if requires_exact_evidence:
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
    errors.extend(_pdf_provenance_errors(claim, evidence))
    errors.extend(_chat_provenance_errors(claim, evidence))
    audio_provenance_errors, audio_provenance_warnings = _audio_provenance_findings(claim, evidence)
    errors.extend(audio_provenance_errors)
    warnings.extend(audio_provenance_warnings)
    image_provenance_errors, image_provenance_warnings = _image_provenance_findings(claim, evidence)
    errors.extend(image_provenance_errors)
    warnings.extend(image_provenance_warnings)
    errors.extend(_context_dependency_errors(claim, evidence, span))
    errors.extend(_audio_risk_errors(claim, evidence, span))
    errors.extend(_image_risk_errors(claim, evidence, review_decisions or []))
    errors.extend(_ocr_risk_errors(claim, evidence, span))

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
        allowed_texts=_allowed_entity_texts(claim, evidence),
    )
    if introduced_entities:
        warnings.append("unsupported_entities_introduced")
        if requires_exact_evidence:
            errors.append("unsupported_entities_introduced")

    summary = ClaimValidationSummary(
        deterministic_valid=not errors,
        evidence_exact_match=evidence_exact_match,
        negation_preserved=negation_preserved,
        uncertainty_preserved=uncertainty_preserved,
        attribution_preserved=attribution_preserved,
        quantities_preserved=quantities_preserved,
        introduced_entities=introduced_entities,
        claim_confidence=claim.confidence,
        confidence_basis="raw_claim_confidence",
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


def _audio_risk_errors(claim: RawClaimRecord, evidence: Optional[EvidenceRecord], span: Optional[SpanRecord]) -> List[str]:
    if claim.source_modality != "audio":
        return []
    risk_flags = set(_combined_risk_flags(claim, evidence, span))
    errors = []
    for reason_code in ("low_asr_confidence", "speaker_uncertain", "overlapping_speech"):
        if reason_code in risk_flags:
            errors.append(reason_code)
    return errors


def _ocr_risk_errors(claim: RawClaimRecord, evidence: Optional[EvidenceRecord], span: Optional[SpanRecord]) -> List[str]:
    if evidence is None or evidence.evidence_type != "ocr_text_span":
        return []
    risk_flags = set(_combined_risk_flags(claim, evidence, span))
    if "low_ocr_confidence" in risk_flags:
        return ["low_ocr_confidence"]
    return []


def _classifier_confidence(claim: RawClaimRecord) -> Optional[float]:
    classifier = claim.attributes.get("classifier")
    if isinstance(classifier, dict):
        confidence = classifier.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            return float(confidence)
    return claim.confidence


def _numeric_value(value: object) -> Optional[float]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _visual_cluster_size(claim: RawClaimRecord, evidence: Optional[EvidenceRecord]) -> Optional[int]:
    if evidence is not None:
        cluster_size = evidence.provenance.get("cluster_size")
        if isinstance(cluster_size, int) and not isinstance(cluster_size, bool):
            return cluster_size
    if isinstance(claim.object, list):
        return len(claim.object)
    return None


def _visual_cluster_source_count(evidence: Optional[EvidenceRecord]) -> Optional[int]:
    if evidence is None:
        return None
    source_ids = evidence.provenance.get("source_ids")
    if isinstance(source_ids, list):
        return len({str(source_id) for source_id in source_ids})
    return None


def _visual_cluster_cohesion(claim: RawClaimRecord, evidence: Optional[EvidenceRecord]) -> Optional[float]:
    if evidence is not None:
        cohesion_score = _numeric_value(evidence.provenance.get("cohesion_score"))
        if cohesion_score is not None:
            return cohesion_score
    return _numeric_value(claim.attributes.get("cohesion_score"))


def _latest_review_decision(review_decisions: Sequence[ReviewDecisionRecord]) -> Optional[ReviewDecisionRecord]:
    if not review_decisions:
        return None
    latest = review_decisions[0]
    for review in review_decisions[1:]:
        if review.reviewed_at >= latest.reviewed_at:
            latest = review
    return latest


def _human_confirmed_visual_label(
    claim: RawClaimRecord,
    latest_review: Optional[ReviewDecisionRecord],
) -> bool:
    if latest_review is not None and latest_review.decision == "accept":
        return True
    if claim.truth_status == "human_confirmed" or claim.attribution.type == "human_reviewer":
        return True
    return bool(claim.attributes.get("human_confirmed") or claim.attributes.get("human_reviewed"))


def _image_risk_errors(
    claim: RawClaimRecord,
    evidence: Optional[EvidenceRecord],
    review_decisions: Sequence[ReviewDecisionRecord],
) -> List[str]:
    if claim.source_modality != "image":
        return []
    if claim.claim_type == "unnamed_visual_feature_cluster":
        return _image_cluster_errors(claim, evidence)
    if claim.claim_type != "named_visual_classification":
        return []
    latest_review = _latest_review_decision(review_decisions)
    if latest_review is not None and latest_review.decision == "reject":
        return ["human_review_rejected_label"]
    if latest_review is not None and latest_review.decision == "needs_review":
        return ["human_review_needs_review"]
    if _human_confirmed_visual_label(claim, latest_review):
        return []
    confidence = _classifier_confidence(claim)
    if confidence is None:
        return ["image_label_missing_confidence"]
    if confidence < IMAGE_CLASSIFICATION_CONFIDENCE_THRESHOLD:
        return ["image_label_low_confidence"]
    return []


def _image_cluster_errors(claim: RawClaimRecord, evidence: Optional[EvidenceRecord]) -> List[str]:
    errors = []
    cluster_size = _visual_cluster_size(claim, evidence)
    if cluster_size is None or cluster_size < IMAGE_CLUSTER_MIN_SIZE:
        errors.append("image_cluster_too_small")

    cohesion_score = _visual_cluster_cohesion(claim, evidence)
    if cohesion_score is None or cohesion_score < IMAGE_CLUSTER_MIN_COHESION:
        errors.append("image_cluster_low_cohesion")

    source_count = _visual_cluster_source_count(evidence)
    if source_count is None or source_count < IMAGE_CLUSTER_MIN_SOURCE_COUNT:
        errors.append("image_cluster_insufficient_cross_source")

    return errors


def _validation_id(claim_id: str) -> str:
    return stable_id("val", {"claim_id": claim_id, "validator": VALIDATOR_VERSION})


def _quarantine_id(claim_id: str) -> str:
    return stable_id("q", {"claim_id": claim_id, "validator": VALIDATOR_VERSION})


def _review_metadata(review_decisions: Sequence[ReviewDecisionRecord]) -> List[Dict[str, object]]:
    return [review.model_dump(mode="json") for review in review_decisions]


def validate_raw_claims(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    claim_ids: Optional[Sequence[str]] = None,
) -> ClaimValidationRunResult:
    paths = config.jsonl_paths()
    requested_claim_ids = set(claim_ids or [])
    evidence_by_id = {record.evidence_id: record for _, record in read_jsonl_records(paths["evidence"], EvidenceRecord)}
    spans_by_id = {record.span_id: record for _, record in read_jsonl_records(paths["spans"], SpanRecord)}
    reviews_by_claim_id: Dict[str, List[ReviewDecisionRecord]] = {}
    for _, review in read_jsonl_records(paths["review_decisions"], ReviewDecisionRecord):
        reviews_by_claim_id.setdefault(review.claim_id, []).append(review)
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
        review_decisions = reviews_by_claim_id.get(claim.claim_id, [])
        decision = validate_claim_deterministically(
            claim,
            evidence,
            span,
            review_decisions=review_decisions,
        )

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
                metadata={
                    "source_modality": claim.source_modality,
                    "validation": decision.summary.model_dump(mode="json"),
                    "review_decisions": _review_metadata(review_decisions),
                },
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
                    confidence=claim.confidence,
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
