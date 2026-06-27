from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.validation.text_support import contains_negation, contains_uncertainty

RULE_EXTRACTOR_VERSION = "rules.v1"
IMAGE_REGION_EXTRACTOR_VERSION = "image_region.rules.v1"
IMAGE_CLUSTER_EXTRACTOR_VERSION = "image_cluster.rules.v1"
IMAGE_OCR_EXTRACTOR_VERSION = "image_ocr.rules.v1"
SUPPORTED_MODALITIES = {"all", "chat", "pdf", "audio", "image"}
TEXT_SPAN_MODALITIES = {"chat", "pdf", "audio", "image"}


@dataclass
class ClaimExtractionResult:
    created: int
    skipped: int


def _claim_id(span: SpanRecord, extractor: str) -> str:
    return stable_id(
        "claim",
        {
            "span_id": span.span_id,
            "evidence_id": span.evidence_id,
            "text": span.text,
            "extractor": extractor,
        },
    )


def _modality_for_text(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("?"):
        return "question_asked"
    if contains_negation(stripped):
        return "negated"
    if contains_uncertainty(stripped):
        return "uncertain_observation"
    return "asserted"


def _chat_claim_text(span: SpanRecord) -> str:
    if (span.text or "").strip().endswith("?"):
        return f"The speaker asked: {span.text}"
    return f"The speaker asserted: {span.text}"


def _audio_claim_text(span: SpanRecord) -> str:
    if (span.text or "").strip().endswith("?"):
        return f"The speaker asked: {span.text}"
    return f"The speaker asserted: {span.text}"


def _pdf_claim_text(span: SpanRecord) -> str:
    return f"The document states: {span.text}"


def _attribution(span: SpanRecord, evidence: EvidenceRecord) -> Dict[str, Optional[str]]:
    if span.source_modality == "chat":
        agent = evidence.provenance.get("sender_id") or evidence.provenance.get("sender_display_name")
        return {"type": "speaker", "agent": str(agent) if agent is not None else None}
    if span.source_modality == "audio":
        agent = evidence.provenance.get("speaker")
        return {"type": "speaker", "agent": str(agent) if agent is not None else None}
    if span.source_modality == "pdf":
        return {"type": "document", "agent": evidence.source_id}
    if span.source_modality == "image" and evidence.evidence_type == "ocr_text_span":
        return {"type": "model", "agent": str(evidence.provenance.get("ocr_model") or "unknown_ocr_model")}
    return {"type": "unknown", "agent": None}


def _truth_status(span: SpanRecord) -> str:
    if span.source_modality in {"chat", "audio"}:
        return "speaker_asserted_unverified"
    if span.source_modality == "image":
        return "model_observation_unverified"
    return "source_asserted_unverified"


def _source_faithful_claim(span: SpanRecord) -> str:
    if span.source_modality == "chat":
        return _chat_claim_text(span)
    if span.source_modality == "audio":
        return _audio_claim_text(span)
    if span.source_modality == "pdf":
        return _pdf_claim_text(span)
    if span.source_modality == "image":
        return f"OCR text states: {span.text}"
    return f"The source states: {span.text}"


def _risk_flags(span: SpanRecord, evidence: EvidenceRecord) -> List[str]:
    return sorted(set(span.risk_flags) | set(evidence.risk_flags))


def _raw_claim_from_span(span: SpanRecord, evidence: EvidenceRecord) -> RawClaimRecord:
    if not span.text:
        raise ValueError("text span is required for deterministic extraction")
    risk_flags = _risk_flags(span, evidence)
    context_dependent = "context_dependent_coreference" in risk_flags and bool(span.context_text)
    extractor = _extractor_for_span(span, evidence)
    return RawClaimRecord(
        claim_id=_claim_id(span, extractor),
        source_id=span.source_id,
        source_modality=span.source_modality,
        span_id=span.span_id,
        evidence_id=span.evidence_id,
        claim_type=_claim_type_for_span(span, evidence),
        source_faithful_claim=_source_faithful_claim(span),
        subject=None,
        predicate=None,
        object=None,
        quantity=None,
        attributes={"extractor": extractor},
        modality=_modality_for_text(span.text),
        evidence_text=span.text,
        context_dependent=context_dependent,
        context_used=span.context_text if context_dependent else None,
        attribution=_attribution(span, evidence),
        truth_status=_truth_status(span),
        confidence=span.score if span.score is not None else 0.5,
        model={
            "provider": "deterministic",
            "model": extractor,
            "prompt_version": None,
        },
        support_status="raw_extracted",
        risk_flags=risk_flags,
    )


def _claim_type_for_span(span: SpanRecord, evidence: EvidenceRecord) -> str:
    if span.source_modality == "image" and evidence.evidence_type == "ocr_text_span":
        return "ocr_text_claim"
    return "attributed_text_claim"


def _extractor_for_span(span: SpanRecord, evidence: EvidenceRecord) -> str:
    if span.source_modality == "image" and evidence.evidence_type == "ocr_text_span":
        return IMAGE_OCR_EXTRACTOR_VERSION
    return RULE_EXTRACTOR_VERSION


def _raw_image_claim_from_evidence(evidence: EvidenceRecord) -> RawClaimRecord:
    provenance = evidence.provenance
    region_id = str(provenance.get("region_id") or evidence.evidence_id)
    proposal_method = str(provenance.get("proposal_method") or "unknown_region_proposal")
    proposal_score = provenance.get("proposal_score")
    confidence = (
        float(proposal_score)
        if isinstance(proposal_score, (int, float)) and not isinstance(proposal_score, bool)
        else 0.5
    )

    return RawClaimRecord(
        claim_id=stable_id(
            "claim_img",
            {
                "evidence_id": evidence.evidence_id,
                "region_id": region_id,
                "proposal_method": proposal_method,
            },
        ),
        source_id=evidence.source_id,
        source_modality="image",
        span_id=None,
        evidence_id=evidence.evidence_id,
        claim_type="visual_region_proposal",
        source_faithful_claim=f"Region {region_id} was proposed as a visual region by {proposal_method}.",
        subject=region_id,
        predicate="proposed_visual_region",
        object={
            "bbox": provenance.get("bbox"),
            "crop_path": provenance.get("crop_path"),
            "mask_path": provenance.get("mask_path"),
        },
        quantity=None,
        attributes={
            "extractor": IMAGE_REGION_EXTRACTOR_VERSION,
            "proposal_method": proposal_method,
        },
        modality="model_observation",
        evidence_text=None,
        context_dependent=False,
        context_used=None,
        attribution={"type": "model", "agent": proposal_method},
        truth_status="model_observation_unverified",
        confidence=confidence,
        model={
            "provider": "deterministic",
            "model": proposal_method,
            "prompt_version": None,
        },
        support_status="raw_extracted",
        risk_flags=evidence.risk_flags,
    )


def _raw_image_cluster_claim_from_evidence(evidence: EvidenceRecord) -> RawClaimRecord:
    provenance = evidence.provenance
    feature_cluster_id = str(provenance.get("feature_cluster_id") or evidence.evidence_id)
    embedding_model = str(provenance.get("embedding_model") or "unknown_embedding_model")
    clustering_method = str(provenance.get("clustering_method") or "unknown_clustering_method")
    raw_member_region_ids = provenance.get("member_region_ids") or []
    if isinstance(raw_member_region_ids, list):
        member_region_ids = [str(region_id) for region_id in raw_member_region_ids]
    else:
        member_region_ids = [str(raw_member_region_ids)]
    cohesion_score = provenance.get("cohesion_score")
    confidence = (
        float(cohesion_score)
        if isinstance(cohesion_score, (int, float)) and not isinstance(cohesion_score, bool)
        else 0.5
    )
    agent = f"{embedding_model}+{clustering_method}"

    return RawClaimRecord(
        claim_id=stable_id(
            "claim_vf",
            {
                "evidence_id": evidence.evidence_id,
                "feature_cluster_id": feature_cluster_id,
                "embedding_model": embedding_model,
                "clustering_method": clustering_method,
            },
        ),
        source_id=evidence.source_id,
        source_modality="image",
        span_id=None,
        evidence_id=evidence.evidence_id,
        claim_type="unnamed_visual_feature_cluster",
        source_faithful_claim=(
            f"Regions {', '.join(member_region_ids)} were clustered as visually similar under {embedding_model}."
        ),
        subject=feature_cluster_id,
        predicate="has_member_regions",
        object=member_region_ids,
        quantity=None,
        attributes={
            "extractor": IMAGE_CLUSTER_EXTRACTOR_VERSION,
            "embedding_model": embedding_model,
            "clustering_method": clustering_method,
            "cohesion_score": cohesion_score,
            "nearest_neighbor_margin": provenance.get("nearest_neighbor_margin"),
            "representative_region_ids": provenance.get("representative_region_ids"),
        },
        modality="model_observation",
        evidence_text=None,
        context_dependent=False,
        context_used=None,
        attribution={"type": "model", "agent": agent},
        truth_status="model_observation_unverified",
        confidence=confidence,
        model={
            "provider": "deterministic",
            "model": agent,
            "prompt_version": None,
        },
        support_status="raw_extracted",
        risk_flags=evidence.risk_flags,
    )


def extract_claims_from_spans(
    config: PipelineConfig,
    modality: str = "all",
    source_id: Optional[str] = None,
) -> ClaimExtractionResult:
    if modality not in SUPPORTED_MODALITIES:
        raise ValueError("baseline extractor currently supports all, chat, pdf, audio, or image")

    paths = config.jsonl_paths()
    evidence_by_id = {
        evidence.evidence_id: evidence
        for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord)
    }
    existing_claim_ids = existing_values(paths["claims_raw"], "claim_id")
    created = 0
    skipped = 0

    if modality in {"all", "image"}:
        for evidence in evidence_by_id.values():
            if evidence.source_modality != "image" or evidence.evidence_type not in {
                "visual_region",
                "visual_cluster",
            }:
                continue
            if source_id is not None and evidence.source_id != source_id:
                continue
            if evidence.evidence_type == "visual_cluster":
                record = _raw_image_cluster_claim_from_evidence(evidence)
            else:
                record = _raw_image_claim_from_evidence(evidence)
            if record.claim_id in existing_claim_ids:
                skipped += 1
                continue
            append_jsonl(paths["claims_raw"], record)
            existing_claim_ids.add(record.claim_id)
            created += 1

    for _, span in read_jsonl_records(paths["spans"], SpanRecord):
        if span.label != "claim_bearing":
            continue
        if modality != "all" and span.source_modality != modality:
            continue
        if source_id is not None and span.source_id != source_id:
            continue
        if span.source_modality not in TEXT_SPAN_MODALITIES:
            continue
        evidence = evidence_by_id.get(span.evidence_id)
        if evidence is None:
            skipped += 1
            continue
        if span.source_modality == "image" and evidence.evidence_type != "ocr_text_span":
            continue
        record = _raw_claim_from_span(span, evidence)
        if record.claim_id in existing_claim_ids:
            skipped += 1
            continue
        append_jsonl(paths["claims_raw"], record)
        existing_claim_ids.add(record.claim_id)
        created += 1

    return ClaimExtractionResult(created=created, skipped=skipped)
