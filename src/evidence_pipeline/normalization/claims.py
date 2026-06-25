from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.normalization.entities import canonical_id
from evidence_pipeline.normalization.predicates import predicate_for_modality
from evidence_pipeline.schemas.claims import (
    EntityResolution,
    NormalizationDetails,
    NormalizedClaimRecord,
    PredicateMapping,
    RawClaimRecord,
    ValidatedClaimRecord,
)

NORMALIZER_VERSION = "normalizer.v1"


@dataclass
class ClaimNormalizationResult:
    created: int
    skipped: int


def _normalized_claim_id(claim_id: str) -> str:
    return stable_id("nclaim", {"claim_id": claim_id, "normalizer": NORMALIZER_VERSION})


def _subject_prefix(raw_claim: Optional[RawClaimRecord], validated_claim: ValidatedClaimRecord) -> str:
    if raw_claim is None:
        return "source"
    attribution_type = raw_claim.attribution.type
    if attribution_type == "speaker":
        return "speaker"
    if attribution_type == "model":
        return "model"
    if attribution_type == "human_reviewer":
        return "reviewer"
    if attribution_type == "document":
        return "source"
    return validated_claim.source_modality


def _subject_surface(raw_claim: Optional[RawClaimRecord], validated_claim: ValidatedClaimRecord) -> str:
    if raw_claim is not None and raw_claim.attribution.agent:
        return raw_claim.attribution.agent
    return validated_claim.source_id


def _object_value(raw_claim: Optional[RawClaimRecord], validated_claim: ValidatedClaimRecord) -> object:
    if raw_claim is not None and raw_claim.object is not None:
        return raw_claim.object
    if validated_claim.evidence_text:
        return validated_claim.evidence_text
    return validated_claim.source_faithful_claim


def _normalization_record(
    validated_claim: ValidatedClaimRecord,
    raw_claim: Optional[RawClaimRecord],
) -> NormalizedClaimRecord:
    subject_surface = _subject_surface(raw_claim, validated_claim)
    subject_id = canonical_id(_subject_prefix(raw_claim, validated_claim), subject_surface)
    predicate = predicate_for_modality(validated_claim.modality)
    raw_attribution = raw_claim.attribution.model_dump(mode="json") if raw_claim is not None else None

    return NormalizedClaimRecord(
        normalized_claim_id=_normalized_claim_id(validated_claim.claim_id),
        claim_id=validated_claim.claim_id,
        source_id=validated_claim.source_id,
        evidence_id=validated_claim.evidence_id,
        normalized_claim={
            "subject": subject_id,
            "predicate": predicate,
            "object": _object_value(raw_claim, validated_claim),
            "qualifiers": {
                "modality": validated_claim.modality,
                "truth_status": validated_claim.truth_status,
                "attribution": raw_attribution,
                "source_faithful_claim": validated_claim.source_faithful_claim,
            },
        },
        normalization=NormalizationDetails(
            entity_resolution=[
                EntityResolution(
                    surface=subject_surface,
                    canonical_id=subject_id,
                    confidence=1.0,
                    basis="attribution_agent" if raw_claim is not None and raw_claim.attribution.agent else "source_id",
                )
            ],
            predicate_mapping=PredicateMapping(surface=validated_claim.modality, canonical=predicate),
            metadata={
                "normalizer_version": NORMALIZER_VERSION,
                "raw_claim_available": raw_claim is not None,
            },
        ),
    )


def normalize_claims(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    claim_ids: Optional[Sequence[str]] = None,
) -> ClaimNormalizationResult:
    paths = config.jsonl_paths()
    requested_claim_ids = set(claim_ids or [])
    raw_by_claim_id = {
        claim.claim_id: claim
        for _, claim in read_jsonl_records(paths["claims_raw"], RawClaimRecord)
    }
    existing_ids = existing_values(paths["claims_normalized"], "normalized_claim_id")
    created = 0
    skipped = 0

    for _, validated_claim in read_jsonl_records(paths["claims_validated"], ValidatedClaimRecord):
        if validated_claim.support_status != "accepted_extracted":
            continue
        if source_id is not None and validated_claim.source_id != source_id:
            continue
        if requested_claim_ids and validated_claim.claim_id not in requested_claim_ids:
            continue
        normalized_id = _normalized_claim_id(validated_claim.claim_id)
        if normalized_id in existing_ids:
            skipped += 1
            continue
        append_jsonl(
            paths["claims_normalized"],
            _normalization_record(validated_claim, raw_by_claim_id.get(validated_claim.claim_id)),
        )
        existing_ids.add(normalized_id)
        created += 1

    return ClaimNormalizationResult(created=created, skipped=skipped)
