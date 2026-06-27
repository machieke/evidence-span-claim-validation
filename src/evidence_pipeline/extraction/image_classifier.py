from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.image import ImageRegionEmbeddingRecord

COLOR_CLASSIFIER_MODEL = "dominant_color_classifier_v1"
COLOR_EMBEDDING_MODEL = "color_rgb_mean_std_v1"


@dataclass
class ImageRegionClassificationResult:
    created: int
    skipped: int


def _evidence_by_region_id(config: PipelineConfig) -> Dict[str, EvidenceRecord]:
    paths = config.jsonl_paths()
    by_region_id: Dict[str, EvidenceRecord] = {}
    for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord):
        if evidence.source_modality != "image" or evidence.evidence_type != "visual_region":
            continue
        region_id = evidence.provenance.get("region_id")
        if region_id is not None:
            by_region_id[str(region_id)] = evidence
    return by_region_id


def _dominant_color(vector: List[float]) -> Tuple[str, float]:
    red, green, blue = (vector + [0.0, 0.0, 0.0])[:3]
    channels = {"red": red, "green": green, "blue": blue}
    maximum = max(channels.values())
    minimum = min(channels.values())
    spread = maximum - minimum

    if maximum < 0.12:
        return "black", 0.9
    if minimum > 0.88:
        return "white", 0.9
    if spread < 0.08:
        return "gray", round(0.55 + min(0.25, 0.08 - spread), 3)

    sorted_channels = sorted(channels.items(), key=lambda item: item[1], reverse=True)
    top_name, top_value = sorted_channels[0]
    second_name, second_value = sorted_channels[1]
    third_value = sorted_channels[2][1]
    if second_value > 0.45 and second_value >= top_value * 0.65 and third_value < top_value * 0.5:
        pair = frozenset({top_name, second_name})
        if pair == {"red", "green"}:
            label = "yellow"
        elif pair == {"green", "blue"}:
            label = "cyan"
        else:
            label = "magenta"
    else:
        label = top_name
    confidence = round(max(0.5, min(1.0, 0.55 + spread / 2)), 3)
    return label, confidence


def _classification_claim_id(evidence: EvidenceRecord, region_id: str, label: str, classifier_model: str) -> str:
    return stable_id(
        "claim_img_cls",
        {
            "evidence_id": evidence.evidence_id,
            "region_id": region_id,
            "label": label,
            "classifier_model": classifier_model,
        },
    )


def _classification_claim(
    embedding: ImageRegionEmbeddingRecord,
    evidence: EvidenceRecord,
    label: str,
    confidence: float,
    classifier_model: str,
) -> RawClaimRecord:
    return RawClaimRecord(
        claim_id=_classification_claim_id(evidence, embedding.region_id, label, classifier_model),
        source_id=evidence.source_id,
        source_modality="image",
        span_id=None,
        evidence_id=evidence.evidence_id,
        claim_type="named_visual_classification",
        source_faithful_claim=f"Model {classifier_model} classified region {embedding.region_id} as {label}.",
        subject=embedding.region_id,
        predicate="classified_as",
        object=label,
        quantity=None,
        attributes={
            "classifier": {
                "model": classifier_model,
                "confidence": confidence,
                "basis": "region_color_embedding",
            },
            "embedding_id": embedding.embedding_id,
            "embedding_model": embedding.embedding_model,
        },
        modality="model_observation",
        evidence_text=None,
        context_dependent=False,
        context_used=None,
        attribution={"type": "model", "agent": classifier_model},
        truth_status="model_observation_unverified",
        confidence=confidence,
        model={
            "provider": "deterministic",
            "model": classifier_model,
            "prompt_version": None,
        },
        support_status="raw_extracted",
        risk_flags=sorted(set(evidence.risk_flags) | set(embedding.risk_flags) | {"color_only_classification"}),
    )


def classify_image_regions(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    embedding_model: str = COLOR_EMBEDDING_MODEL,
    classifier_model: str = COLOR_CLASSIFIER_MODEL,
) -> ImageRegionClassificationResult:
    paths = config.jsonl_paths()
    evidence_by_region_id = _evidence_by_region_id(config)
    existing_claim_ids = existing_values(paths["claims_raw"], "claim_id")
    created = 0
    skipped = 0

    for _, embedding in read_jsonl_records(paths["image_region_embeddings"], ImageRegionEmbeddingRecord):
        if embedding.embedding_model != embedding_model:
            continue
        if source_id is not None and embedding.source_id != source_id:
            continue
        evidence = evidence_by_region_id.get(embedding.region_id)
        if evidence is None:
            skipped += 1
            continue
        label, confidence = _dominant_color(embedding.vector)
        claim = _classification_claim(embedding, evidence, label, confidence, classifier_model)
        if claim.claim_id in existing_claim_ids:
            skipped += 1
            continue
        append_jsonl(paths["claims_raw"], claim)
        existing_claim_ids.add(claim.claim_id)
        created += 1

    return ImageRegionClassificationResult(created=created, skipped=skipped)
