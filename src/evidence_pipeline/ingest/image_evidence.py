from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.image import ImageFeatureClusterRecord, ImageRegionRecord


@dataclass
class ImageEvidenceResult:
    created: int
    skipped: int


def _image_evidence_id(region: ImageRegionRecord) -> str:
    return stable_id("ev_img", {"source_id": region.source_id, "region_id": region.region_id})


def _image_cluster_evidence_id(cluster: ImageFeatureClusterRecord) -> str:
    return stable_id("ev_vf", {"feature_cluster_id": cluster.feature_cluster_id})


def build_image_evidence(config: PipelineConfig, source_id: Optional[str] = None) -> ImageEvidenceResult:
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["evidence"], "evidence_id")
    created = 0
    skipped = 0

    for _, region in read_jsonl_records(paths["image_regions"], ImageRegionRecord):
        if source_id is not None and region.source_id != source_id:
            continue
        evidence_id = _image_evidence_id(region)
        if evidence_id in existing_ids:
            skipped += 1
            continue
        provenance: Dict[str, object] = {
            "image_id": region.image_id,
            "region_id": region.region_id,
            "bbox": region.bbox,
            "crop_path": region.crop_path,
            "mask_path": region.mask_path,
            "proposal_method": region.proposal_method,
            "proposal_score": region.proposal_score,
        }
        append_jsonl(
            paths["evidence"],
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=region.source_id,
                source_modality="image",
                evidence_type="visual_region",
                text=None,
                provenance={key: value for key, value in provenance.items() if value is not None},
                risk_flags=region.risk_flags,
            ),
        )
        existing_ids.add(evidence_id)
        created += 1

    return ImageEvidenceResult(created=created, skipped=skipped)


def build_image_cluster_evidence(config: PipelineConfig, source_id: Optional[str] = None) -> ImageEvidenceResult:
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["evidence"], "evidence_id")
    created = 0
    skipped = 0

    for _, cluster in read_jsonl_records(paths["image_feature_clusters"], ImageFeatureClusterRecord):
        if source_id is not None and source_id not in cluster.source_ids:
            continue
        evidence_id = _image_cluster_evidence_id(cluster)
        if evidence_id in existing_ids:
            skipped += 1
            continue
        source_ids = cluster.source_ids or [cluster.feature_cluster_id]
        provenance: Dict[str, object] = {
            "feature_cluster_id": cluster.feature_cluster_id,
            "embedding_model": cluster.embedding_model,
            "clustering_method": cluster.clustering_method,
            "member_region_ids": cluster.member_region_ids,
            "representative_region_ids": cluster.representative_region_ids,
            "cluster_size": cluster.cluster_size,
            "cohesion_score": cluster.cohesion_score,
            "nearest_neighbor_margin": cluster.nearest_neighbor_margin,
            "source_ids": source_ids,
        }
        append_jsonl(
            paths["evidence"],
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=source_ids[0],
                source_modality="image",
                evidence_type="visual_cluster",
                text=None,
                provenance={key: value for key, value in provenance.items() if value is not None},
                risk_flags=cluster.risk_flags,
            ),
        )
        existing_ids.add(evidence_id)
        created += 1

    return ImageEvidenceResult(created=created, skipped=skipped)
