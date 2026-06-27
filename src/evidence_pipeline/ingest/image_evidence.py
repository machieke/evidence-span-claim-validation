from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.image import ImageRegionRecord


@dataclass
class ImageEvidenceResult:
    created: int
    skipped: int


def _image_evidence_id(region: ImageRegionRecord) -> str:
    return stable_id("ev_img", {"source_id": region.source_id, "region_id": region.region_id})


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
