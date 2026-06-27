from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from PIL import Image

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.image import ImageRecord, ImageRegionRecord


@dataclass
class ImageRegionProposalResult:
    created: int
    skipped: int


def _positions(size: int, patch_size: int, stride: int) -> List[int]:
    if size <= patch_size:
        return [0]
    positions = list(range(0, size - patch_size + 1, stride))
    final = size - patch_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def _grid_bboxes(width: int, height: int, patch_size: int, stride: int) -> Iterable[List[int]]:
    patch_width = min(width, patch_size)
    patch_height = min(height, patch_size)
    for y in _positions(height, patch_height, stride):
        for x in _positions(width, patch_width, stride):
            yield [x, y, patch_width, patch_height]


def _region_id(image_id: str, bbox: List[int], proposal_method: str) -> str:
    return stable_id("img_region", {"image_id": image_id, "bbox": bbox, "proposal_method": proposal_method})


def propose_image_regions(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    patch_size: int = 224,
    stride: int = 112,
) -> ImageRegionProposalResult:
    paths = config.jsonl_paths()
    existing_region_ids = existing_values(paths["image_regions"], "region_id")
    proposal_method = f"grid_{patch_size}_stride{stride}"
    created = 0
    skipped = 0

    for _, image_record in read_jsonl_records(paths["images"], ImageRecord):
        if source_id is not None and image_record.source_id != source_id:
            continue
        image_path = Path(image_record.normalized_file or image_record.source_file)
        with Image.open(image_path) as image:
            for bbox in _grid_bboxes(image_record.width, image_record.height, patch_size, stride):
                region_id = _region_id(image_record.image_id, bbox, proposal_method)
                if region_id in existing_region_ids:
                    skipped += 1
                    continue
                x, y, width, height = bbox
                crop_path = config.paths.work_dir / "crops" / f"{region_id}.png"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                image.crop((x, y, x + width, y + height)).save(crop_path)
                risk_flags = []
                if width < patch_size or height < patch_size:
                    risk_flags.append("edge_patch")
                append_jsonl(
                    paths["image_regions"],
                    ImageRegionRecord(
                        region_id=region_id,
                        image_id=image_record.image_id,
                        source_id=image_record.source_id,
                        region_type="patch",
                        bbox=bbox,
                        crop_path=str(crop_path),
                        proposal_method=proposal_method,
                        risk_flags=risk_flags,
                    ),
                )
                existing_region_ids.add(region_id)
                created += 1

    return ImageRegionProposalResult(created=created, skipped=skipped)
