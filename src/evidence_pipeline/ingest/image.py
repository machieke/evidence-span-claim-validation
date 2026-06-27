from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values
from evidence_pipeline.schemas.image import ImageRecord
from evidence_pipeline.schemas.sources import SourceRecord

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class ImageIngestResult:
    sources_created: int
    images_created: int
    images_skipped: int
    source_ids: List[str]


def iter_image_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            yield item


def _exif_datetime(image: Image.Image) -> Optional[str]:
    try:
        exif = image.getexif()
    except Exception:
        return None
    for tag in (36867, 306):
        value = exif.get(tag)
        if value:
            return str(value)
    return None


def _image_record(path: Path, metadata: Dict[str, Any]) -> ImageRecord:
    sha256 = sha256_file(path)
    source_id = stable_id("src", {"modality": "image", "sha256": sha256})
    with Image.open(path) as image:
        width, height = image.size
        return ImageRecord(
            image_id=source_id,
            source_id=source_id,
            source_file=str(path),
            normalized_file=str(path),
            width=width,
            height=height,
            color_mode=image.mode,
            image_format=image.format,
            exif_datetime=_exif_datetime(image),
            sha256=sha256,
            metadata=metadata,
        )


def ingest_images(path: Path, config: PipelineConfig, metadata: Optional[Dict[str, Any]] = None) -> ImageIngestResult:
    metadata = dict(metadata or {})
    paths = config.jsonl_paths()
    existing_source_ids = existing_values(paths["sources"], "source_id")
    existing_image_ids = existing_values(paths["images"], "image_id")
    sources_created = 0
    images_created = 0
    images_skipped = 0
    source_ids: List[str] = []

    for image_path in iter_image_paths(path):
        record = _image_record(image_path, metadata)
        source_ids.append(record.source_id)
        if record.source_id not in existing_source_ids:
            append_jsonl(
                paths["sources"],
                SourceRecord(
                    source_id=record.source_id,
                    source_modality="image",
                    source_file=record.source_file,
                    sha256=record.sha256,
                    metadata={**metadata, "width": record.width, "height": record.height},
                ),
            )
            existing_source_ids.add(record.source_id)
            sources_created += 1
        if record.image_id in existing_image_ids:
            images_skipped += 1
            continue
        append_jsonl(paths["images"], record)
        existing_image_ids.add(record.image_id)
        images_created += 1

    return ImageIngestResult(
        sources_created=sources_created,
        images_created=images_created,
        images_skipped=images_skipped,
        source_ids=source_ids,
    )
