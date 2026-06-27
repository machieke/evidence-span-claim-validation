from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.image import ImageRecord

LOW_OCR_CONFIDENCE_THRESHOLD = 0.75


@dataclass
class ImageOCRIngestResult:
    created: int
    skipped: int


def _ocr_items(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        for key in ("ocr", "regions", "texts"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("image OCR input must be a JSON object, list, or object with ocr/regions/texts")
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("every OCR item must be a JSON object")
        yield item


def _image_lookup(config: PipelineConfig) -> Dict[str, ImageRecord]:
    paths = config.jsonl_paths()
    lookup: Dict[str, ImageRecord] = {}
    for _, image in read_jsonl_records(paths["images"], ImageRecord):
        lookup[image.image_id] = image
        lookup[image.source_id] = image
        lookup[Path(image.source_file).name] = image
    return lookup


def _bbox(value: object) -> Optional[List[int]]:
    if not isinstance(value, list) or len(value) != 4:
        return None
    return [int(item) for item in value]


def _risk_flags(item: Dict[str, Any]) -> List[str]:
    raw_flags = item.get("risk_flags", [])
    if isinstance(raw_flags, str):
        raw_flags = [raw_flags]
    flags = set(str(flag) for flag in raw_flags if flag)
    confidence = item.get("ocr_confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        if confidence < LOW_OCR_CONFIDENCE_THRESHOLD:
            flags.add("low_ocr_confidence")
    return sorted(flags)


def _evidence_record(item: Dict[str, Any], image: ImageRecord) -> EvidenceRecord:
    text = str(item.get("text") or item.get("ocr_text") or "").strip()
    if not text:
        raise ValueError("OCR item text must not be empty")
    bbox = _bbox(item.get("bbox"))
    ocr_model = str(item.get("ocr_model") or item.get("model") or "unknown_ocr_model")
    evidence_id = stable_id(
        "ev_ocr",
        {
            "source_id": image.source_id,
            "image_id": image.image_id,
            "bbox": bbox,
            "text": text,
            "ocr_model": ocr_model,
        },
    )
    provenance: Dict[str, Any] = {
        "image_id": image.image_id,
        "bbox": bbox,
        "ocr_model": ocr_model,
        "ocr_confidence": item.get("ocr_confidence"),
    }
    return EvidenceRecord(
        evidence_id=evidence_id,
        source_id=image.source_id,
        source_modality="image",
        evidence_type="ocr_text_span",
        text=text,
        provenance={key: value for key, value in provenance.items() if value is not None},
        risk_flags=_risk_flags(item),
    )


def ingest_image_ocr(ocr_path: Path, config: PipelineConfig) -> ImageOCRIngestResult:
    image_by_key = _image_lookup(config)
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["evidence"], "evidence_id")
    created = 0
    skipped = 0

    for item in _ocr_items(ocr_path):
        image_key = item.get("image_id") or item.get("source_id") or item.get("image_file")
        if image_key is None or str(image_key) not in image_by_key:
            raise ValueError(f"OCR item references unknown image: {image_key}")
        record = _evidence_record(item, image_by_key[str(image_key)])
        if record.evidence_id in existing_ids:
            skipped += 1
            continue
        append_jsonl(paths["evidence"], record)
        existing_ids.add(record.evidence_id)
        created += 1

    return ImageOCRIngestResult(created=created, skipped=skipped)
