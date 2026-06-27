from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel

ImageRegionType = Literal["patch", "segmentation_mask", "detector_box", "ocr_box"]


class ImageRecord(StrictModel):
    image_id: str
    source_id: str
    source_modality: Literal["image"] = "image"
    source_file: str
    normalized_file: Optional[str] = None
    width: int
    height: int
    color_mode: Optional[str] = None
    image_format: Optional[str] = None
    exif_datetime: Optional[str] = None
    sha256: Optional[str] = None
    perceptual_hash: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "image.v1"

    @model_validator(mode="after")
    def validate_image(self) -> "ImageRecord":
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image width and height must be positive")
        if self.normalized_file is None:
            self.normalized_file = self.source_file
        return self


class ImageRegionRecord(StrictModel):
    region_id: str
    image_id: str
    source_id: str
    region_type: ImageRegionType = "patch"
    bbox: List[int]
    crop_path: Optional[str] = None
    mask_path: Optional[str] = None
    proposal_method: str
    proposal_score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "image.region.v1"

    @model_validator(mode="after")
    def validate_region(self) -> "ImageRegionRecord":
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain x, y, width, height")
        x, y, width, height = self.bbox
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise ValueError("bbox coordinates must be non-negative with positive width and height")
        if self.proposal_score is not None and not (0 <= self.proposal_score <= 1):
            raise ValueError("proposal_score must be between 0 and 1")
        return self
