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


class ImageRegionEmbeddingRecord(StrictModel):
    embedding_id: str
    region_id: str
    image_id: str
    source_id: str
    embedding_model: str
    embedding_dim: int
    vector: List[float]
    embedding_path: Optional[str] = None
    preprocessing: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "image.region_embedding.v1"

    @model_validator(mode="after")
    def validate_embedding(self) -> "ImageRegionEmbeddingRecord":
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if len(self.vector) != self.embedding_dim:
            raise ValueError("vector length must match embedding_dim")
        return self


class ImageFeatureClusterRecord(StrictModel):
    feature_cluster_id: str
    embedding_model: str
    clustering_method: str
    member_region_ids: List[str]
    cluster_size: int
    cohesion_score: Optional[float] = None
    nearest_neighbor_margin: Optional[float] = None
    representative_region_ids: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    status: Literal["unnamed", "named", "rejected"] = "unnamed"
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "image.feature_cluster.v1"

    @model_validator(mode="after")
    def validate_cluster(self) -> "ImageFeatureClusterRecord":
        if self.cluster_size <= 0:
            raise ValueError("cluster_size must be positive")
        if self.cluster_size != len(self.member_region_ids):
            raise ValueError("cluster_size must match member_region_ids length")
        if not self.representative_region_ids:
            raise ValueError("representative_region_ids must not be empty")
        if any(region_id not in self.member_region_ids for region_id in self.representative_region_ids):
            raise ValueError("representative_region_ids must be cluster members")
        if self.cohesion_score is not None and not (0 <= self.cohesion_score <= 1):
            raise ValueError("cohesion_score must be between 0 and 1")
        if self.nearest_neighbor_margin is not None and not (0 <= self.nearest_neighbor_margin <= 1):
            raise ValueError("nearest_neighbor_margin must be between 0 and 1")
        return self
