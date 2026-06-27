from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, Field


class PathConfig(BaseModel):
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    work_dir: Path = Path("data/work")
    jsonl_dir: Path = Path("data/jsonl")
    reports_dir: Path = Path("data/reports")


class JSONLConfig(BaseModel):
    sources: str = "sources.jsonl"
    chat_messages: str = "chat_messages.jsonl"
    pdf_blocks: str = "pdf_blocks.jsonl"
    audio_utterances: str = "audio_utterances.jsonl"
    images: str = "images.jsonl"
    image_regions: str = "image_regions.jsonl"
    image_region_embeddings: str = "image_region_embeddings.jsonl"
    image_feature_clusters: str = "image_feature_clusters.jsonl"
    evidence: str = "evidence.jsonl"
    chunks: str = "chunks.jsonl"
    spans: str = "spans.jsonl"
    claims_raw: str = "claims.raw.jsonl"
    validations: str = "validations.jsonl"
    claims_validated: str = "claims.validated.jsonl"
    claims_normalized: str = "claims.normalized.jsonl"
    jobs: str = "jobs.jsonl"
    review_decisions: str = "review_decisions.jsonl"
    audit_events: str = "audit_events.jsonl"
    errors: str = "errors.jsonl"
    quarantine: str = "quarantine.jsonl"


class PrivacyConfig(BaseModel):
    local_only_sensitive_sources: bool = True
    sensitive_metadata_keys: List[str] = Field(
        default_factory=lambda: ["sensitive", "contains_pii", "local_only"]
    )
    local_model_providers: List[str] = Field(default_factory=lambda: ["deterministic", "local"])


class RetentionConfig(BaseModel):
    raw_source_retention_days: int = 365


class PipelineConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    jsonl: JSONLConfig = Field(default_factory=JSONLConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    def jsonl_paths(self) -> Dict[str, Path]:
        return {
            key: self.paths.jsonl_dir / value
            for key, value in self.jsonl.model_dump().items()
        }


def load_config(path: Path = Path("configs/pipeline.yaml")) -> PipelineConfig:
    if not path.exists():
        return PipelineConfig()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return PipelineConfig.model_validate(payload)
