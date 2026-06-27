from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel, utc_now

JobStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


class JobRecord(StrictModel):
    job_id: str
    stage: str
    source_id: Optional[str] = None
    input_record_ids: List[str] = Field(default_factory=list)
    config_hash: str
    model_hash: Optional[str] = None
    prompt_hash: Optional[str] = None
    status: JobStatus
    attempts: int = 1
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    error: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    schema_version: str = "job.v1"

    @model_validator(mode="after")
    def validate_job(self) -> "JobRecord":
        if not self.stage.strip():
            raise ValueError("stage must not be empty")
        if self.attempts < 0:
            raise ValueError("attempts must be non-negative")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self
