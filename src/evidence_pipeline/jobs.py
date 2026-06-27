from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values
from evidence_pipeline.schemas.base import utc_now
from evidence_pipeline.schemas.jobs import JobRecord


@dataclass
class JobRecordResult:
    job_id: str
    created: bool


def _config_hash(config: PipelineConfig) -> str:
    return stable_id("cfg", config.model_dump(mode="json"), length=16)


def _optional_hash(prefix: str, value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return stable_id(prefix, value, length=16)


def _input_record_ids(values: Optional[Iterable[str]]) -> List[str]:
    return sorted({str(value) for value in values or []})


def record_job_result(
    config: PipelineConfig,
    stage: str,
    source_id: Optional[str] = None,
    input_record_ids: Optional[Iterable[str]] = None,
    model_id: Optional[str] = None,
    prompt_id: Optional[str] = None,
    status: str = "succeeded",
    metrics: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> JobRecordResult:
    paths = config.jsonl_paths()
    config_hash = _config_hash(config)
    model_hash = _optional_hash("model", model_id)
    prompt_hash = _optional_hash("prompt", prompt_id)
    normalized_input_ids = _input_record_ids(input_record_ids)
    job_id = stable_id(
        "job",
        {
            "stage": stage,
            "source_id": source_id,
            "input_record_ids": normalized_input_ids,
            "config_hash": config_hash,
            "model_hash": model_hash,
            "prompt_hash": prompt_hash,
        },
    )

    if job_id in existing_values(paths["jobs"], "job_id"):
        return JobRecordResult(job_id=job_id, created=False)

    now = utc_now()
    append_jsonl(
        paths["jobs"],
        JobRecord(
            job_id=job_id,
            stage=stage,
            source_id=source_id,
            input_record_ids=normalized_input_ids,
            config_hash=config_hash,
            model_hash=model_hash,
            prompt_hash=prompt_hash,
            status=status,
            attempts=1,
            created_at=now,
            updated_at=now,
            error=error,
            metrics=metrics or {},
            metadata=metadata or {},
        ),
    )
    return JobRecordResult(job_id=job_id, created=True)
