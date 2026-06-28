from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.reports import RetentionPlanRecord
from evidence_pipeline.schemas.sources import SourceRecord


@dataclass
class RetentionPlanResult:
    output_path: Path
    candidate_count: int


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _candidate_record(source: SourceRecord, age_days: int, retention_days: int) -> dict:
    action = "delete_raw_source"
    return RetentionPlanRecord(
        retention_id=stable_id(
            "retention",
            {
                "source_id": source.source_id,
                "source_file": source.source_file,
                "action": action,
                "retention_days": retention_days,
            },
        ),
        action=action,
        source_id=source.source_id,
        source_modality=source.source_modality,
        source_file=source.source_file,
        source_uri=source.source_uri,
        ingested_at=source.ingested_at,
        age_days=age_days,
        retention_days=retention_days,
        reason_code="raw_source_retention_exceeded",
        dry_run=True,
    ).model_dump(mode="json", exclude_none=True)


def write_retention_plan(
    config: PipelineConfig,
    output_path: Optional[Path] = None,
    now: Optional[datetime] = None,
) -> RetentionPlanResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "retention_plan.jsonl"
    current_time = _utc(now or datetime.now(timezone.utc))
    retention_days = config.retention.raw_source_retention_days

    candidates = []
    for _, source in read_jsonl_records(config.jsonl_paths()["sources"], SourceRecord):
        age_days = (current_time - _utc(source.ingested_at)).days
        if age_days >= retention_days:
            candidates.append(_candidate_record(source, age_days, retention_days))

    write_jsonl(output_path, candidates)
    return RetentionPlanResult(output_path=output_path, candidate_count=len(candidates))
