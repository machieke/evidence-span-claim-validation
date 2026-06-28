from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.audit import AuditEventRecord
from evidence_pipeline.schemas.reports import RetentionPlanRecord
from evidence_pipeline.schemas.sources import SourceRecord

RETENTION_PLAN_VERSION = "retention.plan.v1"


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


def _audit_event_id(retention_id: str, output_path: Path) -> str:
    return stable_id(
        "audit",
        {
            "action": "retention_plan",
            "retention_id": retention_id,
            "output_path": str(output_path),
            "plan_version": RETENTION_PLAN_VERSION,
        },
    )


def _audit_retention_candidates(config: PipelineConfig, candidates: List[dict], output_path: Path) -> None:
    existing_audit_ids = existing_values(config.jsonl_paths()["audit_events"], "audit_event_id")
    for candidate in candidates:
        audit_event_id = _audit_event_id(str(candidate["retention_id"]), output_path)
        if audit_event_id in existing_audit_ids:
            continue
        append_jsonl(
            config.jsonl_paths()["audit_events"],
            AuditEventRecord(
                audit_event_id=audit_event_id,
                action="retention_plan",
                actor_id="system",
                target_type="retention_candidate",
                target_id=str(candidate["retention_id"]),
                source_id=candidate.get("source_id"),
                status="created",
                details={
                    "action": candidate["action"],
                    "source_modality": candidate["source_modality"],
                    "age_days": candidate["age_days"],
                    "retention_days": candidate["retention_days"],
                    "reason_code": candidate["reason_code"],
                    "dry_run": candidate["dry_run"],
                    "output_path": str(output_path),
                    "plan_version": RETENTION_PLAN_VERSION,
                },
            ),
        )
        existing_audit_ids.add(audit_event_id)


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
    _audit_retention_candidates(config, candidates, output_path)
    return RetentionPlanResult(output_path=output_path, candidate_count=len(candidates))
