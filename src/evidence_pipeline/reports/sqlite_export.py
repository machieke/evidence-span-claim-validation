from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import ensure_parent, read_jsonl

ARTIFACT_KEY_FIELDS = {
    "sources": "source_id",
    "chat_messages": "message_id",
    "pdf_blocks": "block_id",
    "audio_utterances": "utterance_id",
    "images": "image_id",
    "image_regions": "region_id",
    "image_region_embeddings": "embedding_id",
    "image_feature_clusters": "feature_cluster_id",
    "evidence": "evidence_id",
    "chunks": "chunk_id",
    "spans": "span_id",
    "claims_raw": "claim_id",
    "validations": "validation_id",
    "claims_validated": "claim_id",
    "claims_normalized": "normalized_claim_id",
    "jobs": "job_id",
    "review_decisions": "review_id",
    "audit_events": "audit_event_id",
    "errors": "error_id",
    "quarantine": "quarantine_id",
    "claim_graph": "edge_id",
    "claim_duplicates": "dedupe_id",
    "claim_repairs": "repair_id",
    "gold_eval": "evaluation_id",
    "model_routing": "routing_id",
    "pii_findings": "finding_id",
    "pii_redactions": "redaction_id",
    "privacy_policy_violations": "violation_id",
    "retention_plan": "retention_id",
    "review_queue": "review_queue_id",
}

REPORT_JSONL_FILES = {
    "claim_graph": "claim_graph.jsonl",
    "claim_duplicates": "claim_duplicates.jsonl",
    "claim_repairs": "claim_repairs.jsonl",
    "gold_eval": "gold_eval.jsonl",
    "model_routing": "model_routing.jsonl",
    "pii_findings": "pii_findings.jsonl",
    "pii_redactions": "pii_redactions.jsonl",
    "privacy_policy_violations": "privacy_policy_violations.jsonl",
    "retention_plan": "retention_plan.jsonl",
    "review_queue": "review_queue.jsonl",
}


@dataclass
class SQLiteExportResult:
    output_path: Path
    table_counts: Dict[str, int]


def _safe_table_name(name: str) -> str:
    if not name.replace("_", "").isalnum():
        raise ValueError(f"unsafe table name: {name}")
    return name


def _record_key(artifact_name: str, line_number: int, payload: dict) -> str:
    key_field = ARTIFACT_KEY_FIELDS.get(artifact_name)
    if key_field and key_field in payload:
        return str(payload[key_field])
    return f"{artifact_name}:{line_number}"


def _drop_existing_tables(connection: sqlite3.Connection, table_names: Iterable[str]) -> None:
    for table_name in table_names:
        connection.execute(f'DROP TABLE IF EXISTS "{_safe_table_name(table_name)}"')
    connection.execute("DROP TABLE IF EXISTS artifact_counts")


def _create_artifact_table(connection: sqlite3.Connection, table_name: str) -> None:
    connection.execute(
        f"""
        CREATE TABLE "{_safe_table_name(table_name)}" (
            record_key TEXT PRIMARY KEY,
            line_number INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )


def _insert_artifact_rows(connection: sqlite3.Connection, table_name: str, rows: Iterable[Tuple[int, dict]]) -> int:
    count = 0
    for line_number, payload in rows:
        connection.execute(
            f'INSERT OR REPLACE INTO "{_safe_table_name(table_name)}" '
            "(record_key, line_number, payload_json) VALUES (?, ?, ?)",
            (
                _record_key(table_name, line_number, payload),
                line_number,
                json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
            ),
        )
        count += 1
    return count


def _write_artifact_counts(connection: sqlite3.Connection, table_counts: Dict[str, int]) -> None:
    connection.execute(
        """
        CREATE TABLE artifact_counts (
            artifact_name TEXT PRIMARY KEY,
            record_count INTEGER NOT NULL
        )
        """
    )
    connection.executemany(
        "INSERT INTO artifact_counts (artifact_name, record_count) VALUES (?, ?)",
        sorted(table_counts.items()),
    )


def _export_paths(config: PipelineConfig) -> Dict[str, Path]:
    paths = dict(config.jsonl_paths())
    for artifact_name, filename in REPORT_JSONL_FILES.items():
        path = config.paths.reports_dir / filename
        if path.exists():
            paths[artifact_name] = path
    return paths


def export_sqlite(config: PipelineConfig, output_path: Optional[Path] = None) -> SQLiteExportResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "pipeline.sqlite"
    ensure_parent(output_path)

    paths = _export_paths(config)
    table_counts: Dict[str, int] = {}
    with sqlite3.connect(output_path) as connection:
        _drop_existing_tables(connection, list(paths.keys()) + list(REPORT_JSONL_FILES))
        for artifact_name, path in paths.items():
            _create_artifact_table(connection, artifact_name)
            table_counts[artifact_name] = _insert_artifact_rows(connection, artifact_name, read_jsonl(path))
        _write_artifact_counts(connection, table_counts)
        connection.commit()

    return SQLiteExportResult(output_path=output_path, table_counts=table_counts)
