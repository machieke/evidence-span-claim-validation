from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import ensure_parent, read_jsonl_records
from evidence_pipeline.schemas.claims import NormalizedClaimRecord

METTA_EXPORT_VERSION = "metta.claim_export.v1"


@dataclass
class MeTTaExportResult:
    output_path: Path
    claim_count: int


def _canonical_text(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _atom(value: object) -> str:
    return json.dumps(_canonical_text(value), ensure_ascii=False)


def _claim_expression(record: NormalizedClaimRecord) -> str:
    normalized = record.normalized_claim
    return " ".join(
        [
            "(claim",
            _atom(record.normalized_claim_id),
            _atom(record.claim_id),
            _atom(record.source_id),
            _atom(record.evidence_id),
            _atom(normalized.get("subject")),
            _atom(normalized.get("predicate")),
            _atom(normalized.get("object")),
            _atom(normalized.get("qualifiers") or {}),
            ")",
        ]
    )


def export_metta(config: PipelineConfig, output_path: Optional[Path] = None) -> MeTTaExportResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "claims.metta"
    paths = config.jsonl_paths()
    lines = [
        f"; schema: {METTA_EXPORT_VERSION}",
        "; (claim normalized_claim_id claim_id source_id evidence_id subject predicate object_json qualifiers_json)",
    ]
    claim_count = 0
    for _, record in read_jsonl_records(paths["claims_normalized"], NormalizedClaimRecord):
        lines.append(_claim_expression(record))
        claim_count += 1

    ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MeTTaExportResult(output_path=output_path, claim_count=claim_count)
