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


def _numeric_atom(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):g}"
    return _atom(value)


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


def _claim_provenance_expressions(record: NormalizedClaimRecord) -> list[str]:
    normalized = record.normalized_claim
    raw_qualifiers = normalized.get("qualifiers") or {}
    qualifiers = raw_qualifiers if isinstance(raw_qualifiers, dict) else {}
    expressions = []
    for relation, key in (
        ("claim-modality", "modality"),
        ("claim-truth-status", "truth_status"),
        ("claim-attribution", "attribution"),
        ("claim-source-faithful", "source_faithful_claim"),
    ):
        value = qualifiers.get(key)
        if value is None:
            continue
        expressions.append(" ".join([f"({relation}", _atom(record.normalized_claim_id), _atom(value), ")"]))
    confidence = qualifiers.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        expressions.append(
            " ".join(
                [
                    "(claim-confidence",
                    _atom(record.normalized_claim_id),
                    _numeric_atom(confidence),
                    ")",
                ]
            )
        )
    confidence_basis = qualifiers.get("confidence_basis")
    if confidence_basis is not None:
        expressions.append(
            " ".join(
                [
                    "(claim-confidence-basis",
                    _atom(record.normalized_claim_id),
                    _atom(confidence_basis),
                    ")",
                ]
            )
        )
    return expressions


def export_metta(config: PipelineConfig, output_path: Optional[Path] = None) -> MeTTaExportResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "claims.metta"
    paths = config.jsonl_paths()
    lines = [
        f"; schema: {METTA_EXPORT_VERSION}",
        "; (claim normalized_claim_id claim_id source_id evidence_id subject predicate object_json qualifiers_json)",
        "; (claim-modality normalized_claim_id modality)",
        "; (claim-truth-status normalized_claim_id truth_status)",
        "; (claim-attribution normalized_claim_id attribution_json)",
        "; (claim-source-faithful normalized_claim_id source_faithful_claim)",
        "; (claim-confidence normalized_claim_id confidence)",
        "; (claim-confidence-basis normalized_claim_id basis)",
    ]
    claim_count = 0
    for _, record in read_jsonl_records(paths["claims_normalized"], NormalizedClaimRecord):
        lines.append(_claim_expression(record))
        lines.extend(_claim_provenance_expressions(record))
        claim_count += 1

    ensure_parent(output_path)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return MeTTaExportResult(output_path=output_path, claim_count=claim_count)
