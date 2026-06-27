from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import ensure_parent, read_jsonl


@dataclass
class SummaryReportResult:
    output_path: Path
    record_counts: Dict[str, int]


REPORT_JSONL_FILES = {
    "claim_graph": "claim_graph.jsonl",
    "claim_duplicates": "claim_duplicates.jsonl",
    "claim_repairs": "claim_repairs.jsonl",
    "model_routing": "model_routing.jsonl",
    "pii_findings": "pii_findings.jsonl",
    "privacy_policy_violations": "privacy_policy_violations.jsonl",
    "retention_plan": "retention_plan.jsonl",
}


def _rows(path: Path) -> List[dict]:
    return [payload for _, payload in read_jsonl(path)]


def _optional_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return _rows(path)


def _count_by(rows: Iterable[dict], key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        value = row.get(key)
        if value is not None:
            counter[str(value)] += 1
    return counter


def _count_quarantine_reasons(rows: Iterable[dict]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        for reason in row.get("reason_codes", []):
            counter[str(reason)] += 1
    return counter


def _count_validation_status(rows: Iterable[dict]) -> Counter:
    return _count_by(rows, "status")


def _table(headers: Tuple[str, ...], rows: Iterable[Tuple[object, ...]]) -> List[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return lines


def _counter_table(title: str, key_header: str, counter: Counter) -> List[str]:
    lines = [f"## {title}", ""]
    if not counter:
        lines.append("_None_")
        lines.append("")
        return lines
    lines.extend(_table((key_header, "Count"), sorted(counter.items())))
    lines.append("")
    return lines


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{(numerator / denominator) * 100:.1f}%"


def _has_validation_reason(row: dict, reason_code: str) -> bool:
    return reason_code in row.get("errors", []) or reason_code in row.get("warnings", [])


def _quality_rows(
    claims_raw: List[dict],
    validations: List[dict],
    claims_validated: List[dict],
    quarantine: List[dict],
    repair_suggestions: List[dict],
) -> List[Tuple[str, object]]:
    exact_matches = 0
    text_validated = 0
    for claim in claims_validated:
        if claim.get("source_modality") == "image":
            continue
        text_validated += 1
        validation = claim.get("validation") or {}
        if validation.get("evidence_exact_match") is True:
            exact_matches += 1
    unsupported_entities = sum(
        1
        for validation in validations
        if _has_validation_reason(validation, "unsupported_entities_introduced")
    )
    return [
        ("Accepted text claim exact-evidence rate", _rate(exact_matches, text_validated)),
        ("Raw claim quarantine rate", _rate(len(quarantine), len(claims_raw))),
        ("Unsupported entity validation rate", _rate(unsupported_entities, len(validations))),
        ("Evidence repair suggestion rate", _rate(len(repair_suggestions), len(claims_raw))),
        ("Accepted claims", len(claims_validated)),
        ("Quarantined claims", len(quarantine)),
        ("Evidence repair suggestions", len(repair_suggestions)),
    ]


def render_summary_markdown(config: PipelineConfig) -> Tuple[str, Dict[str, int]]:
    paths = config.jsonl_paths()
    artifacts = {
        "sources": _rows(paths["sources"]),
        "chat_messages": _rows(paths["chat_messages"]),
        "pdf_blocks": _rows(paths["pdf_blocks"]),
        "audio_utterances": _rows(paths["audio_utterances"]),
        "images": _rows(paths["images"]),
        "image_regions": _rows(paths["image_regions"]),
        "image_region_embeddings": _rows(paths["image_region_embeddings"]),
        "image_feature_clusters": _rows(paths["image_feature_clusters"]),
        "evidence": _rows(paths["evidence"]),
        "chunks": _rows(paths["chunks"]),
        "spans": _rows(paths["spans"]),
        "claims_raw": _rows(paths["claims_raw"]),
        "validations": _rows(paths["validations"]),
        "claims_validated": _rows(paths["claims_validated"]),
        "claims_normalized": _rows(paths["claims_normalized"]),
        "jobs": _rows(paths["jobs"]),
        "review_decisions": _rows(paths["review_decisions"]),
        "audit_events": _rows(paths["audit_events"]),
        "errors": _rows(paths["errors"]),
        "quarantine": _rows(paths["quarantine"]),
    }
    for artifact_name, filename in REPORT_JSONL_FILES.items():
        artifacts[artifact_name] = _optional_rows(config.paths.reports_dir / filename)
    record_counts = {name: len(rows) for name, rows in artifacts.items()}

    lines: List[str] = [
        "# Evidence Pipeline Extraction Summary",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Artifact Counts",
        "",
    ]
    lines.extend(_table(("Artifact", "Records"), sorted(record_counts.items())))
    lines.append("")

    source_modalities = _count_by(artifacts["sources"], "source_modality")
    evidence_modalities = _count_by(artifacts["evidence"], "source_modality")
    span_modalities = _count_by(artifacts["spans"], "source_modality")
    raw_claim_modalities = _count_by(artifacts["claims_raw"], "source_modality")
    validated_claim_modalities = _count_by(artifacts["claims_validated"], "source_modality")

    lines.extend(_counter_table("Sources By Modality", "Modality", source_modalities))
    lines.extend(_counter_table("Evidence By Modality", "Modality", evidence_modalities))
    lines.extend(_counter_table("Spans By Modality", "Modality", span_modalities))
    lines.extend(_counter_table("Raw Claims By Modality", "Modality", raw_claim_modalities))
    lines.extend(_counter_table("Validated Claims By Modality", "Modality", validated_claim_modalities))
    lines.extend(_counter_table("Validation Statuses", "Status", _count_validation_status(artifacts["validations"])))
    lines.extend(_counter_table("Jobs By Stage", "Stage", _count_by(artifacts["jobs"], "stage")))
    lines.extend(_counter_table("Review Decisions", "Decision", _count_by(artifacts["review_decisions"], "decision")))
    lines.extend(_counter_table("Audit Events", "Action", _count_by(artifacts["audit_events"], "action")))
    lines.extend(_counter_table("Quarantine Reasons", "Reason", _count_quarantine_reasons(artifacts["quarantine"])))
    lines.extend(_counter_table("PII Findings By Type", "PII Type", _count_by(artifacts["pii_findings"], "pii_type")))
    lines.extend(
        _counter_table(
            "Privacy Policy Violations",
            "Reason",
            _count_by(artifacts["privacy_policy_violations"], "reason_code"),
        )
    )
    lines.extend(
        _counter_table(
            "Retention Plan Reasons",
            "Reason",
            _count_by(artifacts["retention_plan"], "reason_code"),
        )
    )

    lines.extend(["## Quality Metrics", ""])
    lines.extend(
        _table(
            ("Metric", "Value"),
            _quality_rows(
                artifacts["claims_raw"],
                artifacts["validations"],
                artifacts["claims_validated"],
                artifacts["quarantine"],
                artifacts["claim_repairs"],
            ),
        )
    )
    lines.append("")
    return "\n".join(lines), record_counts


def write_summary_report(config: PipelineConfig, output_path: Optional[Path] = None) -> SummaryReportResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "extraction_summary.md"
    markdown, record_counts = render_summary_markdown(config)
    ensure_parent(output_path)
    output_path.write_text(markdown, encoding="utf-8")
    return SummaryReportResult(output_path=output_path, record_counts=record_counts)
