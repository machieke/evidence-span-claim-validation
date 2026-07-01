from __future__ import annotations

import html
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
    "gold_eval": "gold_eval.jsonl",
    "model_routing": "model_routing.jsonl",
    "pii_findings": "pii_findings.jsonl",
    "pii_redactions": "pii_redactions.jsonl",
    "privacy_policy_violations": "privacy_policy_violations.jsonl",
    "retention_plan": "retention_plan.jsonl",
    "review_queue": "review_queue.jsonl",
    "acceptance_check": "acceptance_check.jsonl",
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


def _count_list_values(rows: Iterable[dict], key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        for value in row.get(key, []):
            counter[str(value)] += 1
    return counter


def _count_nested_key(rows: Iterable[dict], container_key: str, key: str) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        container = row.get(container_key) or {}
        if not isinstance(container, dict):
            continue
        value = container.get(key)
        if value is not None:
            counter[str(value)] += 1
    return counter


def _count_entity_resolution_bases(rows: Iterable[dict]) -> Counter:
    counter: Counter = Counter()
    for row in rows:
        normalization = row.get("normalization") or {}
        if not isinstance(normalization, dict):
            continue
        for resolution in normalization.get("entity_resolution", []):
            if not isinstance(resolution, dict):
                continue
            basis = resolution.get("basis")
            if basis is not None:
                counter[str(basis)] += 1
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


def _markdown_table_cells(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _markdown_to_html(markdown: str) -> str:
    body: List[str] = []
    lines = markdown.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line:
            index += 1
            continue
        if line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
            index += 1
            continue
        if line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
            index += 1
            continue
        if line == "_None_":
            body.append("<p><em>None</em></p>")
            index += 1
            continue
        if line.startswith("| "):
            headers = _markdown_table_cells(line)
            index += 2
            rows = []
            while index < len(lines) and lines[index].startswith("| "):
                rows.append(_markdown_table_cells(lines[index]))
                index += 1
            body.append("<table>")
            body.append(
                "<thead><tr>"
                + "".join(f"<th>{html.escape(header)}</th>" for header in headers)
                + "</tr></thead>"
            )
            body.append("<tbody>")
            for row in rows:
                body.append(
                    "<tr>"
                    + "".join(f"<td>{html.escape(cell)}</td>" for cell in row)
                    + "</tr>"
                )
            body.append("</tbody></table>")
            continue
        body.append(f"<p>{html.escape(line)}</p>")
        index += 1

    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            "<title>Evidence Pipeline Extraction Summary</title>",
            "<style>",
            "body{font-family:Arial,sans-serif;line-height:1.45;margin:2rem;max-width:1100px;}",
            "table{border-collapse:collapse;margin:1rem 0;width:100%;}",
            "th,td{border:1px solid #d0d7de;padding:0.4rem 0.6rem;text-align:left;}",
            "th{background:#f6f8fa;}",
            "</style>",
            "</head>",
            "<body>",
            *body,
            "</body>",
            "</html>",
            "",
        ]
    )


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


def _format_optional_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _has_validation_reason(row: dict, reason_code: str) -> bool:
    return reason_code in row.get("errors", []) or reason_code in row.get("warnings", [])


def _validation_flag_rate(rows: Iterable[dict], flag: str) -> str:
    total = 0
    preserved = 0
    for row in rows:
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            continue
        validation = metadata.get("validation") or {}
        if not isinstance(validation, dict):
            continue
        value = validation.get(flag)
        if not isinstance(value, bool):
            continue
        total += 1
        if value:
            preserved += 1
    return _rate(preserved, total)


def _duplicate_claim_rate(claims_normalized: Iterable[dict], claim_duplicates: Iterable[dict]) -> str:
    normalized_claims = list(claims_normalized)
    duplicated_claim_ids = set()
    for group in claim_duplicates:
        member_claim_ids = group.get("member_claim_ids", [])
        if not isinstance(member_claim_ids, list):
            continue
        duplicated_claim_ids.update(str(value) for value in member_claim_ids)
    return _rate(len(duplicated_claim_ids), len(normalized_claims))


def _normalized_confidence_rate(claims_validated: Iterable[dict], claims_normalized: Iterable[dict]) -> str:
    accepted_confidence = {
        str(claim.get("claim_id")): claim.get("confidence")
        for claim in claims_validated
        if claim.get("support_status") == "accepted_extracted" and claim.get("confidence") is not None
    }
    if not accepted_confidence:
        return _rate(0, 0)

    normalized_by_claim_id: Dict[str, List[dict]] = {}
    for normalized in claims_normalized:
        claim_id = normalized.get("claim_id")
        if claim_id is None:
            continue
        normalized_by_claim_id.setdefault(str(claim_id), []).append(normalized)

    preserved = 0
    for claim_id, confidence in accepted_confidence.items():
        for normalized in normalized_by_claim_id.get(claim_id, []):
            normalized_claim = normalized.get("normalized_claim") or {}
            qualifiers = (
                normalized_claim.get("qualifiers")
                if isinstance(normalized_claim, dict)
                else None
            )
            if isinstance(qualifiers, dict) and qualifiers.get("confidence") == confidence:
                preserved += 1
                break
    return _rate(preserved, len(accepted_confidence))


def _numeric_value(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _numeric_metric_total(jobs: Iterable[dict], metric_names: Tuple[str, ...]) -> Optional[float]:
    total = 0.0
    found = False
    for job in jobs:
        metrics = job.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        for metric_name in metric_names:
            value = _numeric_value(metrics.get(metric_name))
            if value is None:
                continue
            total += value
            found = True
    if not found:
        return None
    return total


def _metric_count(metrics: dict, key: str) -> int:
    value = _numeric_value(metrics.get(key))
    if value is None:
        return 0
    return max(0, int(value))


def _repair_application_success_rate(jobs: Iterable[dict]) -> str:
    applied = 0
    failed = 0
    for job in jobs:
        if job.get("stage") != "apply_repairs":
            continue
        metrics = job.get("metrics") or {}
        if not isinstance(metrics, dict):
            continue
        applied += _metric_count(metrics, "repairs_applied")
        failed += _metric_count(metrics, "repairs_failed")
    return _rate(applied, applied + failed)


def _cost_per_accepted_claim(jobs: Iterable[dict], claims_validated: Iterable[dict]) -> str:
    accepted_claims = list(claims_validated)
    total_cost = _numeric_metric_total(jobs, ("cost_usd", "model_cost_usd", "total_cost_usd"))
    if total_cost is None or not accepted_claims:
        return "n/a"
    return f"${total_cost / len(accepted_claims):.4f}"


def _latency_per_source(jobs: Iterable[dict]) -> str:
    job_records = list(jobs)
    total_latency = _numeric_metric_total(
        job_records,
        ("latency_seconds", "duration_seconds", "elapsed_seconds", "runtime_seconds"),
    )
    source_ids = {str(job.get("source_id")) for job in job_records if job.get("source_id")}
    if total_latency is None or not source_ids:
        return "n/a"
    return f"{total_latency / len(source_ids):.2f}s"


def _review_disagreement_rate(rows: Iterable[dict]) -> str:
    latest_by_claim_and_reviewer: Dict[str, Dict[str, Tuple[str, str]]] = {}
    for row in rows:
        claim_id = row.get("claim_id")
        reviewer_id = row.get("reviewer_id")
        decision = row.get("decision")
        if not isinstance(claim_id, str) or not isinstance(reviewer_id, str) or not isinstance(decision, str):
            continue
        reviewed_at = str(row.get("reviewed_at") or "")
        reviewer_decisions = latest_by_claim_and_reviewer.setdefault(claim_id, {})
        previous = reviewer_decisions.get(reviewer_id)
        if previous is None or reviewed_at >= previous[0]:
            reviewer_decisions[reviewer_id] = (reviewed_at, decision)

    multi_reviewer_claims = [
        reviewer_decisions
        for reviewer_decisions in latest_by_claim_and_reviewer.values()
        if len(reviewer_decisions) >= 2
    ]
    disagreements = sum(
        1
        for reviewer_decisions in multi_reviewer_claims
        if len({decision for _, decision in reviewer_decisions.values()}) > 1
    )
    return _rate(disagreements, len(multi_reviewer_claims))


def _quality_rows(
    claims_raw: List[dict],
    validations: List[dict],
    claims_validated: List[dict],
    claims_normalized: List[dict],
    quarantine: List[dict],
    repair_suggestions: List[dict],
    claim_duplicates: List[dict],
    gold_evaluations: List[dict],
    review_decisions: List[dict],
    jobs: List[dict],
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
    rows = [
        ("Accepted text claim exact-evidence rate", _rate(exact_matches, text_validated)),
        ("Raw claim quarantine rate", _rate(len(quarantine), len(claims_raw))),
        ("Unsupported entity validation rate", _rate(unsupported_entities, len(validations))),
        ("Negation preservation rate", _validation_flag_rate(validations, "negation_preserved")),
        ("Uncertainty preservation rate", _validation_flag_rate(validations, "uncertainty_preserved")),
        ("Attribution preservation rate", _validation_flag_rate(validations, "attribution_preserved")),
        ("Quantity preservation rate", _validation_flag_rate(validations, "quantities_preserved")),
        ("Normalized confidence propagation rate", _normalized_confidence_rate(claims_validated, claims_normalized)),
        ("Duplicate normalized claim rate", _duplicate_claim_rate(claims_normalized, claim_duplicates)),
        ("Review disagreement rate", _review_disagreement_rate(review_decisions)),
        ("Evidence repair suggestion rate", _rate(len(repair_suggestions), len(claims_raw))),
        ("Repair application success rate", _repair_application_success_rate(jobs)),
        ("Cost per accepted claim", _cost_per_accepted_claim(jobs, claims_validated)),
        ("Latency per source", _latency_per_source(jobs)),
        ("Accepted claims", len(claims_validated)),
        ("Quarantined claims", len(quarantine)),
        ("Evidence repair suggestions", len(repair_suggestions)),
    ]
    if gold_evaluations:
        latest_gold = gold_evaluations[-1]
        rows.extend(
            [
                (
                    "Gold accepted precision",
                    _format_optional_rate(latest_gold.get("accepted_precision")),
                ),
                ("Gold accepted recall", _format_optional_rate(latest_gold.get("accepted_recall"))),
                (
                    "Gold quarantine precision",
                    _format_optional_rate(latest_gold.get("quarantine_precision")),
                ),
                ("Gold quarantine recall", _format_optional_rate(latest_gold.get("quarantine_recall"))),
                (
                    "Gold evidence exact-match rate",
                    _format_optional_rate(latest_gold.get("evidence_exact_match_rate")),
                ),
                (
                    "Gold attribution preservation rate",
                    _format_optional_rate(latest_gold.get("attribution_preservation_rate")),
                ),
                (
                    "Gold uncertainty preservation rate",
                    _format_optional_rate(latest_gold.get("uncertainty_preservation_rate")),
                ),
                (
                    "Gold negation preservation rate",
                    _format_optional_rate(latest_gold.get("negation_preservation_rate")),
                ),
                (
                    "Gold quantity preservation rate",
                    _format_optional_rate(latest_gold.get("quantity_preservation_rate")),
                ),
                (
                    "Gold unsupported entity rate",
                    _format_optional_rate(latest_gold.get("unsupported_entity_rate")),
                ),
                ("Gold validation quarantine rate", _format_optional_rate(latest_gold.get("quarantine_rate"))),
            ]
        )
    return rows


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
    lines.extend(
        _counter_table(
            "Normalized Claims By Predicate",
            "Predicate",
            _count_nested_key(artifacts["claims_normalized"], "normalized_claim", "predicate"),
        )
    )
    lines.extend(
        _counter_table(
            "Entity Resolution Bases",
            "Basis",
            _count_entity_resolution_bases(artifacts["claims_normalized"]),
        )
    )
    lines.extend(
        _counter_table(
            "Duplicate Groups By Level",
            "Level",
            _count_by(artifacts["claim_duplicates"], "duplicate_level"),
        )
    )
    lines.extend(_counter_table("Validation Statuses", "Status", _count_validation_status(artifacts["validations"])))
    lines.extend(
        _counter_table(
            "Validation Errors",
            "Error",
            _count_list_values(artifacts["validations"], "errors"),
        )
    )
    lines.extend(
        _counter_table(
            "Validation Warnings",
            "Warning",
            _count_list_values(artifacts["validations"], "warnings"),
        )
    )
    lines.extend(_counter_table("Jobs By Stage", "Stage", _count_by(artifacts["jobs"], "stage")))
    lines.extend(
        _counter_table(
            "Model Routing By Tier",
            "Tier",
            _count_by(artifacts["model_routing"], "selected_tier"),
        )
    )
    lines.extend(
        _counter_table(
            "Model Routing By Role",
            "Role",
            _count_by(artifacts["model_routing"], "model_role"),
        )
    )
    lines.extend(_counter_table("Review Decisions", "Decision", _count_by(artifacts["review_decisions"], "decision")))
    lines.extend(_counter_table("Review Queue By State", "State", _count_by(artifacts["review_queue"], "review_state")))
    lines.extend(_counter_table("Acceptance Checks", "Status", _count_by(artifacts["acceptance_check"], "status")))
    lines.extend(
        _counter_table(
            "Review Queue Reasons",
            "Reason",
            _count_list_values(artifacts["review_queue"], "reason_codes"),
        )
    )
    lines.extend(
        _counter_table(
            "Review Queue Warnings",
            "Warning",
            _count_list_values(artifacts["review_queue"], "warnings"),
        )
    )
    lines.extend(
        _counter_table(
            "Review Queue Risk Flags",
            "Risk Flag",
            _count_list_values(artifacts["review_queue"], "risk_flags"),
        )
    )
    lines.extend(_counter_table("Audit Events", "Action", _count_by(artifacts["audit_events"], "action")))
    lines.extend(_counter_table("Quarantine Reasons", "Reason", _count_quarantine_reasons(artifacts["quarantine"])))
    lines.extend(_counter_table("PII Findings By Type", "PII Type", _count_by(artifacts["pii_findings"], "pii_type")))
    lines.extend(
        _counter_table(
            "PII Redactions By Artifact",
            "Artifact",
            _count_by(artifacts["pii_redactions"], "artifact"),
        )
    )
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
                artifacts["claims_normalized"],
                artifacts["quarantine"],
                artifacts["claim_repairs"],
                artifacts["claim_duplicates"],
                artifacts["gold_eval"],
                artifacts["review_decisions"],
                artifacts["jobs"],
            ),
        )
    )
    lines.append("")
    return "\n".join(lines), record_counts


def render_summary_html(config: PipelineConfig) -> Tuple[str, Dict[str, int]]:
    markdown, record_counts = render_summary_markdown(config)
    return _markdown_to_html(markdown), record_counts


def _normalize_report_format(output_format: str) -> str:
    normalized = output_format.strip().lower()
    if normalized in {"markdown", "md"}:
        return "markdown"
    if normalized == "html":
        return "html"
    raise ValueError("report format must be markdown or html")


def write_summary_report(
    config: PipelineConfig,
    output_path: Optional[Path] = None,
    output_format: str = "markdown",
) -> SummaryReportResult:
    normalized_format = _normalize_report_format(output_format)
    if output_path is None:
        filename = "extraction_summary.html" if normalized_format == "html" else "extraction_summary.md"
        output_path = config.paths.reports_dir / filename
    if normalized_format == "html":
        rendered, record_counts = render_summary_html(config)
    else:
        rendered, record_counts = render_summary_markdown(config)
    ensure_parent(output_path)
    output_path.write_text(rendered, encoding="utf-8")
    return SummaryReportResult(output_path=output_path, record_counts=record_counts)
