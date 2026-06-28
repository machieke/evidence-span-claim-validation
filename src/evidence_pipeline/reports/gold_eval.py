from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import ensure_parent, read_jsonl, write_jsonl
from evidence_pipeline.schemas.reports import GoldEvaluationRecord

GOLD_EVAL_VERSION = "gold.eval.v1"

GoldKey = Tuple[str, str]


@dataclass
class GoldEvaluationResult:
    output_path: Path
    metrics_path: Path
    metrics: Dict[str, object]


def _load_gold_claims(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        claims = payload
    elif isinstance(payload, dict) and isinstance(payload.get("claims"), list):
        claims = payload["claims"]
    else:
        raise ValueError("gold file must be a JSON list or an object with a claims list")
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError("every gold claim must be a JSON object")
        if not claim.get("evidence_id"):
            raise ValueError("every gold claim requires evidence_id")
        if not claim.get("evidence_text"):
            raise ValueError("every gold claim requires evidence_text")
    return claims


def _key(record: dict) -> GoldKey:
    return str(record.get("evidence_id")), str(record.get("evidence_text"))


def _gold_keys(claims: Iterable[dict], expected_status: str) -> Set[GoldKey]:
    return {
        _key(claim)
        for claim in claims
        if str(claim.get("expected_status", "accepted")) == expected_status
    }


def _accepted_keys(config: PipelineConfig) -> Set[GoldKey]:
    return {
        _key(payload)
        for _, payload in read_jsonl(config.jsonl_paths()["claims_validated"])
        if payload.get("support_status") == "accepted_extracted"
    }


def _quarantined_keys(config: PipelineConfig) -> Set[GoldKey]:
    keys: Set[GoldKey] = set()
    for _, payload in read_jsonl(config.jsonl_paths()["quarantine"]):
        claim_payload = payload.get("payload") or {}
        if claim_payload.get("evidence_id") and claim_payload.get("evidence_text"):
            keys.add(_key(claim_payload))
    return keys


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator


def _validation_summary(row: dict) -> Optional[dict]:
    metadata = row.get("metadata") or {}
    if not isinstance(metadata, dict):
        return None
    validation = metadata.get("validation") or {}
    if not isinstance(validation, dict):
        return None
    return validation


def _validation_bool_rate(rows: Iterable[dict], field_name: str) -> Optional[float]:
    total = 0
    passed = 0
    for row in rows:
        validation = _validation_summary(row)
        if validation is None:
            continue
        value = validation.get(field_name)
        if not isinstance(value, bool):
            continue
        total += 1
        if value:
            passed += 1
    return _rate(passed, total)


def _unsupported_entity_rate(rows: Iterable[dict]) -> Optional[float]:
    total = 0
    flagged = 0
    for row in rows:
        validation = _validation_summary(row)
        if validation is None:
            continue
        introduced_entities = validation.get("introduced_entities")
        if not isinstance(introduced_entities, list):
            continue
        total += 1
        if introduced_entities:
            flagged += 1
    return _rate(flagged, total)


def _validation_quarantine_rate(rows: Iterable[dict]) -> Optional[float]:
    records = list(rows)
    return _rate(sum(1 for row in records if row.get("status") == "quarantined"), len(records))


def _validation_quality_metrics(config: PipelineConfig) -> Dict[str, Optional[float]]:
    validations = [payload for _, payload in read_jsonl(config.jsonl_paths()["validations"])]
    return {
        "evidence_exact_match_rate": _validation_bool_rate(validations, "evidence_exact_match"),
        "attribution_preservation_rate": _validation_bool_rate(validations, "attribution_preserved"),
        "uncertainty_preservation_rate": _validation_bool_rate(validations, "uncertainty_preserved"),
        "negation_preservation_rate": _validation_bool_rate(validations, "negation_preserved"),
        "quantity_preservation_rate": _validation_bool_rate(validations, "quantities_preserved"),
        "unsupported_entity_rate": _unsupported_entity_rate(validations),
        "quarantine_rate": _validation_quarantine_rate(validations),
    }


def evaluate_gold(config: PipelineConfig, gold_path: Path) -> Dict[str, object]:
    gold_claims = _load_gold_claims(gold_path)
    expected_accepted = _gold_keys(gold_claims, "accepted")
    expected_quarantined = _gold_keys(gold_claims, "quarantined")
    accepted = _accepted_keys(config)
    quarantined = _quarantined_keys(config)

    accepted_matches = accepted & expected_accepted
    accepted_false_positives = accepted - expected_accepted
    accepted_missing = expected_accepted - accepted
    quarantine_matches = quarantined & expected_quarantined
    quarantine_false_positives = quarantined - expected_quarantined
    quarantine_missing = expected_quarantined - quarantined

    metrics = {
        "gold_claims": len(gold_claims),
        "expected_accepted": len(expected_accepted),
        "produced_accepted": len(accepted),
        "accepted_matches": len(accepted_matches),
        "accepted_false_positives": len(accepted_false_positives),
        "accepted_missing": len(accepted_missing),
        "accepted_precision": _rate(len(accepted_matches), len(accepted)),
        "accepted_recall": _rate(len(accepted_matches), len(expected_accepted)),
        "expected_quarantined": len(expected_quarantined),
        "produced_quarantined": len(quarantined),
        "quarantine_matches": len(quarantine_matches),
        "quarantine_false_positives": len(quarantine_false_positives),
        "quarantine_missing": len(quarantine_missing),
        "quarantine_precision": _rate(len(quarantine_matches), len(quarantined)),
        "quarantine_recall": _rate(len(quarantine_matches), len(expected_quarantined)),
        "missing_keys": sorted([{"evidence_id": key[0], "evidence_text": key[1]} for key in accepted_missing], key=str),
        "false_positive_keys": sorted([{"evidence_id": key[0], "evidence_text": key[1]} for key in accepted_false_positives], key=str),
        "missing_quarantine_keys": sorted(
            [{"evidence_id": key[0], "evidence_text": key[1]} for key in quarantine_missing],
            key=str,
        ),
        "false_positive_quarantine_keys": sorted(
            [{"evidence_id": key[0], "evidence_text": key[1]} for key in quarantine_false_positives],
            key=str,
        ),
    }
    metrics.update(_validation_quality_metrics(config))
    return metrics


def _format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _render_markdown(metrics: Dict[str, object], gold_path: Path) -> str:
    lines = [
        "# Gold Evaluation Report",
        "",
        f"Gold file: {gold_path}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Gold claims | {metrics['gold_claims']} |",
        f"| Expected accepted | {metrics['expected_accepted']} |",
        f"| Produced accepted | {metrics['produced_accepted']} |",
        f"| Accepted matches | {metrics['accepted_matches']} |",
        f"| Accepted precision | {_format_rate(metrics['accepted_precision'])} |",
        f"| Accepted recall | {_format_rate(metrics['accepted_recall'])} |",
        f"| Accepted false positives | {metrics['accepted_false_positives']} |",
        f"| Accepted missing | {metrics['accepted_missing']} |",
        f"| Expected quarantined | {metrics['expected_quarantined']} |",
        f"| Produced quarantined | {metrics['produced_quarantined']} |",
        f"| Quarantine matches | {metrics['quarantine_matches']} |",
        f"| Quarantine precision | {_format_rate(metrics['quarantine_precision'])} |",
        f"| Quarantine recall | {_format_rate(metrics['quarantine_recall'])} |",
        f"| Quarantine false positives | {metrics['quarantine_false_positives']} |",
        f"| Quarantine missing | {metrics['quarantine_missing']} |",
        f"| Evidence exact-match rate | {_format_rate(metrics['evidence_exact_match_rate'])} |",
        f"| Attribution preservation rate | {_format_rate(metrics['attribution_preservation_rate'])} |",
        f"| Uncertainty preservation rate | {_format_rate(metrics['uncertainty_preservation_rate'])} |",
        f"| Negation preservation rate | {_format_rate(metrics['negation_preservation_rate'])} |",
        f"| Quantity preservation rate | {_format_rate(metrics['quantity_preservation_rate'])} |",
        f"| Unsupported entity rate | {_format_rate(metrics['unsupported_entity_rate'])} |",
        f"| Validation quarantine rate | {_format_rate(metrics['quarantine_rate'])} |",
        "",
    ]
    for title, key in (
        ("Missing Expected Accepted Claims", "missing_keys"),
        ("Accepted False Positives", "false_positive_keys"),
        ("Missing Expected Quarantined Claims", "missing_quarantine_keys"),
        ("Quarantine False Positives", "false_positive_quarantine_keys"),
    ):
        lines.extend([f"## {title}", ""])
        items = metrics[key]
        if not items:
            lines.extend(["_None_", ""])
            continue
        for item in items:
            lines.append(f"- `{item['evidence_id']}`: {item['evidence_text']}")
        lines.append("")
    return "\n".join(lines)


def _metrics_record(metrics: Dict[str, object], gold_path: Path) -> Dict[str, object]:
    return GoldEvaluationRecord(
        evaluation_id=stable_id(
            "gold_eval",
            {"gold_path": str(gold_path), "metrics": metrics},
        ),
        gold_path=str(gold_path),
        **metrics,
    ).model_dump(mode="json")


def write_gold_eval_report(
    config: PipelineConfig,
    gold_path: Path,
    output_path: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
) -> GoldEvaluationResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "gold_eval.md"
    if metrics_path is None:
        metrics_path = config.paths.reports_dir / "gold_eval.jsonl"
    metrics = evaluate_gold(config, gold_path)
    ensure_parent(output_path)
    output_path.write_text(_render_markdown(metrics, gold_path), encoding="utf-8")
    write_jsonl(metrics_path, [_metrics_record(metrics, gold_path)])
    return GoldEvaluationResult(output_path=output_path, metrics_path=metrics_path, metrics=metrics)
