from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.sources import SourceRecord


@dataclass
class PrivacyCheckResult:
    output_path: Path
    claims_checked: int
    violation_count: int


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "sensitive", "local_only"}
    return False


def _sensitive_keys(source: SourceRecord, configured_keys: List[str]) -> List[str]:
    return sorted(key for key in configured_keys if _truthy(source.metadata.get(key)))


def _sensitive_source_keys(config: PipelineConfig) -> Dict[str, List[str]]:
    paths = config.jsonl_paths()
    sensitive_by_source_id: Dict[str, List[str]] = {}
    if not config.privacy.local_only_sensitive_sources:
        return sensitive_by_source_id
    for _, source in read_jsonl_records(paths["sources"], SourceRecord):
        keys = _sensitive_keys(source, config.privacy.sensitive_metadata_keys)
        if keys:
            sensitive_by_source_id[source.source_id] = keys
    return sensitive_by_source_id


def _local_providers(config: PipelineConfig) -> Set[str]:
    return {provider.strip().lower() for provider in config.privacy.local_model_providers if provider.strip()}


def _violation(claim: RawClaimRecord, sensitive_keys: List[str]) -> dict:
    provider = claim.model.provider or "unknown"
    model_name = claim.model.model or "unknown"
    violation_id = stable_id(
        "privacy",
        {
            "claim_id": claim.claim_id,
            "source_id": claim.source_id,
            "provider": provider,
            "model": model_name,
            "policy": "local_only_sensitive_sources",
        },
    )
    return {
        "violation_id": violation_id,
        "source_id": claim.source_id,
        "claim_id": claim.claim_id,
        "evidence_id": claim.evidence_id,
        "provider": provider,
        "model": model_name,
        "policy": "local_only_sensitive_sources",
        "reason_code": "non_local_provider_for_sensitive_source",
        "sensitive_metadata_keys": sensitive_keys,
        "schema_version": "privacy.violation.v1",
    }


def check_privacy_policy(config: PipelineConfig, output_path: Optional[Path] = None) -> PrivacyCheckResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "privacy_policy_violations.jsonl"

    sensitive_by_source_id = _sensitive_source_keys(config)
    local_providers = _local_providers(config)
    claims_checked = 0
    violations = []
    seen_violation_ids = set()

    for _, claim in read_jsonl_records(config.jsonl_paths()["claims_raw"], RawClaimRecord):
        claims_checked += 1
        sensitive_keys = sensitive_by_source_id.get(claim.source_id)
        if not sensitive_keys:
            continue
        provider = (claim.model.provider or "unknown").strip().lower()
        if provider in local_providers:
            continue
        violation = _violation(claim, sensitive_keys)
        if violation["violation_id"] in seen_violation_ids:
            continue
        seen_violation_ids.add(violation["violation_id"])
        violations.append(violation)

    write_jsonl(output_path, violations)
    return PrivacyCheckResult(
        output_path=output_path,
        claims_checked=claims_checked,
        violation_count=len(violations),
    )
