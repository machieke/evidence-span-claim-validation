from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from pydantic import ValidationError

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import JSONLDecodeError, read_jsonl
from evidence_pipeline.schemas import SCHEMA_REGISTRY

ARTIFACT_SCHEMA_BY_KEY = {
    "sources": "source",
    "chat_messages": "chat_message",
    "pdf_blocks": "pdf_block",
    "audio_utterances": "audio_utterance",
    "images": "image",
    "image_regions": "image_region",
    "image_region_embeddings": "image_region_embedding",
    "image_feature_clusters": "image_feature_cluster",
    "evidence": "evidence",
    "chunks": "chunk",
    "spans": "span",
    "claims_raw": "claim.raw",
    "validations": "validation",
    "claims_validated": "claim.validated",
    "claims_normalized": "claim.normalized",
    "jobs": "job",
    "review_decisions": "review_decision",
    "audit_events": "audit_event",
    "errors": "error",
    "quarantine": "quarantine",
}

REPORT_SCHEMA_BY_KEY = {
    "claim_graph": "claim_graph",
    "claim_duplicates": "claim_duplicates",
    "claim_repairs": "claim_repairs",
    "gold_eval": "gold_eval",
    "model_routing": "model_routing",
    "pii_findings": "pii_findings",
    "pii_redactions": "pii_redactions",
    "privacy_policy_violations": "privacy_policy_violations",
    "retention_plan": "retention_plan",
    "review_queue": "review_queue",
    "acceptance_check": "acceptance_check",
}


@dataclass
class ArtifactValidationFileResult:
    path: Path
    schema: str
    records_checked: int
    errors: List[str]

    @property
    def failures(self) -> int:
        return len(self.errors)


@dataclass
class ArtifactValidationResult:
    files: List[ArtifactValidationFileResult]

    @property
    def records_checked(self) -> int:
        return sum(file.records_checked for file in self.files)

    @property
    def failures(self) -> int:
        return sum(file.failures for file in self.files)


def _validate_path(path: Path, schema: str) -> ArtifactValidationFileResult:
    model = SCHEMA_REGISTRY[schema]
    count = 0
    errors: List[str] = []
    try:
        for line_number, payload in read_jsonl(path):
            try:
                model.model_validate(payload)
            except ValidationError as exc:
                errors.append(f"{path}:{line_number}: {exc}")
            count += 1
    except JSONLDecodeError as exc:
        errors.append(str(exc))
    return ArtifactValidationFileResult(path=path, schema=schema, records_checked=count, errors=errors)


def validate_known_artifacts(config: PipelineConfig, include_reports: bool = False) -> ArtifactValidationResult:
    files: List[ArtifactValidationFileResult] = []
    for key, path in config.jsonl_paths().items():
        schema = ARTIFACT_SCHEMA_BY_KEY.get(key)
        if schema is None or not path.exists():
            continue
        files.append(_validate_path(path, schema))

    if include_reports:
        for key, schema in REPORT_SCHEMA_BY_KEY.items():
            path = config.paths.reports_dir / f"{key}.jsonl"
            if not path.exists():
                continue
            files.append(_validate_path(path, schema))

    return ArtifactValidationResult(files=files)
