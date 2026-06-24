from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from pydantic import ValidationError

from evidence_pipeline.config import PipelineConfig, load_config
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jsonl import JSONLDecodeError, append_jsonl, find_record, read_jsonl
from evidence_pipeline.schemas import SCHEMA_REGISTRY, EvidenceRecord, SourceModality, SourceRecord

app = typer.Typer(help="Evidence-span claim validation pipeline.")


CANONICAL_RAW_DIRS = [
    "chat",
    "pdf",
    "audio",
    "images",
]

CANONICAL_WORK_DIRS = [
    "normalized_audio",
    "normalized_images",
    "crops",
    "masks",
    "vectors",
]


def _parse_metadata(values: Optional[List[str]]) -> dict:
    metadata = {}
    for value in values or []:
        if "=" not in value:
            raise typer.BadParameter("metadata must use KEY=VALUE format")
        key, raw = value.split("=", 1)
        if not key:
            raise typer.BadParameter("metadata key must not be empty")
        try:
            metadata[key] = json.loads(raw)
        except json.JSONDecodeError:
            metadata[key] = raw
    return metadata


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)


def _init_paths(config: PipelineConfig) -> None:
    config.paths.data_dir.mkdir(parents=True, exist_ok=True)
    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    config.paths.work_dir.mkdir(parents=True, exist_ok=True)
    config.paths.jsonl_dir.mkdir(parents=True, exist_ok=True)
    config.paths.reports_dir.mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_RAW_DIRS:
        (config.paths.raw_dir / name).mkdir(parents=True, exist_ok=True)
    for name in CANONICAL_WORK_DIRS:
        (config.paths.work_dir / name).mkdir(parents=True, exist_ok=True)
    for path in config.jsonl_paths().values():
        _touch(path)


@app.command("init")
def init_command(
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create canonical data directories and empty JSONL artifacts."""
    config = load_config(config_path)
    _init_paths(config)
    typer.echo(f"initialized {config.paths.data_dir}")


@app.command("register-source")
def register_source(
    source_file: Path = typer.Argument(..., help="Source file to register."),
    modality: SourceModality = typer.Option(..., "--modality", help="Source modality."),
    source_uri: Optional[str] = typer.Option(None, "--source-uri", help="Optional external URI."),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata as KEY=VALUE."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Register a source file in data/jsonl/sources.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    if not source_file.exists() or not source_file.is_file():
        raise typer.BadParameter(f"source file does not exist: {source_file}")

    sha256 = sha256_file(source_file)
    source_id = stable_id("src", {"modality": modality, "sha256": sha256})
    sources_path = config.jsonl_paths()["sources"]

    existing = find_record(sources_path, "source_id", source_id)
    if existing:
        typer.echo(source_id)
        return

    record = SourceRecord(
        source_id=source_id,
        source_modality=modality,
        source_file=str(source_file),
        source_uri=source_uri,
        sha256=sha256,
        metadata=_parse_metadata(metadata),
    )
    append_jsonl(sources_path, record)
    typer.echo(source_id)


@app.command("validate-jsonl")
def validate_jsonl(
    path: Path = typer.Argument(..., help="JSONL file to validate."),
    schema: str = typer.Option(..., "--schema", help="Schema name, for example source or evidence."),
) -> None:
    """Validate every JSON object in a JSONL file against a registered schema."""
    model = SCHEMA_REGISTRY.get(schema)
    if model is None:
        valid = ", ".join(sorted(SCHEMA_REGISTRY))
        raise typer.BadParameter(f"unknown schema {schema!r}; expected one of: {valid}")
    if not path.exists():
        raise typer.BadParameter(f"path does not exist: {path}")

    count = 0
    try:
        for line_number, payload in read_jsonl(path):
            try:
                model.model_validate(payload)
            except ValidationError as exc:
                typer.echo(f"{path}:{line_number}: {exc}", err=True)
                raise typer.Exit(code=1)
            count += 1
    except JSONLDecodeError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    typer.echo(f"valid {count} records")


@app.command("validate-artifacts")
def validate_artifacts(
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Validate known JSONL artifacts that exist in the configured data directory."""
    config = load_config(config_path)
    schema_by_key = {
        "sources": "source",
        "evidence": "evidence",
        "chunks": "chunk",
        "spans": "span",
        "claims_raw": "claim.raw",
        "validations": "validation",
        "claims_validated": "claim.validated",
        "claims_normalized": "claim.normalized",
        "errors": "error",
        "quarantine": "quarantine",
    }
    failures = 0
    for key, path in config.jsonl_paths().items():
        schema = schema_by_key.get(key)
        if schema is None or not path.exists():
            continue
        model = SCHEMA_REGISTRY[schema]
        count = 0
        try:
            for line_number, payload in read_jsonl(path):
                try:
                    model.model_validate(payload)
                except ValidationError as exc:
                    typer.echo(f"{path}:{line_number}: {exc}", err=True)
                    failures += 1
                count += 1
        except JSONLDecodeError as exc:
            typer.echo(str(exc), err=True)
            failures += 1
        typer.echo(f"{path}: checked {count} records")
    if failures:
        raise typer.Exit(code=1)


@app.command("example-evidence")
def example_evidence() -> None:
    """Print a minimal valid evidence record example."""
    record = EvidenceRecord(
        evidence_id="ev_example",
        source_id="src_example",
        source_modality="chat",
        evidence_type="message_span",
        text="I saw the boat Hope yesterday.",
        provenance={
            "conversation_id": "conv_example",
            "message_id": "msg_example",
            "sender_id": "user_example",
            "char_start": 0,
            "char_end": 32,
        },
    )
    typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, sort_keys=True))
