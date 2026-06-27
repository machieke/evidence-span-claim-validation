from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer
from pydantic import ValidationError

from evidence_pipeline.chunking.chat_chunker import chunk_chat
from evidence_pipeline.chunking.audio_chunker import chunk_audio
from evidence_pipeline.chunking.pdf_chunker import chunk_pdf
from evidence_pipeline.config import PipelineConfig, load_config
from evidence_pipeline.extraction.claim_extractor import extract_claims_from_spans
from evidence_pipeline.ingest.chat import ingest_chat_export
from evidence_pipeline.ingest.chat_evidence import build_chat_evidence
from evidence_pipeline.ingest.audio import ingest_audio_transcript
from evidence_pipeline.ingest.audio_evidence import build_audio_evidence
from evidence_pipeline.ingest.image import ingest_images
from evidence_pipeline.ingest.image_evidence import build_image_evidence
from evidence_pipeline.ingest.pdf import ingest_pdf
from evidence_pipeline.ingest.pdf_evidence import build_pdf_evidence
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jsonl import JSONLDecodeError, append_jsonl, find_record, read_jsonl
from evidence_pipeline.normalization.claims import normalize_claims
from evidence_pipeline.normalization.graph_export import export_graph_jsonl
from evidence_pipeline.reports.summary import write_summary_report
from evidence_pipeline.reports.gold_eval import write_gold_eval_report
from evidence_pipeline.schemas import SCHEMA_REGISTRY, EvidenceRecord, SourceModality, SourceRecord
from evidence_pipeline.spans.image_region_selector import propose_image_regions
from evidence_pipeline.spans.rule_highlighter import detect_audio_spans, detect_chat_spans, detect_pdf_spans
from evidence_pipeline.validation.deterministic import validate_raw_claims

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


@app.command("ingest-chat")
def ingest_chat(
    chat_export: Path = typer.Argument(..., help="Chat export JSON file."),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata as KEY=VALUE."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Ingest a JSON chat export into sources.jsonl and chat_messages.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    if not chat_export.exists() or not chat_export.is_file():
        raise typer.BadParameter(f"chat export does not exist: {chat_export}")
    try:
        result = ingest_chat_export(chat_export, config, metadata=_parse_metadata(metadata))
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(
        f"source_id={result.source_id} source_created={result.source_created} "
        f"messages_created={result.messages_created} messages_skipped={result.messages_skipped}"
    )


@app.command("build-chat-evidence")
def build_chat_evidence_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only build evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create chat message_span evidence records from chat_messages.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_chat_evidence(config, source_id=source_id)
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("chunk-chat")
def chunk_chat_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only chunk this source."),
    previous_messages: int = typer.Option(2, "--previous-messages", min=0, help="Previous messages to include as context."),
    max_tokens: int = typer.Option(1200, "--max-tokens", min=1, help="Policy metadata token budget."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Build thread-aware chat context chunks."""
    config = load_config(config_path)
    _init_paths(config)
    result = chunk_chat(config, source_id=source_id, previous_messages=previous_messages, max_tokens=max_tokens)
    typer.echo(f"chunks_created={result.created} chunks_skipped={result.skipped}")


@app.command("detect-chat-spans")
def detect_chat_spans_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only detect spans for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Detect claim-bearing spans in primary chat evidence."""
    config = load_config(config_path)
    _init_paths(config)
    result = detect_chat_spans(config, source_id=source_id)
    typer.echo(f"spans_created={result.created} spans_skipped={result.skipped}")


@app.command("ingest-pdf")
def ingest_pdf_command(
    pdf_file: Path = typer.Argument(..., help="PDF file to ingest."),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata as KEY=VALUE."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Ingest a PDF into sources.jsonl and pdf_blocks.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    if not pdf_file.exists() or not pdf_file.is_file():
        raise typer.BadParameter(f"PDF file does not exist: {pdf_file}")
    try:
        result = ingest_pdf(pdf_file, config, metadata=_parse_metadata(metadata))
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(
        f"source_id={result.source_id} source_created={result.source_created} "
        f"blocks_created={result.blocks_created} blocks_skipped={result.blocks_skipped} "
        f"extractor={result.extractor}"
    )


@app.command("build-pdf-evidence")
def build_pdf_evidence_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only build evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create PDF text_span evidence records from pdf_blocks.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_pdf_evidence(config, source_id=source_id)
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("chunk-pdf")
def chunk_pdf_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only chunk this source."),
    target_tokens: int = typer.Option(1200, "--target-tokens", min=1, help="Approximate target token budget."),
    overlap_tokens: int = typer.Option(150, "--overlap-tokens", min=0, help="Policy metadata overlap budget."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Build PDF context chunks from text evidence."""
    config = load_config(config_path)
    _init_paths(config)
    result = chunk_pdf(config, source_id=source_id, target_tokens=target_tokens, overlap_tokens=overlap_tokens)
    typer.echo(f"chunks_created={result.created} chunks_skipped={result.skipped}")


@app.command("detect-pdf-spans")
def detect_pdf_spans_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only detect spans for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Detect claim-bearing spans in primary PDF evidence."""
    config = load_config(config_path)
    _init_paths(config)
    result = detect_pdf_spans(config, source_id=source_id)
    typer.echo(f"spans_created={result.created} spans_skipped={result.skipped}")


@app.command("ingest-audio-transcript")
def ingest_audio_transcript_command(
    transcript_file: Path = typer.Argument(..., help="Audio transcript JSON file."),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata as KEY=VALUE."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Ingest a timestamped transcript into sources.jsonl and audio_utterances.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    if not transcript_file.exists() or not transcript_file.is_file():
        raise typer.BadParameter(f"transcript file does not exist: {transcript_file}")
    try:
        result = ingest_audio_transcript(transcript_file, config, metadata=_parse_metadata(metadata))
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(
        f"source_id={result.source_id} source_created={result.source_created} "
        f"utterances_created={result.utterances_created} utterances_skipped={result.utterances_skipped}"
    )


@app.command("build-audio-evidence")
def build_audio_evidence_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only build evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create audio utterance_span evidence records from audio_utterances.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_audio_evidence(config, source_id=source_id)
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("chunk-audio")
def chunk_audio_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only chunk this source."),
    previous_utterances: int = typer.Option(1, "--previous-utterances", min=0, help="Previous utterances to include as context."),
    max_tokens: int = typer.Option(1200, "--max-tokens", min=1, help="Policy metadata token budget."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Build speaker/time-aware audio chunks."""
    config = load_config(config_path)
    _init_paths(config)
    result = chunk_audio(config, source_id=source_id, previous_utterances=previous_utterances, max_tokens=max_tokens)
    typer.echo(f"chunks_created={result.created} chunks_skipped={result.skipped}")


@app.command("detect-audio-spans")
def detect_audio_spans_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only detect spans for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Detect claim-bearing spans in primary audio evidence."""
    config = load_config(config_path)
    _init_paths(config)
    result = detect_audio_spans(config, source_id=source_id)
    typer.echo(f"spans_created={result.created} spans_skipped={result.skipped}")


@app.command("ingest-images")
def ingest_images_command(
    image_path: Path = typer.Argument(..., help="Image file or directory to ingest."),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", "-m", help="Metadata as KEY=VALUE."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Ingest image metadata into sources.jsonl and images.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    if not image_path.exists():
        raise typer.BadParameter(f"image path does not exist: {image_path}")
    result = ingest_images(image_path, config, metadata=_parse_metadata(metadata))
    typer.echo(
        f"sources_created={result.sources_created} images_created={result.images_created} "
        f"images_skipped={result.images_skipped}"
    )


@app.command("propose-image-regions")
def propose_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only propose regions for this source."),
    patch_size: int = typer.Option(224, "--patch-size", min=1, help="Grid patch size in pixels."),
    stride: int = typer.Option(112, "--stride", min=1, help="Grid stride in pixels."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Propose grid patch image regions and persist crop files."""
    config = load_config(config_path)
    _init_paths(config)
    result = propose_image_regions(config, source_id=source_id, patch_size=patch_size, stride=stride)
    typer.echo(f"regions_created={result.created} regions_skipped={result.skipped}")


@app.command("build-image-evidence")
def build_image_evidence_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only build evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create visual_region evidence records from image_regions.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_image_evidence(config, source_id=source_id)
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("validate-claims")
def validate_claims_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only validate claims for this source."),
    claim_id: Optional[List[str]] = typer.Option(None, "--claim-id", help="Only validate the selected claim ID. Repeatable."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Run deterministic validation over claims.raw.jsonl."""
    config = load_config(config_path)
    _init_paths(config)
    result = validate_raw_claims(config, source_id=source_id, claim_ids=claim_id)
    typer.echo(
        f"claims_accepted={result.accepted} claims_quarantined={result.quarantined} "
        f"claims_skipped={result.skipped}"
    )


@app.command("extract-claims")
def extract_claims_command(
    modality: str = typer.Option("all", "--modality", help="Modality to extract: all, chat, pdf, or audio."),
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only extract claims for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Extract source-faithful raw claims from detected spans using the baseline rules extractor."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = extract_claims_from_spans(config, modality=modality, source_id=source_id)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"claims_created={result.created} claims_skipped={result.skipped}")


@app.command("report")
def report_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Markdown report output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write a Markdown extraction summary report."""
    config = load_config(config_path)
    _init_paths(config)
    result = write_summary_report(config, output_path=output)
    typer.echo(str(result.output_path))


@app.command("normalize-claims")
def normalize_claims_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only normalize claims for this source."),
    claim_id: Optional[List[str]] = typer.Option(None, "--claim-id", help="Only normalize the selected claim ID. Repeatable."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Normalize accepted validated claims into derived semantic records."""
    config = load_config(config_path)
    _init_paths(config)
    result = normalize_claims(config, source_id=source_id, claim_ids=claim_id)
    typer.echo(f"claims_normalized={result.created} claims_skipped={result.skipped}")


@app.command("export-graph")
def export_graph_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Graph JSONL output path."),
    format: str = typer.Option("jsonl", "--format", help="Export format. Currently only jsonl is supported."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Export normalized claims as graph-style JSONL edges."""
    if format != "jsonl":
        raise typer.BadParameter("only jsonl graph export is currently supported")
    config = load_config(config_path)
    _init_paths(config)
    result = export_graph_jsonl(config, output_path=output)
    typer.echo(f"{result.output_path} edges={result.edge_count}")


@app.command("eval-gold")
def eval_gold_command(
    gold_file: Path = typer.Argument(..., help="Gold claims JSON file."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Markdown evaluation report output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Evaluate accepted/quarantined claims against a gold JSON file."""
    config = load_config(config_path)
    _init_paths(config)
    if not gold_file.exists() or not gold_file.is_file():
        raise typer.BadParameter(f"gold file does not exist: {gold_file}")
    try:
        result = write_gold_eval_report(config, gold_file, output_path=output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(
        f"{result.output_path} accepted_precision={result.metrics['accepted_precision']} "
        f"accepted_recall={result.metrics['accepted_recall']}"
    )


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
        "chat_messages": "chat_message",
        "pdf_blocks": "pdf_block",
        "audio_utterances": "audio_utterance",
        "images": "image",
        "image_regions": "image_region",
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
