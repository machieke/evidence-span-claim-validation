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
from evidence_pipeline.extraction.claim_extractor import (
    IMAGE_CLUSTER_EXTRACTOR_VERSION,
    IMAGE_REGION_EXTRACTOR_VERSION,
    RULE_EXTRACTOR_VERSION,
    extract_claims_from_spans,
)
from evidence_pipeline.extraction.image_classifier import classify_image_regions
from evidence_pipeline.ingest.chat import ingest_chat_export
from evidence_pipeline.ingest.chat_evidence import build_chat_evidence
from evidence_pipeline.ingest.audio import ingest_audio_transcript
from evidence_pipeline.ingest.audio_evidence import build_audio_evidence
from evidence_pipeline.ingest.image import ingest_images
from evidence_pipeline.ingest.image_evidence import build_image_cluster_evidence, build_image_evidence
from evidence_pipeline.ingest.pdf import ingest_pdf
from evidence_pipeline.ingest.pdf_evidence import build_pdf_evidence
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jobs import record_job_result
from evidence_pipeline.jsonl import JSONLDecodeError, append_jsonl, find_record, read_jsonl
from evidence_pipeline.model_routing import write_model_routing_report
from evidence_pipeline.normalization.claims import NORMALIZER_VERSION, normalize_claims
from evidence_pipeline.normalization.dedupe import dedupe_normalized_claims
from evidence_pipeline.normalization.graph_export import export_graph_jsonl
from evidence_pipeline.normalization.metta_export import export_metta
from evidence_pipeline.reports.summary import write_summary_report
from evidence_pipeline.reports.gold_eval import write_gold_eval_report
from evidence_pipeline.reports.lineage import trace_claim, write_claim_trace
from evidence_pipeline.reports.sqlite_export import export_sqlite
from evidence_pipeline.schemas import SCHEMA_REGISTRY, EvidenceRecord, SourceModality, SourceRecord
from evidence_pipeline.spans.image_region_clusterer import (
    build_image_region_embeddings,
    cluster_image_regions,
)
from evidence_pipeline.spans.image_region_selector import propose_image_regions
from evidence_pipeline.spans.rule_highlighter import detect_audio_spans, detect_chat_spans, detect_pdf_spans
from evidence_pipeline.validation.deterministic import VALIDATOR_VERSION, validate_raw_claims
from evidence_pipeline.validation.pii import detect_pii, redact_pii
from evidence_pipeline.validation.privacy import check_privacy_policy
from evidence_pipeline.validation.repair import suggest_evidence_repairs
from evidence_pipeline.validation.review import record_claim_review

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

EVIDENCE_MODALITIES = {"all", "chat", "pdf", "audio", "image"}
TEXT_CHUNK_MODALITIES = {"all", "chat", "pdf", "audio"}
TEXT_SPAN_MODALITIES = {"all", "chat", "pdf", "audio"}


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


def _ensure_modality(modality: str, allowed: set, command_name: str) -> None:
    if modality not in allowed:
        expected = ", ".join(sorted(allowed))
        raise typer.BadParameter(f"{command_name} supports: {expected}")


def _stage_input_ids(
    default_id: str,
    source_id: Optional[str] = None,
    record_ids: Optional[List[str]] = None,
) -> List[str]:
    if record_ids:
        return record_ids
    if source_id:
        return [source_id]
    return [default_id]


def _extract_model_id(modality: str) -> str:
    if modality == "image":
        return f"{IMAGE_REGION_EXTRACTOR_VERSION}+{IMAGE_CLUSTER_EXTRACTOR_VERSION}"
    return RULE_EXTRACTOR_VERSION


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


@app.command("embed-image-regions")
def embed_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only embed regions for this source."),
    embedding_model: str = typer.Option(
        "color_rgb_mean_std_v1",
        "--embedding-model",
        help="Embedding model identifier.",
    ),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create deterministic image-region embedding records."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_image_region_embeddings(config, source_id=source_id, embedding_model=embedding_model)
    typer.echo(f"embeddings_created={result.created} embeddings_skipped={result.skipped}")


@app.command("classify-image-regions")
def classify_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only classify regions for this source."),
    embedding_model: str = typer.Option(
        "color_rgb_mean_std_v1",
        "--embedding-model",
        help="Embedding model identifier.",
    ),
    classifier_model: str = typer.Option(
        "dominant_color_classifier_v1",
        "--classifier-model",
        help="Classifier model identifier.",
    ),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Emit named visual classification claims from image-region embeddings."""
    config = load_config(config_path)
    _init_paths(config)
    result = classify_image_regions(
        config,
        source_id=source_id,
        embedding_model=embedding_model,
        classifier_model=classifier_model,
    )
    typer.echo(f"classifications_created={result.created} classifications_skipped={result.skipped}")


@app.command("cluster-image-regions")
def cluster_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only cluster regions for this source."),
    embedding_model: str = typer.Option(
        "color_rgb_mean_std_v1",
        "--embedding-model",
        help="Embedding model identifier.",
    ),
    distance_threshold: float = typer.Option(
        0.08,
        "--distance-threshold",
        min=0.0,
        help="Maximum embedding distance for connected regions.",
    ),
    min_cluster_size: int = typer.Option(2, "--min-cluster-size", min=1, help="Minimum regions per cluster."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Cluster image regions using deterministic embedding distances."""
    config = load_config(config_path)
    _init_paths(config)
    result = cluster_image_regions(
        config,
        source_id=source_id,
        embedding_model=embedding_model,
        distance_threshold=distance_threshold,
        min_cluster_size=min_cluster_size,
    )
    typer.echo(
        f"clusters_created={result.created} clusters_skipped={result.skipped} "
        f"clustered_regions={result.clustered_regions}"
    )


@app.command("build-image-cluster-evidence")
def build_image_cluster_evidence_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only build cluster evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create visual_cluster evidence records from image feature clusters."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_image_cluster_evidence(config, source_id=source_id)
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("build-evidence")
def build_evidence_command(
    modality: str = typer.Option("all", "--modality", help="Modality to build: all, chat, pdf, audio, or image."),
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only build evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Build evidence records for one or all modalities."""
    _ensure_modality(modality, EVIDENCE_MODALITIES, "build-evidence")
    config = load_config(config_path)
    _init_paths(config)
    outputs = []
    if modality in {"all", "chat"}:
        result = build_chat_evidence(config, source_id=source_id)
        outputs.append(f"chat_evidence_created={result.created} chat_evidence_skipped={result.skipped}")
    if modality in {"all", "pdf"}:
        result = build_pdf_evidence(config, source_id=source_id)
        outputs.append(f"pdf_evidence_created={result.created} pdf_evidence_skipped={result.skipped}")
    if modality in {"all", "audio"}:
        result = build_audio_evidence(config, source_id=source_id)
        outputs.append(f"audio_evidence_created={result.created} audio_evidence_skipped={result.skipped}")
    if modality in {"all", "image"}:
        region_result = build_image_evidence(config, source_id=source_id)
        cluster_result = build_image_cluster_evidence(config, source_id=source_id)
        outputs.append(
            f"image_evidence_created={region_result.created + cluster_result.created} "
            f"image_evidence_skipped={region_result.skipped + cluster_result.skipped}"
        )
    typer.echo(" ".join(outputs))


@app.command("chunk")
def chunk_command(
    modality: str = typer.Option("all", "--modality", help="Modality to chunk: all, chat, pdf, or audio."),
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only chunk this source."),
    previous_messages: int = typer.Option(2, "--previous-messages", min=0, help="Chat previous messages to include."),
    previous_utterances: int = typer.Option(1, "--previous-utterances", min=0, help="Audio previous utterances to include."),
    max_tokens: int = typer.Option(1200, "--max-tokens", min=1, help="Chat/audio policy metadata token budget."),
    target_tokens: int = typer.Option(1200, "--target-tokens", min=1, help="PDF approximate target token budget."),
    overlap_tokens: int = typer.Option(150, "--overlap-tokens", min=0, help="PDF policy metadata overlap budget."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Build context chunks for one or all text-like modalities."""
    _ensure_modality(modality, TEXT_CHUNK_MODALITIES, "chunk")
    config = load_config(config_path)
    _init_paths(config)
    outputs = []
    if modality in {"all", "chat"}:
        result = chunk_chat(config, source_id=source_id, previous_messages=previous_messages, max_tokens=max_tokens)
        outputs.append(f"chat_chunks_created={result.created} chat_chunks_skipped={result.skipped}")
    if modality in {"all", "pdf"}:
        result = chunk_pdf(
            config,
            source_id=source_id,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        outputs.append(f"pdf_chunks_created={result.created} pdf_chunks_skipped={result.skipped}")
    if modality in {"all", "audio"}:
        result = chunk_audio(
            config,
            source_id=source_id,
            previous_utterances=previous_utterances,
            max_tokens=max_tokens,
        )
        outputs.append(f"audio_chunks_created={result.created} audio_chunks_skipped={result.skipped}")
    typer.echo(" ".join(outputs))


@app.command("detect-spans")
def detect_spans_command(
    modality: str = typer.Option("all", "--modality", help="Modality to detect: all, chat, pdf, or audio."),
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only detect spans for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Detect claim-bearing spans for one or all text-like modalities."""
    _ensure_modality(modality, TEXT_SPAN_MODALITIES, "detect-spans")
    config = load_config(config_path)
    _init_paths(config)
    outputs = []
    if modality in {"all", "chat"}:
        result = detect_chat_spans(config, source_id=source_id)
        outputs.append(f"chat_spans_created={result.created} chat_spans_skipped={result.skipped}")
    if modality in {"all", "pdf"}:
        result = detect_pdf_spans(config, source_id=source_id)
        outputs.append(f"pdf_spans_created={result.created} pdf_spans_skipped={result.skipped}")
    if modality in {"all", "audio"}:
        result = detect_audio_spans(config, source_id=source_id)
        outputs.append(f"audio_spans_created={result.created} audio_spans_skipped={result.skipped}")
    typer.echo(" ".join(outputs))


@app.command("run-chat")
def run_chat_command(
    chat_export: Path = typer.Argument(..., help="Chat export JSON file."),
    previous_messages: int = typer.Option(2, "--previous-messages", min=0, help="Previous messages to include as context."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Run chat ingest through normalized claims, graph export, and report."""
    config = load_config(config_path)
    _init_paths(config)
    if not chat_export.exists() or not chat_export.is_file():
        raise typer.BadParameter(f"chat export does not exist: {chat_export}")
    ingest_result = ingest_chat_export(chat_export, config)
    evidence_result = build_chat_evidence(config, source_id=ingest_result.source_id)
    chunk_result = chunk_chat(config, source_id=ingest_result.source_id, previous_messages=previous_messages)
    span_result = detect_chat_spans(config, source_id=ingest_result.source_id)
    extract_result = extract_claims_from_spans(config, modality="chat", source_id=ingest_result.source_id)
    validation_result = validate_raw_claims(config, source_id=ingest_result.source_id)
    normalization_result = normalize_claims(config, source_id=ingest_result.source_id)
    graph_result = export_graph_jsonl(config)
    report_result = write_summary_report(config)
    typer.echo(
        f"source_id={ingest_result.source_id} messages_created={ingest_result.messages_created} "
        f"evidence_created={evidence_result.created} chunks_created={chunk_result.created} "
        f"spans_created={span_result.created} claims_created={extract_result.created} "
        f"claims_accepted={validation_result.accepted} claims_normalized={normalization_result.created} "
        f"graph_edges={graph_result.edge_count} report={report_result.output_path}"
    )


@app.command("run-pdf")
def run_pdf_command(
    pdf_file: Path = typer.Argument(..., help="PDF file to ingest."),
    target_tokens: int = typer.Option(1200, "--target-tokens", min=1, help="Approximate target token budget."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Run PDF ingest through normalized claims, graph export, and report."""
    config = load_config(config_path)
    _init_paths(config)
    if not pdf_file.exists() or not pdf_file.is_file():
        raise typer.BadParameter(f"PDF file does not exist: {pdf_file}")
    try:
        ingest_result = ingest_pdf(pdf_file, config)
    except RuntimeError as exc:
        raise typer.BadParameter(str(exc))
    evidence_result = build_pdf_evidence(config, source_id=ingest_result.source_id)
    chunk_result = chunk_pdf(config, source_id=ingest_result.source_id, target_tokens=target_tokens)
    span_result = detect_pdf_spans(config, source_id=ingest_result.source_id)
    extract_result = extract_claims_from_spans(config, modality="pdf", source_id=ingest_result.source_id)
    validation_result = validate_raw_claims(config, source_id=ingest_result.source_id)
    normalization_result = normalize_claims(config, source_id=ingest_result.source_id)
    graph_result = export_graph_jsonl(config)
    report_result = write_summary_report(config)
    typer.echo(
        f"source_id={ingest_result.source_id} blocks_created={ingest_result.blocks_created} "
        f"evidence_created={evidence_result.created} chunks_created={chunk_result.created} "
        f"spans_created={span_result.created} claims_created={extract_result.created} "
        f"claims_accepted={validation_result.accepted} claims_normalized={normalization_result.created} "
        f"graph_edges={graph_result.edge_count} report={report_result.output_path}"
    )


@app.command("run-audio-transcript")
def run_audio_transcript_command(
    transcript_file: Path = typer.Argument(..., help="Audio transcript JSON file."),
    previous_utterances: int = typer.Option(1, "--previous-utterances", min=0, help="Previous utterances to include as context."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Run audio transcript ingest through normalized claims, graph export, and report."""
    config = load_config(config_path)
    _init_paths(config)
    if not transcript_file.exists() or not transcript_file.is_file():
        raise typer.BadParameter(f"transcript file does not exist: {transcript_file}")
    ingest_result = ingest_audio_transcript(transcript_file, config)
    evidence_result = build_audio_evidence(config, source_id=ingest_result.source_id)
    chunk_result = chunk_audio(config, source_id=ingest_result.source_id, previous_utterances=previous_utterances)
    span_result = detect_audio_spans(config, source_id=ingest_result.source_id)
    extract_result = extract_claims_from_spans(config, modality="audio", source_id=ingest_result.source_id)
    validation_result = validate_raw_claims(config, source_id=ingest_result.source_id)
    normalization_result = normalize_claims(config, source_id=ingest_result.source_id)
    graph_result = export_graph_jsonl(config)
    report_result = write_summary_report(config)
    typer.echo(
        f"source_id={ingest_result.source_id} utterances_created={ingest_result.utterances_created} "
        f"evidence_created={evidence_result.created} chunks_created={chunk_result.created} "
        f"spans_created={span_result.created} claims_created={extract_result.created} "
        f"claims_accepted={validation_result.accepted} claims_normalized={normalization_result.created} "
        f"graph_edges={graph_result.edge_count} report={report_result.output_path}"
    )


@app.command("run-images")
def run_images_command(
    image_path: Path = typer.Argument(..., help="Image file or directory to ingest."),
    patch_size: int = typer.Option(224, "--patch-size", min=1, help="Grid patch size in pixels."),
    stride: int = typer.Option(112, "--stride", min=1, help="Grid stride in pixels."),
    distance_threshold: float = typer.Option(
        0.08,
        "--cluster-distance-threshold",
        min=0.0,
        help="Maximum embedding distance for cluster membership.",
    ),
    min_cluster_size: int = typer.Option(2, "--min-cluster-size", min=1, help="Minimum regions per cluster."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Run image ingest through normalized region claims, graph export, and report."""
    config = load_config(config_path)
    _init_paths(config)
    if not image_path.exists():
        raise typer.BadParameter(f"image path does not exist: {image_path}")
    ingest_result = ingest_images(image_path, config)
    regions_created = 0
    evidence_created = 0
    embeddings_created = 0
    clusters_created = 0
    claims_created = 0
    claims_accepted = 0
    claims_normalized = 0
    for source_id in ingest_result.source_ids:
        region_result = propose_image_regions(config, source_id=source_id, patch_size=patch_size, stride=stride)
        evidence_result = build_image_evidence(config, source_id=source_id)
        embedding_result = build_image_region_embeddings(config, source_id=source_id)
        cluster_result = cluster_image_regions(
            config,
            source_id=source_id,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        )
        cluster_evidence_result = build_image_cluster_evidence(config, source_id=source_id)
        extract_result = extract_claims_from_spans(config, modality="image", source_id=source_id)
        validation_result = validate_raw_claims(config, source_id=source_id)
        normalization_result = normalize_claims(config, source_id=source_id)
        regions_created += region_result.created
        evidence_created += evidence_result.created + cluster_evidence_result.created
        embeddings_created += embedding_result.created
        clusters_created += cluster_result.created
        claims_created += extract_result.created
        claims_accepted += validation_result.accepted
        claims_normalized += normalization_result.created
    graph_result = export_graph_jsonl(config)
    report_result = write_summary_report(config)
    typer.echo(
        f"sources_created={ingest_result.sources_created} images_created={ingest_result.images_created} "
        f"regions_created={regions_created} evidence_created={evidence_created} "
        f"embeddings_created={embeddings_created} clusters_created={clusters_created} "
        f"claims_created={claims_created} claims_accepted={claims_accepted} "
        f"claims_normalized={claims_normalized} graph_edges={graph_result.edge_count} "
        f"report={report_result.output_path}"
    )


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
    record_job_result(
        config,
        stage="validate_claims",
        source_id=source_id,
        input_record_ids=_stage_input_ids("claims_raw", source_id=source_id, record_ids=claim_id),
        model_id=VALIDATOR_VERSION,
        metrics={
            "claims_accepted": result.accepted,
            "claims_quarantined": result.quarantined,
            "claims_skipped": result.skipped,
        },
    )
    typer.echo(
        f"claims_accepted={result.accepted} claims_quarantined={result.quarantined} "
        f"claims_skipped={result.skipped}"
    )


@app.command("extract-claims")
def extract_claims_command(
    modality: str = typer.Option("all", "--modality", help="Modality to extract: all, chat, pdf, audio, or image."),
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
    record_job_result(
        config,
        stage="extract_claims",
        source_id=source_id,
        input_record_ids=_stage_input_ids(f"modality:{modality}", source_id=source_id),
        model_id=_extract_model_id(modality),
        metrics={"claims_created": result.created, "claims_skipped": result.skipped},
        metadata={"modality": modality},
    )
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


@app.command("route-models")
def route_models_command(
    stage: str = typer.Option("all", "--stage", help="Routing stage: all, extraction, or validation."),
    models_config: Path = typer.Option(Path("configs/models.yaml"), "--models-config", help="Models config path."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Model routing JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write cheap/strong model routing recommendations without invoking models."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = write_model_routing_report(
            config,
            models_config_path=models_config,
            output_path=output,
            stage=stage,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"{result.output_path} recommendations={result.recommendation_count}")


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
    record_job_result(
        config,
        stage="normalize_claims",
        source_id=source_id,
        input_record_ids=_stage_input_ids("claims_validated", source_id=source_id, record_ids=claim_id),
        model_id=NORMALIZER_VERSION,
        metrics={"claims_normalized": result.created, "claims_skipped": result.skipped},
    )
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


@app.command("export-sqlite")
def export_sqlite_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="SQLite output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Export configured JSONL artifacts into SQLite tables."""
    config = load_config(config_path)
    _init_paths(config)
    result = export_sqlite(config, output_path=output)
    typer.echo(
        f"{result.output_path} tables={len(result.table_counts)} "
        f"records={sum(result.table_counts.values())}"
    )


@app.command("export-metta")
def export_metta_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="MeTTa output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Export normalized claims as MeTTa-style S-expressions."""
    config = load_config(config_path)
    _init_paths(config)
    result = export_metta(config, output_path=output)
    typer.echo(f"{result.output_path} claims={result.claim_count}")


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


@app.command("trace-claim")
def trace_claim_command(
    claim_id: str = typer.Argument(..., help="Claim ID to trace."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Trace a claim back through source, evidence, span, validation, and normalization artifacts."""
    config = load_config(config_path)
    _init_paths(config)
    if output is not None:
        trace = write_claim_trace(config, claim_id, output)
        typer.echo(f"{output} found={trace['found']}")
        return
    typer.echo(json.dumps(trace_claim(config, claim_id), indent=2, sort_keys=True))


@app.command("review-claim")
def review_claim_command(
    claim_id: str = typer.Argument(..., help="Claim ID to review."),
    decision: str = typer.Option(..., "--decision", help="Review decision: accept, reject, or needs_review."),
    reviewer_id: str = typer.Option("human_reviewer", "--reviewer-id", help="Reviewer identifier."),
    reason_code: Optional[List[str]] = typer.Option(None, "--reason-code", help="Reason code. Repeatable."),
    notes: Optional[str] = typer.Option(None, "--notes", help="Optional reviewer notes."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Record a human review decision for a claim."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = record_claim_review(
            config,
            claim_id=claim_id,
            decision=decision,
            reviewer_id=reviewer_id,
            reason_codes=reason_code,
            notes=notes,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"review_id={result.review_id} created={result.created}")


@app.command("dedupe-claims")
def dedupe_claims_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Duplicate group JSONL output path."),
    include_singletons: bool = typer.Option(False, "--include-singletons", help="Include non-duplicate singleton groups."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Group duplicate normalized claims into a derived JSONL report."""
    config = load_config(config_path)
    _init_paths(config)
    result = dedupe_normalized_claims(config, output_path=output, include_singletons=include_singletons)
    typer.echo(f"{result.output_path} groups={result.group_count}")


@app.command("repair-claims")
def repair_claims_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Repair suggestion JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write reviewable evidence_text repair suggestions for raw claims."""
    config = load_config(config_path)
    _init_paths(config)
    result = suggest_evidence_repairs(config, output_path=output)
    typer.echo(f"{result.output_path} suggestions={result.suggestion_count}")


@app.command("detect-pii")
def detect_pii_command(
    artifact: str = typer.Option("all", "--artifact", help="Artifact to scan, or all."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="PII findings JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Detect simple PII patterns in text-like artifacts without storing raw matches."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = detect_pii(config, artifact=artifact, output_path=output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"{result.output_path} findings={result.finding_count}")


@app.command("redact-pii")
def redact_pii_command(
    artifact: str = typer.Option(..., "--artifact", help="Artifact to redact."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Redacted JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write a redacted copy of one text-like artifact without modifying canonical JSONL."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = redact_pii(config, artifact=artifact, output_path=output)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"{result.output_path} records={result.records_written} replacements={result.replacement_count}")


@app.command("check-privacy")
def check_privacy_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Privacy violation JSONL output path."),
    fail_on_violation: bool = typer.Option(
        True,
        "--fail-on-violation/--report-only",
        help="Exit non-zero when privacy policy violations are found.",
    ),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Check local-only privacy policy for sensitive sources."""
    config = load_config(config_path)
    _init_paths(config)
    result = check_privacy_policy(config, output_path=output)
    typer.echo(
        f"{result.output_path} claims_checked={result.claims_checked} "
        f"violations={result.violation_count}"
    )
    if fail_on_violation and result.violation_count:
        raise typer.Exit(code=1)


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
