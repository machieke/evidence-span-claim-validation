from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import typer
from pydantic import ValidationError

from evidence_pipeline.artifact_validation import ArtifactValidationResult, validate_known_artifacts
from evidence_pipeline.chunking.chat_chunker import chunk_chat
from evidence_pipeline.chunking.audio_chunker import chunk_audio
from evidence_pipeline.chunking.image_ocr_chunker import chunk_image_ocr
from evidence_pipeline.chunking.pdf_chunker import chunk_pdf
from evidence_pipeline.config import PipelineConfig, load_config
from evidence_pipeline.demo import DEMO_SEED_VERSION, seed_demo_artifacts
from evidence_pipeline.extraction.claim_extractor import (
    IMAGE_CLUSTER_EXTRACTOR_VERSION,
    IMAGE_REGION_EXTRACTOR_VERSION,
    RULE_EXTRACTOR_VERSION,
    extract_claims_from_spans,
)
from evidence_pipeline.extraction.image_classifier import COLOR_CLASSIFIER_MODEL, classify_image_regions
from evidence_pipeline.ingest.chat import ingest_chat_export
from evidence_pipeline.ingest.chat_evidence import build_chat_evidence
from evidence_pipeline.ingest.audio import (
    AUDIO_NORMALIZATION_VERSION,
    ingest_audio_transcript,
    normalize_audio_source,
)
from evidence_pipeline.ingest.audio_evidence import build_audio_evidence
from evidence_pipeline.ingest.image import ingest_images
from evidence_pipeline.ingest.image_evidence import build_image_cluster_evidence, build_image_evidence
from evidence_pipeline.ingest.image_ocr import ingest_image_ocr
from evidence_pipeline.ingest.pdf import ingest_pdf
from evidence_pipeline.ingest.pdf_evidence import build_pdf_evidence
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jobs import record_job_result
from evidence_pipeline.jsonl import JSONLDecodeError, append_jsonl, find_record, read_jsonl
from evidence_pipeline.model_routing import MODEL_ROUTING_VERSION, write_model_routing_report
from evidence_pipeline.normalization.claims import NORMALIZER_VERSION, normalize_claims
from evidence_pipeline.normalization.dedupe import DEDUPE_VERSION, dedupe_normalized_claims
from evidence_pipeline.normalization.graph_export import GRAPH_EXPORT_VERSION, export_graph_jsonl
from evidence_pipeline.normalization.metta_export import METTA_EXPORT_VERSION, export_metta
from evidence_pipeline.retention import RETENTION_PLAN_VERSION, write_retention_plan
from evidence_pipeline.reports.summary import write_summary_report
from evidence_pipeline.reports.acceptance import ACCEPTANCE_CHECK_VERSION, write_acceptance_report
from evidence_pipeline.reports.gold_eval import GOLD_EVAL_VERSION, write_gold_eval_report
from evidence_pipeline.reports.lineage import (
    default_claim_trace_html_path,
    trace_claim,
    write_claim_trace,
    write_claim_trace_html,
)
from evidence_pipeline.reports.sqlite_export import SQLITE_EXPORT_VERSION, export_sqlite
from evidence_pipeline.schemas import SCHEMA_REGISTRY, EvidenceRecord, SourceModality, SourceRecord
from evidence_pipeline.spans.image_region_clusterer import (
    COLOR_CLUSTERING_METHOD,
    COLOR_EMBEDDING_MODEL,
    build_image_region_embeddings,
    cluster_image_regions,
)
from evidence_pipeline.spans.image_region_selector import propose_image_regions
from evidence_pipeline.spans.rule_highlighter import (
    detect_audio_spans,
    detect_chat_spans,
    detect_image_ocr_spans,
    detect_pdf_spans,
)
from evidence_pipeline.validation.deterministic import VALIDATOR_VERSION, validate_raw_claims
from evidence_pipeline.validation.pii import PII_PROCESSOR_VERSION, detect_pii, redact_pii
from evidence_pipeline.validation.privacy import PRIVACY_CHECK_VERSION, check_privacy_policy
from evidence_pipeline.validation.repair import (
    REPAIR_APPLICATION_VERSION,
    REPAIR_REASON_CODES,
    REPAIR_SUGGESTION_VERSION,
    apply_evidence_repairs,
    suggest_evidence_repairs,
)
from evidence_pipeline.validation.review import REVIEW_QUEUE_VERSION, record_claim_review, write_review_queue
from evidence_pipeline.validation.schema_repair import SCHEMA_REPAIR_VERSION, import_raw_claim_candidates

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
TEXT_CHUNK_MODALITIES = {"all", "chat", "pdf", "audio", "image"}
TEXT_SPAN_MODALITIES = {"all", "chat", "pdf", "audio", "image"}

SOURCE_REGISTRATION_VERSION = "source.registration.v1"
INGEST_VERSIONS = {
    "chat": "chat.ingest.v1",
    "pdf": "pdf.ingest.v1",
    "audio": "audio_transcript.ingest.v1",
    "image": "image.ingest.v1",
    "image_ocr": "image_ocr.ingest.v1",
}
EVIDENCE_BUILDER_VERSIONS = {
    "chat": "chat_evidence.builder.v1",
    "pdf": "pdf_evidence.builder.v1",
    "audio": "audio_evidence.builder.v1",
    "image_region": "image_region_evidence.builder.v1",
    "image_cluster": "image_cluster_evidence.builder.v1",
}
CHUNKER_VERSIONS = {
    "chat": "chat_chunker.thread_window.v1",
    "pdf": "pdf_chunker.section_page_block_token_fallback.v2",
    "audio": "audio_chunker.utterance_window.v1",
    "image_ocr": "image_ocr_chunker.single_evidence.v1",
}
SPAN_DETECTOR_VERSIONS = {
    "chat": "chat_rules_v1",
    "pdf": "pdf_rules_v1",
    "audio": "audio_rules_v1",
    "image_ocr": "image_ocr_rules_v1",
}
IMAGE_REGION_PROPOSAL_VERSION = "image_region_proposal.grid.v1"
EVIDENCE_STAGE_INPUTS = {
    "build_chat_evidence": ("chat_messages", None, None),
    "build_pdf_evidence": ("pdf_blocks", None, None),
    "build_audio_evidence": ("audio_utterances", None, None),
    "build_image_evidence": ("image_regions", None, None),
    "build_image_cluster_evidence": ("image_feature_clusters", None, None),
}
CHUNK_STAGE_INPUTS = {
    "chunk_chat": ("evidence", "chat", "message_span"),
    "chunk_pdf": ("evidence", "pdf", "text_span"),
    "chunk_audio": ("evidence", "audio", "utterance_span"),
    "chunk_image_ocr": ("evidence", "image", "ocr_text_span"),
}
SPAN_STAGE_MODALITIES = {
    "detect_chat_spans": "chat",
    "detect_pdf_spans": "pdf",
    "detect_audio_spans": "audio",
    "detect_image_ocr_spans": "image",
}
ACCEPTANCE_CHECK_INPUT_IDS = [
    "sources",
    "evidence",
    "chunks",
    "spans",
    "image_regions",
    "claims_raw",
    "validations",
    "claims_validated",
    "claims_normalized",
    "quarantine",
    "reports:extraction_summary",
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


def _policy_ids(**values: object) -> List[str]:
    return [f"policy:{key}={value}" for key, value in sorted(values.items())]


def _single_source_id(source_ids: List[str]) -> Optional[str]:
    unique_source_ids = sorted(set(source_ids))
    if len(unique_source_ids) == 1:
        return unique_source_ids[0]
    return None


def _source_scope(
    default_id: str,
    source_id: Optional[str] = None,
    source_ids: Optional[List[str]] = None,
    extra_ids: Optional[List[str]] = None,
) -> Tuple[Optional[str], List[str]]:
    source_ids = sorted(set(source_ids or []))
    effective_source_id = source_id or _single_source_id(source_ids)
    if effective_source_id:
        ids = [effective_source_id]
    elif source_ids:
        ids = source_ids
    else:
        ids = [default_id]
    ids.extend(extra_ids or [])
    return effective_source_id, ids


def _jsonl_source_ids(
    config: PipelineConfig,
    artifact: str,
    modality: Optional[str] = None,
    evidence_type: Optional[str] = None,
) -> List[str]:
    source_ids = set()
    for _, row in read_jsonl(config.jsonl_paths()[artifact]):
        if modality is not None and row.get("source_modality") != modality:
            continue
        if evidence_type is not None and row.get("evidence_type") != evidence_type:
            continue
        source_id = row.get("source_id")
        if source_id:
            source_ids.add(str(source_id))
        for value in row.get("source_ids", []) or []:
            source_ids.add(str(value))
    return sorted(source_ids)


def _record_source_registration_job(
    config: PipelineConfig,
    source_id: str,
    source_file: Path,
    modality: SourceModality,
    source_created: bool,
) -> None:
    record_job_result(
        config,
        stage="register_source",
        source_id=source_id,
        input_record_ids=[source_id],
        model_id=SOURCE_REGISTRATION_VERSION,
        metrics={"sources_created": int(source_created), "sources_skipped": int(not source_created)},
        metadata={"source_file": str(source_file), "modality": modality},
    )


def _record_ingest_chat_job(config: PipelineConfig, input_path: Path, result) -> None:
    record_job_result(
        config,
        stage="ingest_chat",
        source_id=result.source_id,
        input_record_ids=[result.source_id],
        model_id=INGEST_VERSIONS["chat"],
        metrics={
            "sources_created": int(result.source_created),
            "sources_skipped": int(not result.source_created),
            "messages_created": result.messages_created,
            "messages_skipped": result.messages_skipped,
        },
        metadata={"input_path": str(input_path), "modality": "chat"},
    )


def _record_ingest_pdf_job(config: PipelineConfig, input_path: Path, result) -> None:
    record_job_result(
        config,
        stage="ingest_pdf",
        source_id=result.source_id,
        input_record_ids=[result.source_id],
        model_id=INGEST_VERSIONS["pdf"],
        metrics={
            "sources_created": int(result.source_created),
            "sources_skipped": int(not result.source_created),
            "blocks_created": result.blocks_created,
            "blocks_skipped": result.blocks_skipped,
        },
        metadata={"input_path": str(input_path), "modality": "pdf", "extractor": result.extractor},
    )


def _record_ingest_audio_job(config: PipelineConfig, input_path: Path, result) -> None:
    record_job_result(
        config,
        stage="ingest_audio_transcript",
        source_id=result.source_id,
        input_record_ids=[result.source_id],
        model_id=INGEST_VERSIONS["audio"],
        metrics={
            "sources_created": int(result.source_created),
            "sources_skipped": int(not result.source_created),
            "utterances_created": result.utterances_created,
            "utterances_skipped": result.utterances_skipped,
        },
        metadata={"input_path": str(input_path), "modality": "audio"},
    )


def _record_normalize_audio_job(
    config: PipelineConfig,
    input_path: Path,
    result,
    sample_rate: int,
    channels: int,
) -> None:
    record_job_result(
        config,
        stage="normalize_audio",
        source_id=result.source_id,
        input_record_ids=[f"audio:{input_path}"],
        model_id=AUDIO_NORMALIZATION_VERSION,
        metrics={
            "source_created": int(result.source_created),
            "executed": int(result.executed),
        },
        metadata={
            "input_path": str(input_path),
            "normalized_file": str(result.normalized_file),
            "sample_rate": sample_rate,
            "channels": channels,
            "command": result.command,
        },
    )


def _record_ingest_images_job(config: PipelineConfig, input_path: Path, result) -> None:
    source_ids = sorted(set(result.source_ids))
    record_job_result(
        config,
        stage="ingest_images",
        source_id=_single_source_id(source_ids),
        input_record_ids=source_ids or [f"path:{input_path}"],
        model_id=INGEST_VERSIONS["image"],
        metrics={
            "sources_created": result.sources_created,
            "images_created": result.images_created,
            "images_skipped": result.images_skipped,
        },
        metadata={"input_path": str(input_path), "modality": "image", "source_count": len(source_ids)},
    )


def _record_ingest_image_ocr_job(config: PipelineConfig, input_path: Path, result) -> None:
    source_ids = sorted(set(result.source_ids))
    record_job_result(
        config,
        stage="ingest_image_ocr",
        source_id=_single_source_id(source_ids),
        input_record_ids=source_ids or [f"path:{input_path}"],
        model_id=INGEST_VERSIONS["image_ocr"],
        metrics={"ocr_evidence_created": result.created, "ocr_evidence_skipped": result.skipped},
        metadata={"input_path": str(input_path), "modality": "image"},
    )


def _record_evidence_job(config: PipelineConfig, stage: str, source_id: Optional[str], result, model_id: str) -> None:
    artifact, modality, evidence_type = EVIDENCE_STAGE_INPUTS[stage]
    effective_source_id, input_record_ids = _source_scope(
        f"artifact:{stage}",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, artifact, modality=modality, evidence_type=evidence_type),
    )
    record_job_result(
        config,
        stage=stage,
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=model_id,
        metrics={"evidence_created": result.created, "evidence_skipped": result.skipped},
    )


def _record_chunk_job(
    config: PipelineConfig,
    stage: str,
    source_id: Optional[str],
    result,
    model_id: str,
    input_artifact: str,
    policy: Optional[dict] = None,
) -> None:
    policy = policy or {}
    artifact, modality, evidence_type = CHUNK_STAGE_INPUTS[stage]
    effective_source_id, input_record_ids = _source_scope(
        f"artifact:{input_artifact}",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, artifact, modality=modality, evidence_type=evidence_type),
        extra_ids=_policy_ids(**policy),
    )
    record_job_result(
        config,
        stage=stage,
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=model_id,
        metrics={"chunks_created": result.created, "chunks_skipped": result.skipped},
        metadata={"policy": policy},
    )


def _record_span_detection_job(
    config: PipelineConfig,
    stage: str,
    source_id: Optional[str],
    result,
    model_id: str,
    modality: str,
) -> None:
    effective_source_id, input_record_ids = _source_scope(
        f"modality:{modality}",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, "chunks", modality=SPAN_STAGE_MODALITIES[stage]),
    )
    record_job_result(
        config,
        stage=stage,
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=model_id,
        metrics={"spans_created": result.created, "spans_skipped": result.skipped},
        metadata={"modality": modality},
    )


def _record_image_region_proposal_job(
    config: PipelineConfig,
    source_id: Optional[str],
    result,
    patch_size: int,
    stride: int,
) -> None:
    effective_source_id, input_record_ids = _source_scope(
        "artifact:images",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, "images"),
        extra_ids=_policy_ids(patch_size=patch_size, stride=stride),
    )
    record_job_result(
        config,
        stage="propose_image_regions",
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=IMAGE_REGION_PROPOSAL_VERSION,
        metrics={"regions_created": result.created, "regions_skipped": result.skipped},
        metadata={"patch_size": patch_size, "stride": stride},
    )


def _record_image_embedding_job(config: PipelineConfig, source_id: Optional[str], result, embedding_model: str) -> None:
    effective_source_id, input_record_ids = _source_scope(
        "artifact:image_regions",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, "image_regions"),
        extra_ids=_policy_ids(embedding_model=embedding_model),
    )
    record_job_result(
        config,
        stage="embed_image_regions",
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=embedding_model,
        metrics={"embeddings_created": result.created, "embeddings_skipped": result.skipped},
        metadata={"embedding_model": embedding_model},
    )


def _record_image_classification_job(
    config: PipelineConfig,
    source_id: Optional[str],
    result,
    embedding_model: str,
    classifier_model: str,
) -> None:
    effective_source_id, input_record_ids = _source_scope(
        "artifact:image_region_embeddings",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, "image_region_embeddings"),
        extra_ids=_policy_ids(embedding_model=embedding_model, classifier_model=classifier_model),
    )
    record_job_result(
        config,
        stage="classify_image_regions",
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=classifier_model,
        metrics={"classifications_created": result.created, "classifications_skipped": result.skipped},
        metadata={"embedding_model": embedding_model, "classifier_model": classifier_model},
    )


def _record_image_clustering_job(
    config: PipelineConfig,
    source_id: Optional[str],
    result,
    embedding_model: str,
    distance_threshold: float,
    min_cluster_size: int,
) -> None:
    effective_source_id, input_record_ids = _source_scope(
        "artifact:image_region_embeddings",
        source_id=source_id,
        source_ids=_jsonl_source_ids(config, "image_region_embeddings"),
        extra_ids=_policy_ids(
            embedding_model=embedding_model,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        ),
    )
    record_job_result(
        config,
        stage="cluster_image_regions",
        source_id=effective_source_id,
        input_record_ids=input_record_ids,
        model_id=f"{COLOR_CLUSTERING_METHOD}+{embedding_model}",
        metrics={
            "clusters_created": result.created,
            "clusters_skipped": result.skipped,
            "clustered_regions": result.clustered_regions,
        },
        metadata={
            "embedding_model": embedding_model,
            "distance_threshold": distance_threshold,
            "min_cluster_size": min_cluster_size,
        },
    )


def _extract_model_id(modality: str) -> str:
    if modality == "image":
        return f"{IMAGE_REGION_EXTRACTOR_VERSION}+{IMAGE_CLUSTER_EXTRACTOR_VERSION}"
    return RULE_EXTRACTOR_VERSION


def _record_extract_claims_job(
    config: PipelineConfig,
    modality: str,
    source_id: Optional[str],
    result,
    batch_size: Optional[int] = None,
) -> None:
    metrics = {"claims_created": result.created, "claims_skipped": result.skipped}
    metadata = {"modality": modality}
    if batch_size is not None:
        metrics["batches_processed"] = result.batches_processed
        metadata["batch_size"] = batch_size
    record_job_result(
        config,
        stage="extract_claims",
        source_id=source_id,
        input_record_ids=_stage_input_ids(f"modality:{modality}", source_id=source_id),
        model_id=_extract_model_id(modality),
        metrics=metrics,
        metadata=metadata,
    )


def _record_validate_claims_job(
    config: PipelineConfig,
    source_id: Optional[str],
    claim_ids: Optional[List[str]],
    result,
) -> None:
    record_job_result(
        config,
        stage="validate_claims",
        source_id=source_id,
        input_record_ids=_stage_input_ids("claims_raw", source_id=source_id, record_ids=claim_ids),
        model_id=VALIDATOR_VERSION,
        metrics={
            "claims_accepted": result.accepted,
            "claims_quarantined": result.quarantined,
            "claims_skipped": result.skipped,
        },
    )


def _record_normalize_claims_job(
    config: PipelineConfig,
    source_id: Optional[str],
    claim_ids: Optional[List[str]],
    result,
) -> None:
    record_job_result(
        config,
        stage="normalize_claims",
        source_id=source_id,
        input_record_ids=_stage_input_ids("claims_validated", source_id=source_id, record_ids=claim_ids),
        model_id=NORMALIZER_VERSION,
        metrics={"claims_normalized": result.created, "claims_skipped": result.skipped},
    )


def _record_graph_export_job(config: PipelineConfig, result, output_format: str = "jsonl") -> None:
    record_job_result(
        config,
        stage="export_graph",
        input_record_ids=["claims_normalized"],
        model_id=GRAPH_EXPORT_VERSION,
        metrics={"edges": result.edge_count},
        metadata={"format": output_format, "output_path": str(result.output_path)},
    )


def _record_acceptance_check_job(config: PipelineConfig, result) -> int:
    failed_checks = sum(1 for check in result.checks if check["status"] == "failed")
    record_job_result(
        config,
        stage="acceptance_check",
        input_record_ids=ACCEPTANCE_CHECK_INPUT_IDS,
        model_id=ACCEPTANCE_CHECK_VERSION,
        metrics={"checks": len(result.checks), "failed_checks": failed_checks},
        metadata={"output_path": str(result.output_path), "passed": result.passed},
    )
    return failed_checks


def _gold_eval_metrics(result) -> dict:
    return {
        "gold_claims": result.metrics["gold_claims"],
        "accepted_precision": result.metrics["accepted_precision"],
        "accepted_recall": result.metrics["accepted_recall"],
        "quarantine_precision": result.metrics["quarantine_precision"],
        "quarantine_recall": result.metrics["quarantine_recall"],
        "evidence_exact_match_rate": result.metrics["evidence_exact_match_rate"],
        "attribution_preservation_rate": result.metrics["attribution_preservation_rate"],
        "uncertainty_preservation_rate": result.metrics["uncertainty_preservation_rate"],
        "negation_preservation_rate": result.metrics["negation_preservation_rate"],
        "quantity_preservation_rate": result.metrics["quantity_preservation_rate"],
        "unsupported_entity_rate": result.metrics["unsupported_entity_rate"],
        "quarantine_rate": result.metrics["quarantine_rate"],
    }


def _record_gold_eval_job(config: PipelineConfig, gold_file: Path, result) -> None:
    record_job_result(
        config,
        stage="eval_gold",
        input_record_ids=["claims_validated", "quarantine", f"gold:{gold_file}"],
        model_id=GOLD_EVAL_VERSION,
        metrics=_gold_eval_metrics(result),
        metadata={
            "gold_path": str(gold_file),
            "metrics_path": str(result.metrics_path),
            "output_path": str(result.output_path),
        },
    )


def _record_sqlite_export_job(config: PipelineConfig, result) -> None:
    record_job_result(
        config,
        stage="export_sqlite",
        input_record_ids=["artifacts:jsonl", "reports:jsonl"],
        model_id=SQLITE_EXPORT_VERSION,
        metrics={
            "tables": len(result.table_counts),
            "records": sum(result.table_counts.values()),
        },
        metadata={"output_path": str(result.output_path), "table_counts": result.table_counts},
    )


def _echo_artifact_validation_result(result: ArtifactValidationResult, verbose: bool = True) -> None:
    for file_result in result.files:
        for error in file_result.errors:
            typer.echo(error, err=True)
        if verbose:
            typer.echo(f"{file_result.path}: checked {file_result.records_checked} records")


@app.command("init")
def init_command(
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create canonical data directories and empty JSONL artifacts."""
    config = load_config(config_path)
    _init_paths(config)
    typer.echo(f"initialized {config.paths.data_dir}")


@app.command("seed-demo-artifacts")
def seed_demo_artifacts_command(
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Seed deterministic multimodal demo artifacts for acceptance and finalization."""
    config = load_config(config_path)
    _init_paths(config)
    result = seed_demo_artifacts(config)
    record_job_result(
        config,
        stage="seed_demo_artifacts",
        input_record_ids=["demo:multimodal_acceptance_v1"],
        model_id=DEMO_SEED_VERSION,
        metrics={
            "records_created": result.created,
            "records_skipped": result.skipped,
            "gold_claims": result.gold_claims,
        },
        metadata={"artifact_counts": result.artifact_counts, "gold_path": str(result.gold_path)},
    )
    typer.echo(
        f"records_created={result.created} records_skipped={result.skipped} "
        f"artifacts={len(result.artifact_counts)} gold={result.gold_path} "
        f"gold_claims={result.gold_claims}"
    )


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
        _record_source_registration_job(config, source_id, source_file, modality, source_created=False)
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
    _record_source_registration_job(config, source_id, source_file, modality, source_created=True)
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
    _record_ingest_chat_job(config, chat_export, result)
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
    _record_evidence_job(
        config,
        "build_chat_evidence",
        source_id,
        result,
        EVIDENCE_BUILDER_VERSIONS["chat"],
    )
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
    _record_chunk_job(
        config,
        "chunk_chat",
        source_id,
        result,
        CHUNKER_VERSIONS["chat"],
        "evidence",
        {"previous_messages": previous_messages, "max_tokens": max_tokens},
    )
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
    _record_span_detection_job(
        config,
        "detect_chat_spans",
        source_id,
        result,
        SPAN_DETECTOR_VERSIONS["chat"],
        "chat",
    )
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
    _record_ingest_pdf_job(config, pdf_file, result)
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
    _record_evidence_job(
        config,
        "build_pdf_evidence",
        source_id,
        result,
        EVIDENCE_BUILDER_VERSIONS["pdf"],
    )
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
    _record_chunk_job(
        config,
        "chunk_pdf",
        source_id,
        result,
        CHUNKER_VERSIONS["pdf"],
        "evidence",
        {"target_tokens": target_tokens, "overlap_tokens": overlap_tokens},
    )
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
    _record_span_detection_job(
        config,
        "detect_pdf_spans",
        source_id,
        result,
        SPAN_DETECTOR_VERSIONS["pdf"],
        "pdf",
    )
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
    _record_ingest_audio_job(config, transcript_file, result)
    typer.echo(
        f"source_id={result.source_id} source_created={result.source_created} "
        f"utterances_created={result.utterances_created} utterances_skipped={result.utterances_skipped}"
    )


@app.command("normalize-audio")
def normalize_audio_command(
    audio_file: Path = typer.Argument(..., help="Audio media file to register and optionally normalize."),
    output: Optional[Path] = typer.Option(None, "--output", help="Normalized audio output path."),
    sample_rate: int = typer.Option(16000, "--sample-rate", min=1, help="Target sample rate."),
    channels: int = typer.Option(1, "--channels", min=1, help="Target channel count."),
    execute: bool = typer.Option(False, "--execute", help="Run ffmpeg instead of only planning the normalization."),
    metadata: Optional[List[str]] = typer.Option(None, "--metadata", help="Additional source metadata key=value. Repeatable."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Register an audio media source and plan or run ffmpeg normalization."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = normalize_audio_source(
            audio_file,
            config,
            normalized_file=output,
            sample_rate=sample_rate,
            channels=channels,
            execute=execute,
            metadata=_parse_metadata(metadata),
        )
    except (RuntimeError, ValueError) as exc:
        raise typer.BadParameter(str(exc))
    _record_normalize_audio_job(config, audio_file, result, sample_rate, channels)
    typer.echo(
        f"source_id={result.source_id} source_created={result.source_created} "
        f"normalized_file={result.normalized_file} executed={result.executed}"
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
    _record_evidence_job(
        config,
        "build_audio_evidence",
        source_id,
        result,
        EVIDENCE_BUILDER_VERSIONS["audio"],
    )
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
    _record_chunk_job(
        config,
        "chunk_audio",
        source_id,
        result,
        CHUNKER_VERSIONS["audio"],
        "evidence",
        {"previous_utterances": previous_utterances, "max_tokens": max_tokens},
    )
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
    _record_span_detection_job(
        config,
        "detect_audio_spans",
        source_id,
        result,
        SPAN_DETECTOR_VERSIONS["audio"],
        "audio",
    )
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
    _record_ingest_images_job(config, image_path, result)
    typer.echo(
        f"sources_created={result.sources_created} images_created={result.images_created} "
        f"images_skipped={result.images_skipped}"
    )


@app.command("ingest-image-ocr")
def ingest_image_ocr_command(
    ocr_file: Path = typer.Argument(..., help="OCR sidecar JSON file."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Ingest image OCR sidecar text as ocr_text_span evidence."""
    config = load_config(config_path)
    _init_paths(config)
    if not ocr_file.exists() or not ocr_file.is_file():
        raise typer.BadParameter(f"OCR file does not exist: {ocr_file}")
    try:
        result = ingest_image_ocr(ocr_file, config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    _record_ingest_image_ocr_job(config, ocr_file, result)
    typer.echo(f"ocr_evidence_created={result.created} ocr_evidence_skipped={result.skipped}")


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
    _record_image_region_proposal_job(config, source_id, result, patch_size, stride)
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
    _record_evidence_job(
        config,
        "build_image_evidence",
        source_id,
        result,
        EVIDENCE_BUILDER_VERSIONS["image_region"],
    )
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("embed-image-regions")
def embed_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only embed regions for this source."),
    embedding_model: str = typer.Option(
        COLOR_EMBEDDING_MODEL,
        "--embedding-model",
        help="Embedding model identifier.",
    ),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create deterministic image-region embedding records."""
    config = load_config(config_path)
    _init_paths(config)
    result = build_image_region_embeddings(config, source_id=source_id, embedding_model=embedding_model)
    _record_image_embedding_job(config, source_id, result, embedding_model)
    typer.echo(f"embeddings_created={result.created} embeddings_skipped={result.skipped}")


@app.command("classify-image-regions")
def classify_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only classify regions for this source."),
    embedding_model: str = typer.Option(
        COLOR_EMBEDDING_MODEL,
        "--embedding-model",
        help="Embedding model identifier.",
    ),
    classifier_model: str = typer.Option(
        COLOR_CLASSIFIER_MODEL,
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
    _record_image_classification_job(config, source_id, result, embedding_model, classifier_model)
    typer.echo(f"classifications_created={result.created} classifications_skipped={result.skipped}")


@app.command("cluster-image-regions")
def cluster_image_regions_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only cluster regions for this source."),
    embedding_model: str = typer.Option(
        COLOR_EMBEDDING_MODEL,
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
    _record_image_clustering_job(config, source_id, result, embedding_model, distance_threshold, min_cluster_size)
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
    _record_evidence_job(
        config,
        "build_image_cluster_evidence",
        source_id,
        result,
        EVIDENCE_BUILDER_VERSIONS["image_cluster"],
    )
    typer.echo(f"evidence_created={result.created} evidence_skipped={result.skipped}")


@app.command("chunk-image-ocr")
def chunk_image_ocr_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only chunk OCR evidence for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Create text chunks from image OCR evidence."""
    config = load_config(config_path)
    _init_paths(config)
    result = chunk_image_ocr(config, source_id=source_id)
    _record_chunk_job(
        config,
        "chunk_image_ocr",
        source_id,
        result,
        CHUNKER_VERSIONS["image_ocr"],
        "evidence",
    )
    typer.echo(f"chunks_created={result.created} chunks_skipped={result.skipped}")


@app.command("detect-image-ocr-spans")
def detect_image_ocr_spans_command(
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only detect OCR spans for this source."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Detect claim-bearing spans in image OCR evidence."""
    config = load_config(config_path)
    _init_paths(config)
    result = detect_image_ocr_spans(config, source_id=source_id)
    _record_span_detection_job(
        config,
        "detect_image_ocr_spans",
        source_id,
        result,
        SPAN_DETECTOR_VERSIONS["image_ocr"],
        "image",
    )
    typer.echo(f"spans_created={result.created} spans_skipped={result.skipped}")


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
        _record_evidence_job(
            config,
            "build_chat_evidence",
            source_id,
            result,
            EVIDENCE_BUILDER_VERSIONS["chat"],
        )
        outputs.append(f"chat_evidence_created={result.created} chat_evidence_skipped={result.skipped}")
    if modality in {"all", "pdf"}:
        result = build_pdf_evidence(config, source_id=source_id)
        _record_evidence_job(
            config,
            "build_pdf_evidence",
            source_id,
            result,
            EVIDENCE_BUILDER_VERSIONS["pdf"],
        )
        outputs.append(f"pdf_evidence_created={result.created} pdf_evidence_skipped={result.skipped}")
    if modality in {"all", "audio"}:
        result = build_audio_evidence(config, source_id=source_id)
        _record_evidence_job(
            config,
            "build_audio_evidence",
            source_id,
            result,
            EVIDENCE_BUILDER_VERSIONS["audio"],
        )
        outputs.append(f"audio_evidence_created={result.created} audio_evidence_skipped={result.skipped}")
    if modality in {"all", "image"}:
        region_result = build_image_evidence(config, source_id=source_id)
        _record_evidence_job(
            config,
            "build_image_evidence",
            source_id,
            region_result,
            EVIDENCE_BUILDER_VERSIONS["image_region"],
        )
        cluster_result = build_image_cluster_evidence(config, source_id=source_id)
        _record_evidence_job(
            config,
            "build_image_cluster_evidence",
            source_id,
            cluster_result,
            EVIDENCE_BUILDER_VERSIONS["image_cluster"],
        )
        outputs.append(
            f"image_evidence_created={region_result.created + cluster_result.created} "
            f"image_evidence_skipped={region_result.skipped + cluster_result.skipped}"
        )
    typer.echo(" ".join(outputs))


@app.command("chunk")
def chunk_command(
    modality: str = typer.Option("all", "--modality", help="Modality to chunk: all, chat, pdf, audio, or image."),
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
        _record_chunk_job(
            config,
            "chunk_chat",
            source_id,
            result,
            CHUNKER_VERSIONS["chat"],
            "evidence",
            {"previous_messages": previous_messages, "max_tokens": max_tokens},
        )
        outputs.append(f"chat_chunks_created={result.created} chat_chunks_skipped={result.skipped}")
    if modality in {"all", "pdf"}:
        result = chunk_pdf(
            config,
            source_id=source_id,
            target_tokens=target_tokens,
            overlap_tokens=overlap_tokens,
        )
        _record_chunk_job(
            config,
            "chunk_pdf",
            source_id,
            result,
            CHUNKER_VERSIONS["pdf"],
            "evidence",
            {"target_tokens": target_tokens, "overlap_tokens": overlap_tokens},
        )
        outputs.append(f"pdf_chunks_created={result.created} pdf_chunks_skipped={result.skipped}")
    if modality in {"all", "audio"}:
        result = chunk_audio(
            config,
            source_id=source_id,
            previous_utterances=previous_utterances,
            max_tokens=max_tokens,
        )
        _record_chunk_job(
            config,
            "chunk_audio",
            source_id,
            result,
            CHUNKER_VERSIONS["audio"],
            "evidence",
            {"previous_utterances": previous_utterances, "max_tokens": max_tokens},
        )
        outputs.append(f"audio_chunks_created={result.created} audio_chunks_skipped={result.skipped}")
    if modality in {"all", "image"}:
        result = chunk_image_ocr(config, source_id=source_id)
        _record_chunk_job(
            config,
            "chunk_image_ocr",
            source_id,
            result,
            CHUNKER_VERSIONS["image_ocr"],
            "evidence",
        )
        outputs.append(f"image_ocr_chunks_created={result.created} image_ocr_chunks_skipped={result.skipped}")
    typer.echo(" ".join(outputs))


@app.command("detect-spans")
def detect_spans_command(
    modality: str = typer.Option("all", "--modality", help="Modality to detect: all, chat, pdf, audio, or image."),
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
        _record_span_detection_job(
            config,
            "detect_chat_spans",
            source_id,
            result,
            SPAN_DETECTOR_VERSIONS["chat"],
            "chat",
        )
        outputs.append(f"chat_spans_created={result.created} chat_spans_skipped={result.skipped}")
    if modality in {"all", "pdf"}:
        result = detect_pdf_spans(config, source_id=source_id)
        _record_span_detection_job(
            config,
            "detect_pdf_spans",
            source_id,
            result,
            SPAN_DETECTOR_VERSIONS["pdf"],
            "pdf",
        )
        outputs.append(f"pdf_spans_created={result.created} pdf_spans_skipped={result.skipped}")
    if modality in {"all", "audio"}:
        result = detect_audio_spans(config, source_id=source_id)
        _record_span_detection_job(
            config,
            "detect_audio_spans",
            source_id,
            result,
            SPAN_DETECTOR_VERSIONS["audio"],
            "audio",
        )
        outputs.append(f"audio_spans_created={result.created} audio_spans_skipped={result.skipped}")
    if modality in {"all", "image"}:
        result = detect_image_ocr_spans(config, source_id=source_id)
        _record_span_detection_job(
            config,
            "detect_image_ocr_spans",
            source_id,
            result,
            SPAN_DETECTOR_VERSIONS["image_ocr"],
            "image",
        )
        outputs.append(f"image_ocr_spans_created={result.created} image_ocr_spans_skipped={result.skipped}")
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
    _record_ingest_chat_job(config, chat_export, ingest_result)
    evidence_result = build_chat_evidence(config, source_id=ingest_result.source_id)
    _record_evidence_job(
        config,
        "build_chat_evidence",
        ingest_result.source_id,
        evidence_result,
        EVIDENCE_BUILDER_VERSIONS["chat"],
    )
    chunk_result = chunk_chat(config, source_id=ingest_result.source_id, previous_messages=previous_messages)
    _record_chunk_job(
        config,
        "chunk_chat",
        ingest_result.source_id,
        chunk_result,
        CHUNKER_VERSIONS["chat"],
        "evidence",
        {"previous_messages": previous_messages, "max_tokens": 1200},
    )
    span_result = detect_chat_spans(config, source_id=ingest_result.source_id)
    _record_span_detection_job(
        config,
        "detect_chat_spans",
        ingest_result.source_id,
        span_result,
        SPAN_DETECTOR_VERSIONS["chat"],
        "chat",
    )
    extract_result = extract_claims_from_spans(config, modality="chat", source_id=ingest_result.source_id)
    validation_result = validate_raw_claims(config, source_id=ingest_result.source_id)
    normalization_result = normalize_claims(config, source_id=ingest_result.source_id)
    _record_extract_claims_job(config, "chat", ingest_result.source_id, extract_result)
    _record_validate_claims_job(config, ingest_result.source_id, None, validation_result)
    _record_normalize_claims_job(config, ingest_result.source_id, None, normalization_result)
    graph_result = export_graph_jsonl(config)
    _record_graph_export_job(config, graph_result)
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
    _record_ingest_pdf_job(config, pdf_file, ingest_result)
    evidence_result = build_pdf_evidence(config, source_id=ingest_result.source_id)
    _record_evidence_job(
        config,
        "build_pdf_evidence",
        ingest_result.source_id,
        evidence_result,
        EVIDENCE_BUILDER_VERSIONS["pdf"],
    )
    chunk_result = chunk_pdf(config, source_id=ingest_result.source_id, target_tokens=target_tokens)
    _record_chunk_job(
        config,
        "chunk_pdf",
        ingest_result.source_id,
        chunk_result,
        CHUNKER_VERSIONS["pdf"],
        "evidence",
        {"target_tokens": target_tokens, "overlap_tokens": 150},
    )
    span_result = detect_pdf_spans(config, source_id=ingest_result.source_id)
    _record_span_detection_job(
        config,
        "detect_pdf_spans",
        ingest_result.source_id,
        span_result,
        SPAN_DETECTOR_VERSIONS["pdf"],
        "pdf",
    )
    extract_result = extract_claims_from_spans(config, modality="pdf", source_id=ingest_result.source_id)
    validation_result = validate_raw_claims(config, source_id=ingest_result.source_id)
    normalization_result = normalize_claims(config, source_id=ingest_result.source_id)
    _record_extract_claims_job(config, "pdf", ingest_result.source_id, extract_result)
    _record_validate_claims_job(config, ingest_result.source_id, None, validation_result)
    _record_normalize_claims_job(config, ingest_result.source_id, None, normalization_result)
    graph_result = export_graph_jsonl(config)
    _record_graph_export_job(config, graph_result)
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
    _record_ingest_audio_job(config, transcript_file, ingest_result)
    evidence_result = build_audio_evidence(config, source_id=ingest_result.source_id)
    _record_evidence_job(
        config,
        "build_audio_evidence",
        ingest_result.source_id,
        evidence_result,
        EVIDENCE_BUILDER_VERSIONS["audio"],
    )
    chunk_result = chunk_audio(config, source_id=ingest_result.source_id, previous_utterances=previous_utterances)
    _record_chunk_job(
        config,
        "chunk_audio",
        ingest_result.source_id,
        chunk_result,
        CHUNKER_VERSIONS["audio"],
        "evidence",
        {"previous_utterances": previous_utterances, "max_tokens": 1200},
    )
    span_result = detect_audio_spans(config, source_id=ingest_result.source_id)
    _record_span_detection_job(
        config,
        "detect_audio_spans",
        ingest_result.source_id,
        span_result,
        SPAN_DETECTOR_VERSIONS["audio"],
        "audio",
    )
    extract_result = extract_claims_from_spans(config, modality="audio", source_id=ingest_result.source_id)
    validation_result = validate_raw_claims(config, source_id=ingest_result.source_id)
    normalization_result = normalize_claims(config, source_id=ingest_result.source_id)
    _record_extract_claims_job(config, "audio", ingest_result.source_id, extract_result)
    _record_validate_claims_job(config, ingest_result.source_id, None, validation_result)
    _record_normalize_claims_job(config, ingest_result.source_id, None, normalization_result)
    graph_result = export_graph_jsonl(config)
    _record_graph_export_job(config, graph_result)
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
    _record_ingest_images_job(config, image_path, ingest_result)
    regions_created = 0
    evidence_created = 0
    embeddings_created = 0
    clusters_created = 0
    claims_created = 0
    claims_accepted = 0
    claims_normalized = 0
    for source_id in ingest_result.source_ids:
        region_result = propose_image_regions(config, source_id=source_id, patch_size=patch_size, stride=stride)
        _record_image_region_proposal_job(config, source_id, region_result, patch_size, stride)
        evidence_result = build_image_evidence(config, source_id=source_id)
        _record_evidence_job(
            config,
            "build_image_evidence",
            source_id,
            evidence_result,
            EVIDENCE_BUILDER_VERSIONS["image_region"],
        )
        embedding_result = build_image_region_embeddings(config, source_id=source_id)
        _record_image_embedding_job(config, source_id, embedding_result, COLOR_EMBEDDING_MODEL)
        cluster_result = cluster_image_regions(
            config,
            source_id=source_id,
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size,
        )
        _record_image_clustering_job(
            config,
            source_id,
            cluster_result,
            COLOR_EMBEDDING_MODEL,
            distance_threshold,
            min_cluster_size,
        )
        cluster_evidence_result = build_image_cluster_evidence(config, source_id=source_id)
        _record_evidence_job(
            config,
            "build_image_cluster_evidence",
            source_id,
            cluster_evidence_result,
            EVIDENCE_BUILDER_VERSIONS["image_cluster"],
        )
        extract_result = extract_claims_from_spans(config, modality="image", source_id=source_id)
        validation_result = validate_raw_claims(config, source_id=source_id)
        normalization_result = normalize_claims(config, source_id=source_id)
        _record_extract_claims_job(config, "image", source_id, extract_result)
        _record_validate_claims_job(config, source_id, None, validation_result)
        _record_normalize_claims_job(config, source_id, None, normalization_result)
        regions_created += region_result.created
        evidence_created += evidence_result.created + cluster_evidence_result.created
        embeddings_created += embedding_result.created
        clusters_created += cluster_result.created
        claims_created += extract_result.created
        claims_accepted += validation_result.accepted
        claims_normalized += normalization_result.created
    graph_result = export_graph_jsonl(config)
    _record_graph_export_job(config, graph_result)
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
    _record_validate_claims_job(config, source_id, claim_id, result)
    typer.echo(
        f"claims_accepted={result.accepted} claims_quarantined={result.quarantined} "
        f"claims_skipped={result.skipped}"
    )


@app.command("extract-claims")
def extract_claims_command(
    modality: str = typer.Option("all", "--modality", help="Modality to extract: all, chat, pdf, audio, or image."),
    source_id: Optional[str] = typer.Option(None, "--source-id", help="Only extract claims for this source."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Append extracted claims in batches of N."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Extract source-faithful raw claims from detected spans using the baseline rules extractor."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = extract_claims_from_spans(
            config,
            modality=modality,
            source_id=source_id,
            batch_size=batch_size,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    _record_extract_claims_job(config, modality, source_id, result, batch_size=batch_size)
    message = f"claims_created={result.created} claims_skipped={result.skipped}"
    if batch_size is not None:
        message = f"{message} batches_processed={result.batches_processed}"
    typer.echo(message)


@app.command("import-raw-claims")
def import_raw_claims_command(
    input_path: Path = typer.Argument(..., help="JSON or JSONL file of raw claim candidates."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Import raw claim candidates after narrow schema repair and validation."""
    config = load_config(config_path)
    _init_paths(config)
    if not input_path.exists() or not input_path.is_file():
        raise typer.BadParameter(f"input path does not exist: {input_path}")
    try:
        result = import_raw_claim_candidates(config, input_path)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    record_job_result(
        config,
        stage="import_raw_claims",
        input_record_ids=[str(input_path)],
        model_id=SCHEMA_REPAIR_VERSION,
        metrics={
            "claims_imported": result.imported,
            "claims_repaired": result.repaired,
            "claims_quarantined": result.quarantined,
            "claims_skipped": result.skipped,
        },
        metadata={"input_path": str(input_path)},
    )
    typer.echo(
        f"claims_imported={result.imported} claims_repaired={result.repaired} "
        f"claims_quarantined={result.quarantined} claims_skipped={result.skipped}"
    )


@app.command("report")
def report_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Report output path."),
    format: str = typer.Option("markdown", "--format", help="Report format: markdown or html."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write an extraction summary report."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = write_summary_report(config, output_path=output, output_format=format)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(str(result.output_path))


@app.command("acceptance-check")
def acceptance_check_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Acceptance check JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write and enforce the definition-of-done acceptance check report."""
    config = load_config(config_path)
    _init_paths(config)
    result = write_acceptance_report(config, output_path=output)
    failed_checks = _record_acceptance_check_job(config, result)
    typer.echo(f"{result.output_path} checks={len(result.checks)} failed_checks={failed_checks} passed={result.passed}")
    if not result.passed:
        raise typer.Exit(code=1)


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
    record_job_result(
        config,
        stage="route_models",
        input_record_ids=[f"models_config:{models_config}", f"stage:{stage}"],
        model_id=MODEL_ROUTING_VERSION,
        metrics={"recommendations": result.recommendation_count},
        metadata={
            "models_config": str(models_config),
            "output_path": str(result.output_path),
            "stage": stage,
        },
    )
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
    _record_normalize_claims_job(config, source_id, claim_id, result)
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
    _record_graph_export_job(config, result, output_format=format)
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
    _record_sqlite_export_job(config, result)
    typer.echo(
        f"{result.output_path} tables={len(result.table_counts)} "
        f"records={sum(result.table_counts.values())}"
    )


@app.command("finalize-run")
def finalize_run_command(
    gold_file: Optional[Path] = typer.Option(None, "--gold", help="Optional gold claims JSON file to evaluate."),
    validate_outputs: bool = typer.Option(
        True,
        "--validate-artifacts/--no-validate-artifacts",
        help="Validate core and report JSONL artifacts before exporting SQLite.",
    ),
    sqlite: bool = typer.Option(True, "--sqlite/--no-sqlite", help="Also export a SQLite snapshot."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Produce final graph, optional gold eval, reports, acceptance checks, and SQLite snapshot."""
    config = load_config(config_path)
    _init_paths(config)
    if gold_file is not None and (not gold_file.exists() or not gold_file.is_file()):
        raise typer.BadParameter(f"gold file does not exist: {gold_file}")
    graph_result = export_graph_jsonl(config)
    _record_graph_export_job(config, graph_result)
    gold_result = None
    if gold_file is not None:
        try:
            gold_result = write_gold_eval_report(config, gold_file)
        except ValueError as exc:
            raise typer.BadParameter(str(exc))
        _record_gold_eval_job(config, gold_file, gold_result)
    write_summary_report(config)
    acceptance_result = write_acceptance_report(config)
    failed_checks = _record_acceptance_check_job(config, acceptance_result)
    summary_result = write_summary_report(config)
    artifact_validation_result = validate_known_artifacts(config, include_reports=True) if validate_outputs else None
    if artifact_validation_result is not None:
        _echo_artifact_validation_result(artifact_validation_result, verbose=False)
    sqlite_result = (
        export_sqlite(config)
        if sqlite and (artifact_validation_result is None or artifact_validation_result.failures == 0)
        else None
    )

    message = (
        f"graph={graph_result.output_path} graph_edges={graph_result.edge_count} "
        f"report={summary_result.output_path} acceptance={acceptance_result.output_path} "
        f"failed_checks={failed_checks} passed={acceptance_result.passed}"
    )
    if gold_result is not None:
        message = (
            f"{message} gold_eval={gold_result.output_path} "
            f"gold_claims={gold_result.metrics['gold_claims']}"
        )
    if artifact_validation_result is not None:
        message = (
            f"{message} artifact_paths={len(artifact_validation_result.files)} "
            f"artifact_records={artifact_validation_result.records_checked} "
            f"artifact_failures={artifact_validation_result.failures}"
        )
    if sqlite_result is not None:
        _record_sqlite_export_job(config, sqlite_result)
        message = (
            f"{message} sqlite={sqlite_result.output_path} "
            f"sqlite_tables={len(sqlite_result.table_counts)} "
            f"sqlite_records={sum(sqlite_result.table_counts.values())}"
        )
    typer.echo(message)
    if not acceptance_result.passed or (
        artifact_validation_result is not None and artifact_validation_result.failures
    ):
        raise typer.Exit(code=1)


@app.command("export-metta")
def export_metta_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="MeTTa output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Export normalized claims as MeTTa-style S-expressions."""
    config = load_config(config_path)
    _init_paths(config)
    result = export_metta(config, output_path=output)
    record_job_result(
        config,
        stage="export_metta",
        input_record_ids=["claims_normalized"],
        model_id=METTA_EXPORT_VERSION,
        metrics={"claims": result.claim_count},
        metadata={"output_path": str(result.output_path)},
    )
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
    _record_gold_eval_job(config, gold_file, result)
    typer.echo(
        f"{result.output_path} accepted_precision={result.metrics['accepted_precision']} "
        f"accepted_recall={result.metrics['accepted_recall']} "
        f"quarantine_precision={result.metrics['quarantine_precision']} "
        f"quarantine_recall={result.metrics['quarantine_recall']} "
        f"evidence_exact_match_rate={result.metrics['evidence_exact_match_rate']}"
    )


@app.command("trace-claim")
def trace_claim_command(
    claim_id: str = typer.Argument(..., help="Claim ID to trace."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional JSON output path."),
    format: str = typer.Option("json", "--format", help="Trace format: json or html."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Trace a claim back through source, evidence, span, validation, and normalization artifacts."""
    config = load_config(config_path)
    _init_paths(config)
    normalized_format = format.strip().lower()
    if normalized_format not in {"json", "html"}:
        raise typer.BadParameter("trace format must be json or html")
    if normalized_format == "html":
        output_path = output or default_claim_trace_html_path(config, claim_id)
        trace = write_claim_trace_html(config, claim_id, output_path)
        typer.echo(f"{output_path} found={trace['found']}")
        return
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
    corrected_claim: Optional[str] = typer.Option(None, "--corrected-claim", help="Optional corrected claim text."),
    normalized_claim_json: Optional[str] = typer.Option(
        None,
        "--normalized-claim-json",
        help="Optional corrected normalized claim JSON object.",
    ),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Record a human review decision for a claim."""
    config = load_config(config_path)
    _init_paths(config)
    review_metadata = {}
    if corrected_claim is not None:
        review_metadata["corrected_source_faithful_claim"] = corrected_claim
    if normalized_claim_json is not None:
        try:
            normalized_claim = json.loads(normalized_claim_json)
        except json.JSONDecodeError as exc:
            raise typer.BadParameter(f"normalized claim JSON must be valid JSON: {exc.msg}")
        if not isinstance(normalized_claim, dict):
            raise typer.BadParameter("normalized claim JSON must be a JSON object")
        review_metadata["corrected_normalized_claim"] = normalized_claim
    try:
        result = record_claim_review(
            config,
            claim_id=claim_id,
            decision=decision,
            reviewer_id=reviewer_id,
            reason_codes=reason_code,
            notes=notes,
            metadata=review_metadata or None,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    typer.echo(f"review_id={result.review_id} created={result.created}")


@app.command("review-queue")
def review_queue_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Optional output path."),
    include_reviewed: bool = typer.Option(False, "--include-reviewed", help="Include already reviewed claims."),
    format: str = typer.Option("jsonl", "--format", help="Review queue format: jsonl or html."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write reviewable claim packets from validation and evidence artifacts."""
    config = load_config(config_path)
    _init_paths(config)
    try:
        result = write_review_queue(
            config,
            output_path=output,
            include_reviewed=include_reviewed,
            output_format=format,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc))
    record_job_result(
        config,
        stage="review_queue",
        input_record_ids=[
            "claims_normalized",
            "claims_raw",
            f"format:{format}",
            f"include_reviewed:{include_reviewed}",
            "review_decisions",
            "validations",
        ],
        model_id=REVIEW_QUEUE_VERSION,
        metrics={"review_items": result.item_count},
        metadata={
            "format": format,
            "include_reviewed": include_reviewed,
            "output_path": str(result.output_path),
        },
    )
    typer.echo(f"{result.output_path} review_items={result.item_count}")


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
    record_job_result(
        config,
        stage="dedupe_claims",
        input_record_ids=["claims_normalized"],
        model_id=DEDUPE_VERSION,
        metrics={"groups": result.group_count},
        metadata={"include_singletons": include_singletons, "output_path": str(result.output_path)},
    )
    typer.echo(f"{result.output_path} groups={result.group_count}")


@app.command("repair-claims")
def repair_claims_command(
    only: Optional[List[str]] = typer.Option(None, "--only", help="Only suggest repairs for this reason code. Repeatable."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Repair suggestion JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write reviewable evidence_text repair suggestions for raw claims."""
    config = load_config(config_path)
    _init_paths(config)
    unknown_reasons = sorted(set(only or []) - REPAIR_REASON_CODES)
    if unknown_reasons:
        supported = ", ".join(sorted(REPAIR_REASON_CODES))
        requested = ", ".join(unknown_reasons)
        raise typer.BadParameter(f"repair-claims supports reason codes: {supported}; unsupported: {requested}")
    result = suggest_evidence_repairs(config, output_path=output, only_reason_codes=only)
    record_job_result(
        config,
        stage="repair_claims",
        input_record_ids=["claims_raw", "evidence", "spans"],
        model_id=REPAIR_SUGGESTION_VERSION,
        metrics={"suggestions": result.suggestion_count},
        metadata={"only_reason_codes": sorted(only or []), "output_path": str(result.output_path)},
    )
    typer.echo(f"{result.output_path} suggestions={result.suggestion_count}")


@app.command("apply-repairs")
def apply_repairs_command(
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="Repair suggestion JSONL input path."),
    repair_id: Optional[List[str]] = typer.Option(None, "--repair-id", help="Only apply the selected repair ID. Repeatable."),
    actor_id: Optional[str] = typer.Option(None, "--actor-id", help="Optional actor applying repairs."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Apply exact evidence_text repair suggestions as new raw claims."""
    config = load_config(config_path)
    _init_paths(config)
    if input_path is not None and (not input_path.exists() or not input_path.is_file()):
        raise typer.BadParameter(f"repair input does not exist: {input_path}")
    result = apply_evidence_repairs(
        config,
        input_path=input_path,
        repair_ids=repair_id,
        actor_id=actor_id,
    )
    effective_input_path = input_path or config.paths.reports_dir / "claim_repairs.jsonl"
    input_ids = result.claim_ids or ["claims_raw"]
    input_ids.append(f"repairs:{effective_input_path}")
    input_ids.extend(f"repair_id:{value}" for value in repair_id or [])
    record_job_result(
        config,
        stage="apply_repairs",
        source_id=_single_source_id(result.source_ids),
        input_record_ids=input_ids,
        model_id=REPAIR_APPLICATION_VERSION,
        metrics={"repairs_applied": result.applied, "repairs_skipped": result.skipped, "repairs_failed": result.failed},
        metadata={"actor_id": actor_id, "input_path": str(effective_input_path)},
    )
    typer.echo(
        f"repairs_applied={result.applied} repairs_skipped={result.skipped} "
        f"repairs_failed={result.failed}"
    )


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
    record_job_result(
        config,
        stage="detect_pii",
        input_record_ids=[f"artifact:{artifact}"],
        model_id=PII_PROCESSOR_VERSION,
        metrics={"findings": result.finding_count},
        metadata={"artifact": artifact, "output_path": str(result.output_path)},
    )
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
    record_job_result(
        config,
        stage="redact_pii",
        input_record_ids=[f"artifact:{artifact}"],
        model_id=PII_PROCESSOR_VERSION,
        metrics={
            "records_written": result.records_written,
            "replacements": result.replacement_count,
            "redactions": result.redaction_count,
        },
        metadata={
            "artifact": artifact,
            "output_path": str(result.output_path),
            "manifest_path": str(result.manifest_path),
        },
    )
    typer.echo(
        f"{result.output_path} records={result.records_written} "
        f"replacements={result.replacement_count} redactions={result.redaction_count} "
        f"manifest={result.manifest_path}"
    )


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
    record_job_result(
        config,
        stage="check_privacy",
        input_record_ids=["claims_raw", "sources"],
        model_id=PRIVACY_CHECK_VERSION,
        metrics={
            "claims_checked": result.claims_checked,
            "violations": result.violation_count,
        },
        metadata={"output_path": str(result.output_path), "fail_on_violation": fail_on_violation},
    )
    typer.echo(
        f"{result.output_path} claims_checked={result.claims_checked} "
        f"violations={result.violation_count}"
    )
    if fail_on_violation and result.violation_count:
        raise typer.Exit(code=1)


@app.command("retention-plan")
def retention_plan_command(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Retention plan JSONL output path."),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Write a dry-run raw source retention plan."""
    config = load_config(config_path)
    _init_paths(config)
    result = write_retention_plan(config, output_path=output)
    record_job_result(
        config,
        stage="retention_plan",
        input_record_ids=["sources"],
        model_id=RETENTION_PLAN_VERSION,
        metrics={"candidates": result.candidate_count},
        metadata={"output_path": str(result.output_path)},
    )
    typer.echo(f"{result.output_path} candidates={result.candidate_count}")


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
    include_reports: bool = typer.Option(
        False,
        "--include-reports",
        help="Also validate known report JSONLs with registered schemas.",
    ),
    config_path: Path = typer.Option(Path("configs/pipeline.yaml"), "--config", help="Pipeline config path."),
) -> None:
    """Validate known JSONL artifacts that exist in the configured data directory."""
    config = load_config(config_path)
    result = validate_known_artifacts(config, include_reports=include_reports)
    _echo_artifact_validation_result(result)
    if result.failures:
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
