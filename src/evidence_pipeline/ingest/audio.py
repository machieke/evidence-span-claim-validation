from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl, write_jsonl
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.sources import SourceRecord

AUDIO_NORMALIZATION_VERSION = "audio.normalization.ffmpeg_plan.v1"


@dataclass
class AudioNormalizationResult:
    source_id: str
    source_created: bool
    source_updated: bool
    normalized_file: Path
    command: List[str]
    normalization_policy_id: str
    executed: bool


@dataclass
class AudioTranscriptIngestResult:
    source_id: str
    source_created: bool
    utterances_created: int
    utterances_skipped: int


def _load_transcript(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    defaults: Dict[str, Any] = {}
    if isinstance(payload, list):
        utterances = payload
    elif isinstance(payload, dict) and isinstance(payload.get("utterances"), list):
        utterances = payload["utterances"]
        defaults = {key: value for key, value in payload.items() if key != "utterances"}
    else:
        raise ValueError("audio transcript must be a JSON list or an object with an utterances list")
    for utterance in utterances:
        if not isinstance(utterance, dict):
            raise ValueError("every utterance must be a JSON object")
    return utterances, defaults


def _sample_rate_label(sample_rate: int) -> str:
    if sample_rate % 1000 == 0:
        return f"{sample_rate // 1000}khz"
    return f"{sample_rate}hz"


def _channel_label(channels: int) -> str:
    if channels == 1:
        return "mono"
    if channels == 2:
        return "stereo"
    return f"{channels}ch"


def _default_normalized_audio_path(
    config: PipelineConfig,
    path: Path,
    sample_rate: int,
    channels: int,
) -> Path:
    target = f"{_sample_rate_label(sample_rate)}_{_channel_label(channels)}"
    return config.paths.work_dir / "normalized_audio" / f"{path.stem}_{target}.wav"


def _normalization_command(
    path: Path,
    normalized_file: Path,
    sample_rate: int,
    channels: int,
) -> List[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(path),
        "-ac",
        str(channels),
        "-ar",
        str(sample_rate),
        str(normalized_file),
    ]


def _normalization_policy_id(normalized_file: Path, sample_rate: int, channels: int) -> str:
    return stable_id(
        "audio_norm",
        {
            "normalizer": AUDIO_NORMALIZATION_VERSION,
            "normalized_file": str(normalized_file),
            "sample_rate": sample_rate,
            "channels": channels,
        },
        length=16,
    )


def _normalization_metadata(
    normalized_file: Path,
    command: List[str],
    sample_rate: int,
    channels: int,
    execute: bool,
    normalization_policy_id: str,
) -> Dict[str, Any]:
    return {
        "normalization_policy_id": normalization_policy_id,
        "normalized_file": str(normalized_file),
        "normalization_status": "created" if execute else "planned",
        "normalization_command": command,
        "target_sample_rate": sample_rate,
        "target_channels": channels,
        "normalizer": AUDIO_NORMALIZATION_VERSION,
    }


def _merge_audio_normalizations(
    existing: List[Dict[str, Any]],
    current: Dict[str, Any],
) -> List[Dict[str, Any]]:
    current_policy_id = current["normalization_policy_id"]
    merged = []
    replaced = False
    for item in existing:
        if not isinstance(item, dict):
            continue
        if item.get("normalization_policy_id") == current_policy_id:
            merged.append(current)
            replaced = True
        else:
            merged.append(item)
    if not replaced:
        merged.append(current)
    return merged


def _effective_normalization_metadata(
    existing: List[Dict[str, Any]],
    current: Dict[str, Any],
) -> Dict[str, Any]:
    current_policy_id = current["normalization_policy_id"]
    for item in existing:
        if not isinstance(item, dict):
            continue
        if item.get("normalization_policy_id") != current_policy_id:
            continue
        if item.get("normalization_status") == "created" and current.get("normalization_status") == "planned":
            return item
    return current


def _upsert_audio_source(
    path: Path,
    config: PipelineConfig,
    sha256: str,
    source_id: str,
    source_metadata: Dict[str, Any],
    metadata: Optional[Dict[str, Any]],
) -> Tuple[bool, bool]:
    paths = config.jsonl_paths()
    records = [payload for _, payload in read_jsonl(paths["sources"])]
    source_created = True
    source_updated = False
    next_records: List[Dict[str, Any]] = []
    for payload in records:
        if payload.get("source_id") != source_id:
            next_records.append(payload)
            continue
        source_created = False
        existing_metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        normalizations = existing_metadata.get("audio_normalizations")
        normalizations = normalizations if isinstance(normalizations, list) else []
        effective_source_metadata = _effective_normalization_metadata(normalizations, source_metadata)
        updated_metadata = {
            **existing_metadata,
            **(metadata or {}),
            "media_kind": "audio_source",
            **effective_source_metadata,
            "audio_normalizations": _merge_audio_normalizations(normalizations, effective_source_metadata),
        }
        updated_payload = {
            **payload,
            "source_modality": "audio",
            "source_file": str(path),
            "sha256": sha256,
            "metadata": updated_metadata,
        }
        next_records.append(updated_payload)
        source_updated = updated_payload != payload
    if source_created:
        append_jsonl(
            paths["sources"],
            SourceRecord(
                source_id=source_id,
                source_modality="audio",
                source_file=str(path),
                sha256=sha256,
                metadata={
                    **(metadata or {}),
                    **source_metadata,
                    "media_kind": "audio_source",
                    "audio_normalizations": [source_metadata],
                },
            ),
        )
        return True, False
    if source_updated:
        write_jsonl(paths["sources"], next_records)
    return False, source_updated


def normalize_audio_source(
    path: Path,
    config: PipelineConfig,
    normalized_file: Optional[Path] = None,
    sample_rate: int = 16000,
    channels: int = 1,
    execute: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> AudioNormalizationResult:
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if channels < 1:
        raise ValueError("channels must be positive")
    if not path.exists() or not path.is_file():
        raise ValueError(f"audio media file does not exist: {path}")

    normalized_file = normalized_file or _default_normalized_audio_path(config, path, sample_rate, channels)
    if normalized_file.resolve(strict=False) == path.resolve(strict=False):
        raise ValueError("normalized audio output path must differ from the input path")
    command = _normalization_command(path, normalized_file, sample_rate, channels)
    sha256 = sha256_file(path)
    source_id = stable_id("src", {"modality": "audio", "sha256": sha256})
    normalization_policy_id = _normalization_policy_id(normalized_file, sample_rate, channels)
    source_metadata = _normalization_metadata(
        normalized_file,
        command,
        sample_rate,
        channels,
        execute,
        normalization_policy_id,
    )
    if execute:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is required when --execute is set")
        normalized_file.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True)

    source_created, source_updated = _upsert_audio_source(
        path,
        config,
        sha256,
        source_id,
        source_metadata,
        metadata,
    )

    return AudioNormalizationResult(
        source_id=source_id,
        source_created=source_created,
        source_updated=source_updated,
        normalized_file=normalized_file,
        command=command,
        normalization_policy_id=normalization_policy_id,
        executed=execute,
    )


def _first_present(record: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _coerce_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


def _risk_flags(record: Dict[str, Any]) -> List[str]:
    flags = _coerce_list(record.get("risk_flags"))
    asr_confidence = record.get("asr_confidence")
    diarization_confidence = record.get("diarization_confidence")
    if asr_confidence is not None and float(asr_confidence) < 0.55 and "low_asr_confidence" not in flags:
        flags.append("low_asr_confidence")
    if diarization_confidence is not None and float(diarization_confidence) < 0.50 and "speaker_uncertain" not in flags:
        flags.append("speaker_uncertain")
    return flags


def _utterance_id(source_id: str, index: int, utterance: Dict[str, Any]) -> str:
    raw_id = _first_present(utterance, ["utterance_id", "id"])
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)
    return stable_id(
        "utt",
        {
            "source_id": source_id,
            "index": index,
            "speaker": _first_present(utterance, ["speaker", "speaker_label"], "SPEAKER_UNKNOWN"),
            "start": _first_present(utterance, ["start", "start_seconds"], 0),
            "end": _first_present(utterance, ["end", "end_seconds"], 0),
            "text": _first_present(utterance, ["text", "transcript"], ""),
        },
    )


def _build_utterance(source_id: str, index: int, utterance: Dict[str, Any], defaults: Dict[str, Any]) -> AudioUtteranceRecord:
    return AudioUtteranceRecord(
        utterance_id=_utterance_id(source_id, index, utterance),
        source_id=source_id,
        speaker=str(_first_present(utterance, ["speaker", "speaker_label"], "SPEAKER_UNKNOWN")),
        start=float(_first_present(utterance, ["start", "start_seconds"], 0)),
        end=float(_first_present(utterance, ["end", "end_seconds"], 0)),
        text=str(_first_present(utterance, ["text", "transcript"], "")),
        asr_segment_ids=_coerce_list(utterance.get("asr_segment_ids")),
        turn_ids=_coerce_list(utterance.get("turn_ids")),
        asr_confidence=utterance.get("asr_confidence"),
        diarization_confidence=utterance.get("diarization_confidence"),
        language=_first_present(utterance, ["language"], defaults.get("language")),
        metadata=utterance.get("metadata") if isinstance(utterance.get("metadata"), dict) else {},
        risk_flags=_risk_flags(utterance),
    )


def _mark_overlapping_speech(records: List[AudioUtteranceRecord]) -> List[AudioUtteranceRecord]:
    overlapping_ids = set()
    sorted_records = sorted(records, key=lambda record: (record.start, record.end, record.utterance_id))
    for index, current in enumerate(sorted_records):
        for other in sorted_records[index + 1:]:
            if other.start >= current.end:
                break
            if other.end > current.start:
                overlapping_ids.add(current.utterance_id)
                overlapping_ids.add(other.utterance_id)

    marked = []
    for record in records:
        if record.utterance_id not in overlapping_ids:
            marked.append(record)
            continue
        risk_flags = sorted(set(record.risk_flags) | {"overlapping_speech"})
        marked.append(record.model_copy(update={"risk_flags": risk_flags}))
    return marked


def ingest_audio_transcript(path: Path, config: PipelineConfig, metadata: Optional[Dict[str, Any]] = None) -> AudioTranscriptIngestResult:
    utterances, defaults = _load_transcript(path)
    metadata = dict(metadata or {})
    if isinstance(defaults.get("metadata"), dict):
        merged = dict(defaults["metadata"])
        merged.update(metadata)
        metadata = merged

    sha256 = sha256_file(path)
    source_id = stable_id("src", {"modality": "audio", "sha256": sha256})
    paths = config.jsonl_paths()
    records = [
        _build_utterance(source_id, index, utterance, defaults)
        for index, utterance in enumerate(utterances)
    ]
    duration_seconds = max((record.end for record in records), default=0.0)
    existing_sources = existing_values(paths["sources"], "source_id")
    source_created = source_id not in existing_sources
    if source_created:
        append_jsonl(
            paths["sources"],
            SourceRecord(
                source_id=source_id,
                source_modality="audio",
                source_file=str(defaults.get("source_file") or path),
                sha256=sha256,
                metadata={
                    **metadata,
                    "duration_seconds": duration_seconds,
                    "utterance_count": len(utterances),
                    "transcript_file": str(path),
                },
            ),
        )

    existing_utterance_ids = existing_values(paths["audio_utterances"], "utterance_id")
    created = 0
    skipped = 0
    for record in _mark_overlapping_speech(records):
        if record.utterance_id in existing_utterance_ids:
            skipped += 1
            continue
        append_jsonl(paths["audio_utterances"], record)
        existing_utterance_ids.add(record.utterance_id)
        created += 1

    return AudioTranscriptIngestResult(
        source_id=source_id,
        source_created=source_created,
        utterances_created=created,
        utterances_skipped=skipped,
    )
