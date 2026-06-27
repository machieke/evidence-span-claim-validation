from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


@dataclass
class AudioChunkResult:
    created: int
    skipped: int


def _load_utterance_evidence(config: PipelineConfig) -> Dict[str, EvidenceRecord]:
    paths = config.jsonl_paths()
    evidence_by_utterance_id: Dict[str, EvidenceRecord] = {}
    for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord):
        if evidence.source_modality != "audio" or evidence.evidence_type != "utterance_span":
            continue
        utterance_id = evidence.provenance.get("utterance_id")
        if isinstance(utterance_id, str):
            evidence_by_utterance_id[utterance_id] = evidence
    return evidence_by_utterance_id


def _sort_key(utterance: AudioUtteranceRecord) -> Tuple[float, float, str]:
    return utterance.start, utterance.end, utterance.utterance_id


def _format_utterance(utterance: AudioUtteranceRecord) -> str:
    return f"{utterance.speaker}: {utterance.text}"


def chunk_audio(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    previous_utterances: int = 1,
    max_tokens: int = 1200,
) -> AudioChunkResult:
    paths = config.jsonl_paths()
    evidence_by_utterance_id = _load_utterance_evidence(config)
    utterances = [
        utterance
        for _, utterance in read_jsonl_records(paths["audio_utterances"], AudioUtteranceRecord)
        if source_id is None or utterance.source_id == source_id
    ]
    grouped: Dict[str, List[AudioUtteranceRecord]] = {}
    for utterance in utterances:
        grouped.setdefault(utterance.source_id, []).append(utterance)
    for source_utterances in grouped.values():
        source_utterances.sort(key=_sort_key)

    existing_chunk_ids = existing_values(paths["chunks"], "chunk_id")
    created = 0
    skipped = 0

    for source_utterances in grouped.values():
        for index, utterance in enumerate(source_utterances):
            primary_evidence = evidence_by_utterance_id.get(utterance.utterance_id)
            if primary_evidence is None:
                continue
            start = max(0, index - previous_utterances)
            context_utterances = source_utterances[start : index + 1]
            evidence_ids = [
                evidence_by_utterance_id[item.utterance_id].evidence_id
                for item in context_utterances
                if item.utterance_id in evidence_by_utterance_id
            ]
            overlap_ids = [
                evidence_by_utterance_id[item.utterance_id].evidence_id
                for item in context_utterances[:-1]
                if item.utterance_id in evidence_by_utterance_id
            ]
            chunk_id = stable_id(
                "chunk_audio",
                {
                    "primary_evidence_id": primary_evidence.evidence_id,
                    "previous_utterances": previous_utterances,
                    "max_tokens": max_tokens,
                },
            )
            if chunk_id in existing_chunk_ids:
                skipped += 1
                continue
            append_jsonl(
                paths["chunks"],
                ChunkRecord(
                    chunk_id=chunk_id,
                    source_id=utterance.source_id,
                    source_modality="audio",
                    evidence_ids=evidence_ids,
                    primary_evidence_ids=[primary_evidence.evidence_id],
                    overlap_evidence_ids=overlap_ids,
                    text="\n".join(_format_utterance(item) for item in context_utterances),
                    provenance_summary={
                        "utterance_ids": [item.utterance_id for item in context_utterances],
                        "primary_utterance_id": utterance.utterance_id,
                        "start": context_utterances[0].start,
                        "end": context_utterances[-1].end,
                        "speakers": sorted({item.speaker for item in context_utterances}),
                    },
                    chunking_policy={
                        "strategy": "utterance_window",
                        "previous_utterances": previous_utterances,
                        "max_tokens": max_tokens,
                    },
                ),
            )
            existing_chunk_ids.add(chunk_id)
            created += 1
    return AudioChunkResult(created=created, skipped=skipped)
