from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


@dataclass
class AudioEvidenceResult:
    created: int
    skipped: int


def _audio_evidence_id(utterance: AudioUtteranceRecord) -> str:
    return stable_id("ev_audio", {"source_id": utterance.source_id, "utterance_id": utterance.utterance_id})


def build_audio_evidence(config: PipelineConfig, source_id: Optional[str] = None) -> AudioEvidenceResult:
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["evidence"], "evidence_id")
    created = 0
    skipped = 0

    for _, utterance in read_jsonl_records(paths["audio_utterances"], AudioUtteranceRecord):
        if source_id is not None and utterance.source_id != source_id:
            continue
        evidence_id = _audio_evidence_id(utterance)
        if evidence_id in existing_ids:
            skipped += 1
            continue
        provenance: Dict[str, object] = {
            "utterance_id": utterance.utterance_id,
            "speaker": utterance.speaker,
            "start": utterance.start,
            "end": utterance.end,
            "asr_confidence": utterance.asr_confidence,
            "diarization_confidence": utterance.diarization_confidence,
        }
        append_jsonl(
            paths["evidence"],
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=utterance.source_id,
                source_modality="audio",
                evidence_type="utterance_span",
                text=utterance.text,
                provenance={key: value for key, value in provenance.items() if value is not None},
                risk_flags=utterance.risk_flags,
            ),
        )
        existing_ids.add(evidence_id)
        created += 1

    return AudioEvidenceResult(created=created, skipped=skipped)
