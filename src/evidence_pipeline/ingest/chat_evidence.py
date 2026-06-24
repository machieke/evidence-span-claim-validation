from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


@dataclass
class ChatEvidenceResult:
    created: int
    skipped: int


def _message_evidence_id(message: ChatMessageRecord) -> str:
    return stable_id("ev_msg", {"source_id": message.source_id, "message_id": message.message_id})


def build_chat_evidence(config: PipelineConfig, source_id: Optional[str] = None) -> ChatEvidenceResult:
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["evidence"], "evidence_id")
    created = 0
    skipped = 0
    for _, message in read_jsonl_records(paths["chat_messages"], ChatMessageRecord):
        if source_id is not None and message.source_id != source_id:
            continue
        evidence_id = _message_evidence_id(message)
        if evidence_id in existing_ids:
            skipped += 1
            continue
        provenance: Dict[str, object] = {
            "conversation_id": message.conversation_id,
            "thread_id": message.thread_id,
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "sender_display_name": message.sender_display_name,
            "sender_role": message.sender_role,
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
            "char_start": 0,
            "char_end": len(message.text),
        }
        append_jsonl(
            paths["evidence"],
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=message.source_id,
                source_modality="chat",
                evidence_type="message_span",
                text=message.text,
                provenance={key: value for key, value in provenance.items() if value is not None},
                risk_flags=message.risk_flags,
            ),
        )
        existing_ids.add(evidence_id)
        created += 1
    return ChatEvidenceResult(created=created, skipped=skipped)
