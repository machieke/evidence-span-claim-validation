from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


@dataclass
class ChatChunkResult:
    created: int
    skipped: int


def _sort_key(message: ChatMessageRecord) -> Tuple[str, int, str]:
    timestamp = message.timestamp.isoformat() if message.timestamp else ""
    return timestamp, message.message_index if message.message_index is not None else 0, message.message_id


def _format_message(message: ChatMessageRecord) -> str:
    label = message.sender_display_name or message.sender_id
    return f"{label}: {message.text}"


def _load_message_evidence(config: PipelineConfig) -> Dict[str, EvidenceRecord]:
    paths = config.jsonl_paths()
    evidence_by_message_id: Dict[str, EvidenceRecord] = {}
    for _, evidence in read_jsonl_records(paths["evidence"], EvidenceRecord):
        if evidence.source_modality != "chat" or evidence.evidence_type != "message_span":
            continue
        message_id = evidence.provenance.get("message_id")
        if isinstance(message_id, str):
            evidence_by_message_id[message_id] = evidence
    return evidence_by_message_id


def _group_messages(messages: Iterable[ChatMessageRecord]) -> Dict[Tuple[str, str, str], List[ChatMessageRecord]]:
    grouped: Dict[Tuple[str, str, str], List[ChatMessageRecord]] = defaultdict(list)
    for message in messages:
        grouped[(message.source_id, message.conversation_id, message.thread_id or message.conversation_id)].append(message)
    for key in grouped:
        grouped[key].sort(key=_sort_key)
    return grouped


def chunk_chat(config: PipelineConfig, source_id: Optional[str] = None, previous_messages: int = 2, max_tokens: int = 1200) -> ChatChunkResult:
    paths = config.jsonl_paths()
    evidence_by_message_id = _load_message_evidence(config)
    messages = [
        message
        for _, message in read_jsonl_records(paths["chat_messages"], ChatMessageRecord)
        if source_id is None or message.source_id == source_id
    ]
    grouped = _group_messages(messages)
    existing_chunk_ids = existing_values(paths["chunks"], "chunk_id")
    created = 0
    skipped = 0

    for (_, conversation_id, thread_id), thread_messages in grouped.items():
        for index, message in enumerate(thread_messages):
            primary_evidence = evidence_by_message_id.get(message.message_id)
            if primary_evidence is None:
                continue
            start = max(0, index - previous_messages)
            context_messages = thread_messages[start : index + 1]
            evidence_ids = [
                evidence_by_message_id[item.message_id].evidence_id
                for item in context_messages
                if item.message_id in evidence_by_message_id
            ]
            overlap_ids = [
                evidence_by_message_id[item.message_id].evidence_id
                for item in context_messages[:-1]
                if item.message_id in evidence_by_message_id
            ]
            chunk_id = stable_id(
                "chunk_chat",
                {
                    "primary_evidence_id": primary_evidence.evidence_id,
                    "previous_messages": previous_messages,
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
                    source_id=message.source_id,
                    source_modality="chat",
                    evidence_ids=evidence_ids,
                    primary_evidence_ids=[primary_evidence.evidence_id],
                    overlap_evidence_ids=overlap_ids,
                    text="\n".join(_format_message(item) for item in context_messages),
                    provenance_summary={
                        "conversation_id": conversation_id,
                        "thread_id": thread_id,
                        "message_ids": [item.message_id for item in context_messages],
                        "primary_message_id": message.message_id,
                    },
                    chunking_policy={
                        "strategy": "thread_window",
                        "previous_messages": previous_messages,
                        "max_tokens": max_tokens,
                    },
                ),
            )
            existing_chunk_ids.add(chunk_id)
            created += 1
    return ChatChunkResult(created=created, skipped=skipped)
