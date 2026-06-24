from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import sha256_file, stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.sources import SourceRecord


@dataclass
class ChatIngestResult:
    source_id: str
    source_created: bool
    messages_created: int
    messages_skipped: int


def _load_export(path: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    defaults: Dict[str, Any] = {}
    if isinstance(payload, list):
        messages = payload
    elif isinstance(payload, dict) and isinstance(payload.get("messages"), list):
        messages = payload["messages"]
        defaults = {key: value for key, value in payload.items() if key != "messages"}
    else:
        raise ValueError("chat export must be a JSON list or an object with a messages list")

    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("every chat message must be a JSON object")
    return messages, defaults


def _coerce_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _coerce_dict_list(value: Any) -> List[Dict[str, Any]]:
    result = []
    for item in _coerce_list(value):
        if isinstance(item, dict):
            result.append(item)
        else:
            result.append({"value": item})
    return result


def _first_present(message: Dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    for key in keys:
        if key in message and message[key] is not None:
            return message[key]
    return default


def _sender_fields(message: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    author = message.get("author")
    sender = message.get("sender")
    if isinstance(author, dict):
        sender_id = str(author.get("id") or author.get("username") or author.get("name") or "unknown")
        display = author.get("display_name") or author.get("name") or author.get("username")
        return sender_id, str(display) if display is not None else None
    if isinstance(sender, dict):
        sender_id = str(sender.get("id") or sender.get("username") or sender.get("name") or "unknown")
        display = sender.get("display_name") or sender.get("name") or sender.get("username")
        return sender_id, str(display) if display is not None else None
    sender_id = str(_first_present(message, ["sender_id", "user_id", "author_id", "sender"], "unknown"))
    display = _first_present(message, ["sender_display_name", "sender_name", "author_name", "username"])
    return sender_id, str(display) if display is not None else None


def _sender_role(message: Dict[str, Any]) -> str:
    raw = str(_first_present(message, ["sender_role", "role"], "external")).lower()
    if raw in {"user", "assistant", "system", "external"}:
        return raw
    if raw in {"bot", "ai"}:
        return "assistant"
    return "external"


def _message_text(message: Dict[str, Any]) -> str:
    text = _first_present(message, ["text", "content", "body", "message"], "")
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    return str(text)


def _message_id(source_id: str, index: int, message: Dict[str, Any], text: str, sender_id: str) -> str:
    raw_id = _first_present(message, ["message_id", "id"])
    if raw_id is not None and str(raw_id).strip():
        return str(raw_id)
    return stable_id(
        "msg",
        {
            "source_id": source_id,
            "index": index,
            "sender_id": sender_id,
            "timestamp": _first_present(message, ["timestamp", "created_at", "ts", "date"]),
            "text": text,
        },
    )


def _build_message(
    source_id: str,
    index: int,
    message: Dict[str, Any],
    defaults: Dict[str, Any],
    metadata: Dict[str, Any],
) -> ChatMessageRecord:
    sender_id, sender_display_name = _sender_fields(message)
    text = _message_text(message)
    conversation_id = str(
        _first_present(
            message,
            ["conversation_id", "channel_id", "chat_id", "room_id"],
            defaults.get("conversation_id") or defaults.get("channel_id") or source_id,
        )
    )
    thread_id = _first_present(message, ["thread_id"], defaults.get("thread_id") or conversation_id)
    quoted_message_ids = [str(item) for item in _coerce_list(message.get("quoted_message_ids"))]
    attachments = _coerce_dict_list(message.get("attachments"))
    edit_history = _coerce_dict_list(message.get("edit_history"))
    risk_flags = [str(item) for item in _coerce_list(message.get("risk_flags"))]
    if edit_history and "edited_message" not in risk_flags:
        risk_flags.append("edited_message")

    return ChatMessageRecord(
        message_id=_message_id(source_id, index, message, text, sender_id),
        source_id=source_id,
        conversation_id=conversation_id,
        thread_id=str(thread_id) if thread_id is not None else conversation_id,
        sender_id=sender_id,
        sender_display_name=sender_display_name,
        sender_role=_sender_role(message),
        timestamp=_first_present(message, ["timestamp", "created_at", "ts", "date"]),
        text=text,
        reply_to_message_id=_first_present(message, ["reply_to_message_id", "reply_to", "parent_id"]),
        quoted_message_ids=quoted_message_ids,
        edit_history=edit_history,
        attachments=attachments,
        message_index=index,
        metadata=metadata,
        risk_flags=risk_flags,
    )


def ingest_chat_export(path: Path, config: PipelineConfig, metadata: Optional[Dict[str, Any]] = None) -> ChatIngestResult:
    messages, defaults = _load_export(path)
    metadata = dict(metadata or {})
    if isinstance(defaults.get("metadata"), dict):
        merged = dict(defaults["metadata"])
        merged.update(metadata)
        metadata = merged

    sha256 = sha256_file(path)
    source_id = stable_id("src", {"modality": "chat", "sha256": sha256})
    paths = config.jsonl_paths()
    existing_sources = existing_values(paths["sources"], "source_id")
    source_created = source_id not in existing_sources
    if source_created:
        append_jsonl(
            paths["sources"],
            SourceRecord(
                source_id=source_id,
                source_modality="chat",
                source_file=str(path),
                sha256=sha256,
                metadata={**metadata, "message_count": len(messages)},
            ),
        )

    existing_message_ids = existing_values(paths["chat_messages"], "message_id")
    created = 0
    skipped = 0
    for index, message in enumerate(messages):
        record = _build_message(source_id, index, message, defaults, metadata)
        if record.message_id in existing_message_ids:
            skipped += 1
            continue
        append_jsonl(paths["chat_messages"], record)
        existing_message_ids.add(record.message_id)
        created += 1

    return ChatIngestResult(
        source_id=source_id,
        source_created=source_created,
        messages_created=created,
        messages_skipped=skipped,
    )
