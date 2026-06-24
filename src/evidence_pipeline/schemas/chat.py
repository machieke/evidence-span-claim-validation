from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import StrictModel

SenderRole = Literal["user", "assistant", "system", "external"]


class ChatMessageRecord(StrictModel):
    message_id: str
    source_id: str
    conversation_id: str
    thread_id: Optional[str] = None
    sender_id: str
    sender_display_name: Optional[str] = None
    sender_role: SenderRole = "external"
    timestamp: Optional[datetime] = None
    text: str
    reply_to_message_id: Optional[str] = None
    quoted_message_ids: List[str] = Field(default_factory=list)
    edit_history: List[Dict[str, Any]] = Field(default_factory=list)
    attachments: List[Dict[str, Any]] = Field(default_factory=list)
    message_index: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "chat.message.v1"

    @model_validator(mode="after")
    def validate_message(self) -> "ChatMessageRecord":
        if not self.text.strip():
            raise ValueError("chat message text must not be empty")
        if self.thread_id is None:
            self.thread_id = self.conversation_id
        return self
