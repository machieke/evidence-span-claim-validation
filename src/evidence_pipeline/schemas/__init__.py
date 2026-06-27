from evidence_pipeline.schemas.base import SourceModality
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.claims import NormalizedClaimRecord, RawClaimRecord, ValidatedClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.pdf import PDFBlockRecord
from evidence_pipeline.schemas.sources import SourceRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.schemas.validation import ErrorRecord, QuarantineRecord, ValidationRecord

SCHEMA_REGISTRY = {
    "source": SourceRecord,
    "sources": SourceRecord,
    "chat_message": ChatMessageRecord,
    "chat-message": ChatMessageRecord,
    "chat_messages": ChatMessageRecord,
    "pdf_block": PDFBlockRecord,
    "pdf-block": PDFBlockRecord,
    "pdf_blocks": PDFBlockRecord,
    "audio_utterance": AudioUtteranceRecord,
    "audio-utterance": AudioUtteranceRecord,
    "audio_utterances": AudioUtteranceRecord,
    "evidence": EvidenceRecord,
    "chunk": ChunkRecord,
    "chunks": ChunkRecord,
    "span": SpanRecord,
    "spans": SpanRecord,
    "claim.raw": RawClaimRecord,
    "raw-claim": RawClaimRecord,
    "claims.raw": RawClaimRecord,
    "claim.validated": ValidatedClaimRecord,
    "validated-claim": ValidatedClaimRecord,
    "claims.validated": ValidatedClaimRecord,
    "claim.normalized": NormalizedClaimRecord,
    "normalized-claim": NormalizedClaimRecord,
    "claims.normalized": NormalizedClaimRecord,
    "validation": ValidationRecord,
    "validations": ValidationRecord,
    "quarantine": QuarantineRecord,
    "error": ErrorRecord,
    "errors": ErrorRecord,
}

__all__ = [
    "ChunkRecord",
    "ChatMessageRecord",
    "AudioUtteranceRecord",
    "ErrorRecord",
    "EvidenceRecord",
    "NormalizedClaimRecord",
    "PDFBlockRecord",
    "QuarantineRecord",
    "RawClaimRecord",
    "SCHEMA_REGISTRY",
    "SourceModality",
    "SourceRecord",
    "SpanRecord",
    "ValidatedClaimRecord",
    "ValidationRecord",
]
