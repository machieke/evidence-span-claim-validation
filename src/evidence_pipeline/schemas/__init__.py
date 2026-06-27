from evidence_pipeline.schemas.base import SourceModality
from evidence_pipeline.schemas.audit import AuditEventRecord
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.claims import NormalizedClaimRecord, RawClaimRecord, ValidatedClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.image import (
    ImageFeatureClusterRecord,
    ImageRecord,
    ImageRegionEmbeddingRecord,
    ImageRegionRecord,
)
from evidence_pipeline.schemas.pdf import PDFBlockRecord
from evidence_pipeline.schemas.review import ReviewDecisionRecord
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
    "image": ImageRecord,
    "images": ImageRecord,
    "image_region": ImageRegionRecord,
    "image-region": ImageRegionRecord,
    "image_regions": ImageRegionRecord,
    "image_region_embedding": ImageRegionEmbeddingRecord,
    "image-region-embedding": ImageRegionEmbeddingRecord,
    "image_region_embeddings": ImageRegionEmbeddingRecord,
    "image_feature_cluster": ImageFeatureClusterRecord,
    "image-feature-cluster": ImageFeatureClusterRecord,
    "image_feature_clusters": ImageFeatureClusterRecord,
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
    "review_decision": ReviewDecisionRecord,
    "review-decision": ReviewDecisionRecord,
    "review_decisions": ReviewDecisionRecord,
    "audit_event": AuditEventRecord,
    "audit-event": AuditEventRecord,
    "audit_events": AuditEventRecord,
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
    "AuditEventRecord",
    "ErrorRecord",
    "EvidenceRecord",
    "ImageFeatureClusterRecord",
    "ImageRecord",
    "ImageRegionEmbeddingRecord",
    "ImageRegionRecord",
    "NormalizedClaimRecord",
    "PDFBlockRecord",
    "QuarantineRecord",
    "RawClaimRecord",
    "ReviewDecisionRecord",
    "SCHEMA_REGISTRY",
    "SourceModality",
    "SourceRecord",
    "SpanRecord",
    "ValidatedClaimRecord",
    "ValidationRecord",
]
