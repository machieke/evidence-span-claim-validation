from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import append_jsonl, ensure_parent, existing_values, read_jsonl
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.chunks import ChunkRecord
from evidence_pipeline.schemas.claims import ClaimValidationSummary, NormalizedClaimRecord, RawClaimRecord, ValidatedClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.image import ImageRecord, ImageRegionRecord
from evidence_pipeline.schemas.pdf import PDFBlockRecord
from evidence_pipeline.schemas.sources import SourceRecord
from evidence_pipeline.schemas.spans import SpanRecord
from evidence_pipeline.schemas.validation import QuarantineRecord, ValidationRecord

DEMO_SEED_VERSION = "demo.seed.v1"


@dataclass
class DemoSeedResult:
    created: int
    skipped: int
    artifact_counts: Dict[str, int]
    gold_path: Path
    gold_claims: int


def _append_unique(
    path: Path,
    key: str,
    record: Any,
    existing_ids: set,
    artifact_counts: Dict[str, int],
    artifact_name: str,
) -> Tuple[int, int]:
    record_id = getattr(record, key)
    if record_id in existing_ids:
        return 0, 1
    append_jsonl(path, record)
    existing_ids.add(record_id)
    artifact_counts[artifact_name] = artifact_counts.get(artifact_name, 0) + 1
    return 1, 0


def _source_records() -> List[SourceRecord]:
    records = [
        SourceRecord(
            source_id="src_demo_chat",
            source_modality="chat",
            source_file="data/raw/demo/chat.json",
            metadata={"demo": True, "message_count": 10},
        ),
        SourceRecord(
            source_id="src_demo_audio",
            source_modality="audio",
            source_file="data/raw/demo/audio_transcript.json",
            metadata={"demo": True, "utterance_count": 3},
        ),
    ]
    for index in range(1, 4):
        records.append(
            SourceRecord(
                source_id=f"src_demo_pdf_{index}",
                source_modality="pdf",
                source_file=f"data/raw/demo/report_{index}.pdf",
                metadata={"demo": True, "page_count": 1},
            )
        )
    for index in range(1, 21):
        records.append(
            SourceRecord(
                source_id=f"src_demo_image_{index:02d}",
                source_modality="image",
                source_file=f"data/raw/demo/images/image_{index:02d}.png",
                metadata={"demo": True, "width": 64, "height": 64},
            )
        )
    return records


def _chat_messages() -> List[ChatMessageRecord]:
    texts = [
        ("msg_demo_01", "alice", "Did Hope have masts?"),
        ("msg_demo_02", "bob", "Hope had three masts."),
        ("msg_demo_03", "alice", "When was the engine replaced?"),
        ("msg_demo_04", "bob", "The engine was replaced in 2024."),
        ("msg_demo_05", "alice", "Any fuel leak?"),
        ("msg_demo_06", "bob", "No active fuel leak was found."),
        ("msg_demo_07", "alice", "What needs inspection?"),
        ("msg_demo_08", "bob", "The mast may need inspection."),
        ("msg_demo_09", "alice", "Anything else?"),
        ("msg_demo_10", "bob", "The bilge pump was tested yesterday."),
    ]
    return [
        ChatMessageRecord(
            message_id=message_id,
            source_id="src_demo_chat",
            conversation_id="conv_demo",
            thread_id="conv_demo",
            sender_id=sender_id,
            sender_display_name=sender_id.title(),
            sender_role="external",
            timestamp=f"2026-06-24T08:{index:02d}:00Z",
            text=text,
            message_index=index,
            metadata={"demo": True},
        )
        for index, (message_id, sender_id, text) in enumerate(texts)
    ]


def _chat_evidence(messages: Iterable[ChatMessageRecord]) -> List[EvidenceRecord]:
    return [
        EvidenceRecord(
            evidence_id=f"ev_{message.message_id}",
            source_id=message.source_id,
            source_modality="chat",
            evidence_type="message_span",
            text=message.text,
            provenance={
                "conversation_id": message.conversation_id,
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "timestamp": message.timestamp.isoformat() if message.timestamp else None,
                "char_start": 0,
                "char_end": len(message.text),
            },
        )
        for message in messages
    ]


def _pdf_blocks_and_evidence() -> Tuple[List[PDFBlockRecord], List[EvidenceRecord]]:
    texts = [
        (1, "The survey report states vessel Hope had three masts."),
        (2, "The maintenance memo states the engine was replaced in 2024."),
        (3, "The inspection note states no active fuel leak was found."),
    ]
    blocks = []
    evidence = []
    for index, text in texts:
        source_id = f"src_demo_pdf_{index}"
        block_id = f"pdf_demo_block_{index}"
        evidence_id = f"ev_pdf_demo_{index}"
        blocks.append(
            PDFBlockRecord(
                block_id=block_id,
                source_id=source_id,
                source_file=f"data/raw/demo/report_{index}.pdf",
                page=1,
                block_no=0,
                text=text,
                bbox=[72.0, 100.0, 520.0, 125.0],
                char_start_document=0,
                char_end_document=len(text),
                section_path=["Demo", "Survey"],
                extractor="demo_fixture",
                metadata={"demo": True},
            )
        )
        evidence.append(
            EvidenceRecord(
                evidence_id=evidence_id,
                source_id=source_id,
                source_modality="pdf",
                evidence_type="text_span",
                text=text,
                provenance={
                    "page": 1,
                    "page_number": 1,
                    "block_id": block_id,
                    "bbox": [72.0, 100.0, 520.0, 125.0],
                    "char_start": 0,
                    "char_end": len(text),
                },
            )
        )
    return blocks, evidence


def _audio_utterances_and_evidence() -> Tuple[List[AudioUtteranceRecord], List[EvidenceRecord]]:
    rows = [
        ("utt_demo_01", "captain", 0.0, 4.0, "Hope departed at 09:00."),
        ("utt_demo_02", "engineer", 4.5, 8.0, "The engine sounded stable."),
        ("utt_demo_03", "captain", 8.5, 11.0, "No alarms were active."),
    ]
    utterances = []
    evidence = []
    for utterance_id, speaker, start, end, text in rows:
        utterances.append(
            AudioUtteranceRecord(
                utterance_id=utterance_id,
                source_id="src_demo_audio",
                speaker=speaker,
                start=start,
                end=end,
                text=text,
                asr_confidence=0.96,
                diarization_confidence=0.94,
                language="en",
                metadata={"demo": True},
            )
        )
        evidence.append(
            EvidenceRecord(
                evidence_id=f"ev_{utterance_id}",
                source_id="src_demo_audio",
                source_modality="audio",
                evidence_type="utterance_span",
                text=text,
                provenance={
                    "utterance_id": utterance_id,
                    "speaker": speaker,
                    "start_seconds": start,
                    "end_seconds": end,
                    "asr_confidence": 0.96,
                    "diarization_confidence": 0.94,
                },
            )
        )
    return utterances, evidence


def _image_records_regions_and_evidence() -> Tuple[List[ImageRecord], List[ImageRegionRecord], List[EvidenceRecord]]:
    images = []
    regions = []
    evidence = []
    for index in range(1, 21):
        source_id = f"src_demo_image_{index:02d}"
        image_id = source_id
        region_id = f"region_demo_{index:02d}"
        bbox = [8, 8, 32, 32]
        images.append(
            ImageRecord(
                image_id=image_id,
                source_id=source_id,
                source_file=f"data/raw/demo/images/image_{index:02d}.png",
                normalized_file=f"data/raw/demo/images/image_{index:02d}.png",
                width=64,
                height=64,
                color_mode="RGB",
                image_format="PNG",
                metadata={"demo": True},
            )
        )
        regions.append(
            ImageRegionRecord(
                region_id=region_id,
                image_id=image_id,
                source_id=source_id,
                bbox=bbox,
                crop_path=f"data/work/crops/{region_id}.png",
                proposal_method="demo_grid_v1",
                proposal_score=0.91,
                metadata={"demo": True},
            )
        )
        evidence.append(
            EvidenceRecord(
                evidence_id=f"ev_{region_id}",
                source_id=source_id,
                source_modality="image",
                evidence_type="visual_region",
                text=None,
                provenance={
                    "image_id": image_id,
                    "region_id": region_id,
                    "bbox": bbox,
                    "crop_path": f"data/work/crops/{region_id}.png",
                    "proposal_method": "demo_grid_v1",
                    "proposal_score": 0.91,
                },
            )
        )
    return images, regions, evidence


def _chunk_records(chat_evidence: List[EvidenceRecord], pdf_evidence: List[EvidenceRecord], audio_evidence: List[EvidenceRecord]) -> List[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="chunk_demo_chat",
            source_id="src_demo_chat",
            source_modality="chat",
            evidence_ids=[record.evidence_id for record in chat_evidence],
            primary_evidence_ids=[record.evidence_id for record in chat_evidence],
            text="\n".join(record.text or "" for record in chat_evidence),
            provenance_summary={"conversation_id": "conv_demo"},
        ),
        *[
            ChunkRecord(
                chunk_id=f"chunk_demo_pdf_{index}",
                source_id=record.source_id,
                source_modality="pdf",
                evidence_ids=[record.evidence_id],
                primary_evidence_ids=[record.evidence_id],
                text=record.text,
                provenance_summary={"page": 1},
            )
            for index, record in enumerate(pdf_evidence, start=1)
        ],
        ChunkRecord(
            chunk_id="chunk_demo_audio",
            source_id="src_demo_audio",
            source_modality="audio",
            evidence_ids=[record.evidence_id for record in audio_evidence],
            primary_evidence_ids=[record.evidence_id for record in audio_evidence],
            text="\n".join(record.text or "" for record in audio_evidence),
            provenance_summary={"speakers": ["captain", "engineer"]},
        ),
    ]


def _accepted_validation_summary(exact: bool = True) -> ClaimValidationSummary:
    return ClaimValidationSummary(
        deterministic_valid=True,
        evidence_exact_match=exact,
        negation_preserved=True,
        uncertainty_preserved=True,
        attribution_preserved=True,
        quantities_preserved=True,
    )


def _text_claim(
    claim_id: str,
    source_id: str,
    source_modality: str,
    span_id: str,
    evidence_id: str,
    text: str,
    attribution: Dict[str, str],
    truth_status: str,
    modality: str = "asserted",
) -> RawClaimRecord:
    prefix = "The document states" if source_modality == "pdf" else "The speaker asserted"
    return RawClaimRecord(
        claim_id=claim_id,
        source_id=source_id,
        source_modality=source_modality,
        span_id=span_id,
        evidence_id=evidence_id,
        source_faithful_claim=f"{prefix}: {text}",
        subject="Hope",
        predicate="states",
        object=text,
        modality=modality,
        evidence_text=text,
        attribution=attribution,
        truth_status=truth_status,
        confidence=0.9,
        model={"provider": "deterministic", "model": DEMO_SEED_VERSION},
    )


def _span(span_id: str, source_id: str, source_modality: str, evidence_id: str, chunk_id: str, text: str) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        chunk_id=chunk_id,
        source_id=source_id,
        source_modality=source_modality,
        evidence_id=evidence_id,
        text=text,
        char_start=0,
        char_end=len(text),
        label="claim_bearing",
        score=0.9,
        detector={"model": DEMO_SEED_VERSION},
    )


def _text_claim_records() -> Tuple[List[SpanRecord], List[RawClaimRecord], List[ValidatedClaimRecord], List[NormalizedClaimRecord]]:
    specs = [
        ("chat_01", "src_demo_chat", "chat", "ev_msg_demo_02", "chunk_demo_chat", "Hope had three masts.", {"type": "speaker", "agent": "bob"}, "speaker_asserted_unverified", "asserted"),
        ("chat_02", "src_demo_chat", "chat", "ev_msg_demo_04", "chunk_demo_chat", "The engine was replaced in 2024.", {"type": "speaker", "agent": "bob"}, "speaker_asserted_unverified", "asserted"),
        ("chat_03", "src_demo_chat", "chat", "ev_msg_demo_06", "chunk_demo_chat", "No active fuel leak was found.", {"type": "speaker", "agent": "bob"}, "speaker_asserted_unverified", "negated"),
        ("chat_04", "src_demo_chat", "chat", "ev_msg_demo_08", "chunk_demo_chat", "The mast may need inspection.", {"type": "speaker", "agent": "bob"}, "speaker_asserted_unverified", "uncertain_observation"),
        ("pdf_01", "src_demo_pdf_1", "pdf", "ev_pdf_demo_1", "chunk_demo_pdf_1", "The survey report states vessel Hope had three masts.", {"type": "document", "agent": "src_demo_pdf_1"}, "source_asserted_unverified", "asserted"),
        ("pdf_02", "src_demo_pdf_2", "pdf", "ev_pdf_demo_2", "chunk_demo_pdf_2", "The maintenance memo states the engine was replaced in 2024.", {"type": "document", "agent": "src_demo_pdf_2"}, "source_asserted_unverified", "asserted"),
        ("pdf_03", "src_demo_pdf_3", "pdf", "ev_pdf_demo_3", "chunk_demo_pdf_3", "The inspection note states no active fuel leak was found.", {"type": "document", "agent": "src_demo_pdf_3"}, "source_asserted_unverified", "negated"),
        ("audio_01", "src_demo_audio", "audio", "ev_utt_demo_01", "chunk_demo_audio", "Hope departed at 09:00.", {"type": "speaker", "agent": "captain"}, "speaker_asserted_unverified", "asserted"),
        ("audio_02", "src_demo_audio", "audio", "ev_utt_demo_02", "chunk_demo_audio", "The engine sounded stable.", {"type": "speaker", "agent": "engineer"}, "speaker_asserted_unverified", "asserted"),
    ]
    spans = []
    raw_claims = []
    validated = []
    normalized = []
    for suffix, source_id, source_modality, evidence_id, chunk_id, text, attribution, truth_status, modality in specs:
        span_id = f"span_demo_{suffix}"
        claim_id = f"claim_demo_{suffix}"
        spans.append(_span(span_id, source_id, source_modality, evidence_id, chunk_id, text))
        raw = _text_claim(claim_id, source_id, source_modality, span_id, evidence_id, text, attribution, truth_status, modality)
        raw_claims.append(raw)
        validated.append(
            ValidatedClaimRecord(
                claim_id=claim_id,
                source_id=source_id,
                source_modality=source_modality,
                span_id=span_id,
                evidence_id=evidence_id,
                source_faithful_claim=raw.source_faithful_claim,
                evidence_text=text,
                modality=modality,
                truth_status=truth_status,
                support_status="accepted_extracted",
                validation=_accepted_validation_summary(),
            )
        )
        normalized.append(
            NormalizedClaimRecord(
                normalized_claim_id=f"nclaim_demo_{suffix}",
                claim_id=claim_id,
                source_id=source_id,
                evidence_id=evidence_id,
                normalized_claim={"subject": "entity:hope", "predicate": "states", "object": text},
            )
        )
    return spans, raw_claims, validated, normalized


def _image_claim_records(image_evidence: List[EvidenceRecord]) -> Tuple[List[RawClaimRecord], List[ValidatedClaimRecord], List[NormalizedClaimRecord]]:
    raw_claims = []
    validated = []
    normalized = []
    for index, evidence in enumerate(image_evidence, start=1):
        region_id = str(evidence.provenance["region_id"])
        claim_id = f"claim_demo_image_{index:02d}"
        raw = RawClaimRecord(
            claim_id=claim_id,
            source_id=evidence.source_id,
            source_modality="image",
            evidence_id=evidence.evidence_id,
            claim_type="visual_region_proposal",
            source_faithful_claim=f"Region {region_id} was proposed as a visual region by demo_grid_v1.",
            subject=region_id,
            predicate="proposed_visual_region",
            object={"bbox": evidence.provenance["bbox"], "crop_path": evidence.provenance["crop_path"]},
            attributes={"extractor": DEMO_SEED_VERSION},
            modality="model_observation",
            attribution={"type": "model", "agent": "demo_grid_v1"},
            truth_status="model_observation_unverified",
            confidence=0.91,
            model={"provider": "deterministic", "model": DEMO_SEED_VERSION},
        )
        raw_claims.append(raw)
        validated.append(
            ValidatedClaimRecord(
                claim_id=claim_id,
                source_id=evidence.source_id,
                source_modality="image",
                evidence_id=evidence.evidence_id,
                source_faithful_claim=raw.source_faithful_claim,
                modality="model_observation",
                truth_status="model_observation_unverified",
                support_status="accepted_extracted",
                validation=_accepted_validation_summary(exact=False).model_copy(update={"evidence_exact_match": None}),
            )
        )
        normalized.append(
            NormalizedClaimRecord(
                normalized_claim_id=f"nclaim_demo_image_{index:02d}",
                claim_id=claim_id,
                source_id=evidence.source_id,
                evidence_id=evidence.evidence_id,
                normalized_claim={"subject": f"image_region:{region_id}", "predicate": "proposed_visual_region", "object": "visual_region"},
            )
        )
    return raw_claims, validated, normalized


def _quarantined_records() -> Tuple[RawClaimRecord, ValidationRecord, QuarantineRecord]:
    claim = _text_claim(
        "claim_demo_quarantined",
        "src_demo_chat",
        "chat",
        "span_demo_quarantined",
        "ev_msg_demo_10",
        "Hope had five engines.",
        {"type": "speaker", "agent": "bob"},
        "speaker_asserted_unverified",
    )
    validation = ValidationRecord(
        validation_id="val_demo_quarantined",
        claim_id=claim.claim_id,
        stage="deterministic_validation",
        status="quarantined",
        errors=["evidence_not_exact_substring"],
        validator_version=DEMO_SEED_VERSION,
    )
    quarantine = QuarantineRecord(
        quarantine_id="q_demo_quarantined",
        record_type="claim_raw",
        record_id=claim.claim_id,
        source_id=claim.source_id,
        evidence_id=claim.evidence_id,
        claim_id=claim.claim_id,
        stage="deterministic_validation",
        reason_codes=["evidence_not_exact_substring"],
        payload=claim.model_dump(mode="json", exclude_none=True),
    )
    return claim, validation, quarantine


def _validation_records(validated_claims: Iterable[ValidatedClaimRecord]) -> List[ValidationRecord]:
    return [
        ValidationRecord(
            validation_id=f"val_{claim.claim_id}",
            claim_id=claim.claim_id,
            stage="deterministic_validation",
            status="accepted_extracted",
            errors=[],
            validator_version=DEMO_SEED_VERSION,
            metadata={"validation": claim.validation.model_dump(mode="json", exclude_none=True)},
        )
        for claim in validated_claims
    ]


def seed_demo_artifacts(config: PipelineConfig) -> DemoSeedResult:
    paths = config.jsonl_paths()
    artifact_counts: Dict[str, int] = {}
    created = 0
    skipped = 0

    chat_messages = _chat_messages()
    chat_evidence = _chat_evidence(chat_messages)
    pdf_blocks, pdf_evidence = _pdf_blocks_and_evidence()
    audio_utterances, audio_evidence = _audio_utterances_and_evidence()
    images, image_regions, image_evidence = _image_records_regions_and_evidence()
    chunks = _chunk_records(chat_evidence, pdf_evidence, audio_evidence)
    spans, text_raw, text_validated, text_normalized = _text_claim_records()
    image_raw, image_validated, image_normalized = _image_claim_records(image_evidence)
    quarantined_claim, quarantined_validation, quarantine = _quarantined_records()
    validations = _validation_records([*text_validated, *image_validated]) + [quarantined_validation]

    plan = [
        ("sources", "source_id", _source_records()),
        ("chat_messages", "message_id", chat_messages),
        ("pdf_blocks", "block_id", pdf_blocks),
        ("audio_utterances", "utterance_id", audio_utterances),
        ("images", "image_id", images),
        ("image_regions", "region_id", image_regions),
        ("evidence", "evidence_id", [*chat_evidence, *pdf_evidence, *audio_evidence, *image_evidence]),
        ("chunks", "chunk_id", chunks),
        ("spans", "span_id", [*spans, _span("span_demo_quarantined", "src_demo_chat", "chat", "ev_msg_demo_10", "chunk_demo_chat", "The bilge pump was tested yesterday.")]),
        ("claims_raw", "claim_id", [*text_raw, *image_raw, quarantined_claim]),
        ("validations", "validation_id", validations),
        ("claims_validated", "claim_id", [*text_validated, *image_validated]),
        ("claims_normalized", "normalized_claim_id", [*text_normalized, *image_normalized]),
        ("quarantine", "quarantine_id", [quarantine]),
    ]

    for artifact_name, key, records in plan:
        existing_ids = existing_values(paths[artifact_name], key)
        for record in records:
            made, skipped_record = _append_unique(
                paths[artifact_name],
                key,
                record,
                existing_ids,
                artifact_counts,
                artifact_name,
            )
            created += made
            skipped += skipped_record

    gold_result = write_demo_gold_file(config)
    return DemoSeedResult(
        created=created,
        skipped=skipped,
        artifact_counts=artifact_counts,
        gold_path=gold_result.gold_path,
        gold_claims=gold_result.gold_claims,
    )


@dataclass
class DemoGoldResult:
    gold_path: Path
    gold_claims: int


def _gold_claim_from_payload(payload: dict, expected_status: str) -> Optional[dict]:
    evidence_id = payload.get("evidence_id")
    evidence_text = payload.get("evidence_text")
    if not evidence_id or not evidence_text:
        return None
    claim = {
        "evidence_id": evidence_id,
        "evidence_text": evidence_text,
        "expected_status": expected_status,
    }
    claim_id = payload.get("claim_id")
    if claim_id:
        claim["claim_id"] = claim_id
    return claim


def write_demo_gold_file(config: PipelineConfig, output_path: Optional[Path] = None) -> DemoGoldResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "demo_gold.json"
    claims: List[dict] = []
    paths = config.jsonl_paths()
    for _, payload in read_jsonl(paths["claims_validated"]):
        claim_id = str(payload.get("claim_id") or "")
        if not claim_id.startswith("claim_demo_"):
            continue
        claim = _gold_claim_from_payload(payload, "accepted")
        if claim is not None:
            claims.append(claim)
    for _, payload in read_jsonl(paths["quarantine"]):
        claim_id = str(payload.get("claim_id") or "")
        if not claim_id.startswith("claim_demo_"):
            continue
        claim_payload = payload.get("payload") or {}
        if not isinstance(claim_payload, dict):
            continue
        claim = _gold_claim_from_payload(claim_payload, "quarantined")
        if claim is not None:
            claims.append(claim)

    ensure_parent(output_path)
    output_path.write_text(
        json.dumps({"claims": claims}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return DemoGoldResult(gold_path=output_path, gold_claims=len(claims))
