from pathlib import Path

import pytest
from pydantic import BaseModel, model_validator

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.extraction import claim_extractor
from evidence_pipeline.extraction.claim_extractor import (
    RULE_EXTRACTOR_VERSION,
    extract_claims_from_spans,
)
from evidence_pipeline.extraction.llm_client import (
    DeterministicJsonExtractor,
    JsonExtractionError,
    JsonExtractionRequest,
    extract_json,
)
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord


class DemoPayload(BaseModel):
    name: str
    confidence: float

    @model_validator(mode="after")
    def validate_confidence(self) -> "DemoPayload":
        if not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
        return self


def _request(payload):
    return JsonExtractionRequest(
        prompt="Return JSON.",
        schema_name="DemoPayload",
        schema=DemoPayload.model_json_schema(),
        provider="deterministic",
        model="fixture-model",
        metadata={"payload": payload},
    )


def test_deterministic_json_extractor_validates_payload():
    decoded = extract_json(
        DeterministicJsonExtractor(),
        _request({"name": "claim", "confidence": 0.7}),
        DemoPayload,
    )

    assert decoded.name == "claim"
    assert decoded.confidence == 0.7


def test_json_extractor_reports_missing_or_invalid_payload():
    missing = JsonExtractionRequest(
        prompt="Return JSON.",
        schema_name="DemoPayload",
        schema=DemoPayload.model_json_schema(),
        provider="deterministic",
        model="fixture-model",
    )
    with pytest.raises(JsonExtractionError, match="no object payload"):
        DeterministicJsonExtractor().extract_json(missing)

    with pytest.raises(JsonExtractionError, match="failed schema validation"):
        extract_json(
            DeterministicJsonExtractor(),
            _request({"name": "claim", "confidence": 1.5}),
            DemoPayload,
        )


class RecordingJsonExtractor(DeterministicJsonExtractor):
    def __init__(self) -> None:
        super().__init__()
        self.requests = []

    def extract_json(self, request):
        self.requests.append(request)
        return super().extract_json(request)


def test_claim_extraction_uses_json_adapter(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    config = PipelineConfig()
    for path in config.jsonl_paths().values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    append_jsonl(
        config.jsonl_paths()["evidence"],
        EvidenceRecord(
            evidence_id="ev_msg_1",
            source_id="src_chat_1",
            source_modality="chat",
            evidence_type="message_span",
            text="Hope had three masts.",
            provenance={"sender_id": "alice"},
        ),
    )
    append_jsonl(
        config.jsonl_paths()["spans"],
        SpanRecord(
            span_id="span_msg_1",
            source_id="src_chat_1",
            source_modality="chat",
            evidence_id="ev_msg_1",
            text="Hope had three masts.",
            char_start=0,
            char_end=21,
            label="claim_bearing",
            score=0.9,
        ),
    )

    recorder = RecordingJsonExtractor()
    monkeypatch.setattr(claim_extractor, "CLAIM_JSON_EXTRACTOR", recorder)

    result = extract_claims_from_spans(config, modality="chat")

    assert result.created == 1
    assert len(recorder.requests) == 1
    request = recorder.requests[0]
    assert request.schema_name == "RawClaimRecord"
    assert request.provider == "deterministic"
    assert request.model == RULE_EXTRACTOR_VERSION
    claims = [payload for _, payload in read_jsonl(config.jsonl_paths()["claims_raw"])]
    assert claims[0]["model"]["provider"] == "deterministic"
    assert claims[0]["attributes"]["extractor"] == RULE_EXTRACTOR_VERSION
