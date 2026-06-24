import pytest
from pydantic import ValidationError

from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord


def test_text_like_evidence_requires_text():
    with pytest.raises(ValidationError):
        EvidenceRecord(
            evidence_id="ev_1",
            source_id="src_1",
            source_modality="chat",
            evidence_type="message_span",
            text=None,
        )


def test_visual_region_evidence_allows_null_text():
    record = EvidenceRecord(
        evidence_id="ev_1",
        source_id="src_1",
        source_modality="image",
        evidence_type="visual_region",
        text=None,
        provenance={"bbox": [0, 0, 10, 10]},
    )

    assert record.text is None


def test_span_score_bounds():
    with pytest.raises(ValidationError):
        SpanRecord(
            span_id="span_1",
            source_id="src_1",
            source_modality="chat",
            evidence_id="ev_1",
            text="claim",
            label="claim_bearing",
            score=1.1,
        )
