import pytest
from pydantic import ValidationError

from evidence_pipeline.schemas import SCHEMA_REGISTRY
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.reports import GraphEdgeRecord
from evidence_pipeline.schemas.review import ReviewQueueRecord
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


def test_review_queue_record_requires_valid_state_and_claim_id():
    record = ReviewQueueRecord(
        review_queue_id="reviewq_1",
        claim_id="claim_1",
        source_id="src_1",
        evidence_id="ev_1",
        source_modality="image",
        validation_status="quarantined",
        reason_codes=["image_label_low_confidence"],
        review_state="unreviewed",
    )

    assert record.schema_version == "review.queue.v1"
    assert SCHEMA_REGISTRY["review_queue"] is ReviewQueueRecord

    with pytest.raises(ValidationError):
        ReviewQueueRecord(
            review_queue_id="reviewq_2",
            claim_id="claim_2",
            validation_status="quarantined",
            review_state="done",
        )

    with pytest.raises(ValidationError):
        ReviewQueueRecord(
            review_queue_id="",
            claim_id="claim_3",
            validation_status="quarantined",
            review_state="unreviewed",
        )


def test_graph_edge_record_requires_stable_identifiers():
    record = GraphEdgeRecord(
        edge_id="edge_1",
        normalized_claim_id="nclaim_1",
        claim_id="claim_1",
        source_id="src_1",
        evidence_id="ev_1",
        subject="speaker:alice",
        predicate="asserts",
        object="Hope had three masts.",
    )

    assert record.schema_version == "graph.edge.v1"
    assert SCHEMA_REGISTRY["claim_graph"] is GraphEdgeRecord

    with pytest.raises(ValidationError):
        GraphEdgeRecord(
            edge_id="edge_2",
            normalized_claim_id="nclaim_1",
            claim_id="claim_1",
            source_id="src_1",
            evidence_id="ev_1",
            subject="speaker:alice",
            predicate="",
            object="Hope had three masts.",
        )
