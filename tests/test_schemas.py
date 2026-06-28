import pytest
from pydantic import ValidationError

from evidence_pipeline.schemas import SCHEMA_REGISTRY
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.reports import (
    ClaimDuplicateGroupRecord,
    ClaimRepairSuggestionRecord,
    GoldEvaluationRecord,
    GraphEdgeRecord,
    ModelRoutingRecord,
    PIIFindingRecord,
    PIIRedactionRecord,
    PrivacyPolicyViolationRecord,
    RetentionPlanRecord,
)
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


def test_model_routing_record_requires_valid_tier_role_and_score():
    record = ModelRoutingRecord(
        routing_id="route_1",
        stage="extract_claims",
        record_type="span",
        record_id="span_1",
        source_id="src_1",
        source_modality="chat",
        model_role="extraction",
        selected_tier="strong",
        selected_model="strong_extract",
        reasons=["span_score_lt:0.7"],
        score=0.5,
    )

    assert record.schema_version == "model.routing.v1"
    assert SCHEMA_REGISTRY["model_routing"] is ModelRoutingRecord

    with pytest.raises(ValidationError):
        ModelRoutingRecord(
            routing_id="route_2",
            stage="extract_claims",
            record_type="span",
            record_id="span_1",
            source_id="src_1",
            source_modality="chat",
            model_role="ranking",
            selected_tier="strong",
            selected_model="strong_extract",
        )

    with pytest.raises(ValidationError):
        ModelRoutingRecord(
            routing_id="route_3",
            stage="extract_claims",
            record_type="span",
            record_id="span_1",
            source_id="src_1",
            source_modality="chat",
            model_role="extraction",
            selected_tier="strong",
            selected_model="strong_extract",
            score=1.5,
        )


def test_privacy_policy_violation_record_requires_reason_code():
    record = PrivacyPolicyViolationRecord(
        violation_id="privacy_1",
        source_id="src_1",
        claim_id="claim_1",
        evidence_id="ev_1",
        provider="openai",
        model="external_model",
        policy="local_only_sensitive_sources",
        reason_code="non_local_provider_for_sensitive_source",
        sensitive_metadata_keys=["contains_pii"],
    )

    assert record.schema_version == "privacy.violation.v1"
    assert SCHEMA_REGISTRY["privacy_policy_violations"] is PrivacyPolicyViolationRecord

    with pytest.raises(ValidationError):
        PrivacyPolicyViolationRecord(
            violation_id="privacy_2",
            source_id="src_1",
            claim_id="claim_1",
            evidence_id="ev_1",
            provider="openai",
            model="external_model",
            policy="local_only_sensitive_sources",
            reason_code="",
        )


def test_retention_plan_record_requires_positive_retention_days():
    record = RetentionPlanRecord(
        retention_id="retention_1",
        action="delete_raw_source",
        source_id="src_1",
        source_modality="chat",
        source_file="chat.json",
        ingested_at="2026-01-01T00:00:00Z",
        age_days=400,
        retention_days=365,
        reason_code="raw_source_retention_exceeded",
    )

    assert record.schema_version == "retention.plan.v1"
    assert SCHEMA_REGISTRY["retention_plan"] is RetentionPlanRecord

    with pytest.raises(ValidationError):
        RetentionPlanRecord(
            retention_id="retention_2",
            action="delete_raw_source",
            source_id="src_1",
            source_modality="chat",
            source_file="chat.json",
            ingested_at="2026-01-01T00:00:00Z",
            age_days=400,
            retention_days=0,
            reason_code="raw_source_retention_exceeded",
        )


def test_pii_finding_record_requires_valid_offsets():
    record = PIIFindingRecord(
        finding_id="pii_1",
        artifact="chat_messages",
        record_id="msg_1",
        field="text",
        pii_type="email",
        match_hash="hash_1",
        redacted_preview="a***@example.com",
        char_start=6,
        char_end=23,
    )

    assert record.schema_version == "pii.finding.v1"
    assert SCHEMA_REGISTRY["pii_findings"] is PIIFindingRecord

    with pytest.raises(ValidationError):
        PIIFindingRecord(
            finding_id="pii_2",
            artifact="chat_messages",
            record_id="msg_1",
            field="text",
            pii_type="email",
            match_hash="hash_1",
            redacted_preview="a***@example.com",
            char_start=23,
            char_end=6,
        )


def test_pii_redaction_record_requires_fields_and_replacements():
    record = PIIRedactionRecord(
        redaction_id="redact_1",
        artifact="chat_messages",
        record_id="msg_1",
        fields=["text"],
        replacement_count=1,
        output_path="data/reports/chat_messages.redacted.jsonl",
    )

    assert record.schema_version == "pii.redaction.v1"
    assert SCHEMA_REGISTRY["pii_redactions"] is PIIRedactionRecord

    with pytest.raises(ValidationError):
        PIIRedactionRecord(
            redaction_id="redact_2",
            artifact="chat_messages",
            record_id="msg_1",
            fields=[],
            replacement_count=0,
            output_path="data/reports/chat_messages.redacted.jsonl",
        )


def test_claim_repair_suggestion_record_requires_reason_codes():
    record = ClaimRepairSuggestionRecord(
        repair_id="repair_1",
        claim_id="claim_1",
        source_id="src_1",
        evidence_id="ev_1",
        reason_codes=["evidence_not_exact_substring"],
        original_evidence_text="The vessel   Hope appears old.",
        suggested_evidence_text="The vessel Hope appears old.",
        support_scope="span",
    )

    assert record.schema_version == "claim.repair_suggestion.v1"
    assert SCHEMA_REGISTRY["claim_repairs"] is ClaimRepairSuggestionRecord

    with pytest.raises(ValidationError):
        ClaimRepairSuggestionRecord(
            repair_id="repair_2",
            claim_id="claim_1",
            source_id="src_1",
            evidence_id="ev_1",
            reason_codes=[],
            original_evidence_text="The vessel   Hope appears old.",
            suggested_evidence_text="The vessel Hope appears old.",
            support_scope="span",
        )


def test_claim_duplicate_group_record_requires_consistent_member_count():
    record = ClaimDuplicateGroupRecord(
        dedupe_id="dedupe_1",
        normalized_proposition={
            "subject": "speaker:bob",
            "predicate": "asserts",
            "object": "Hope had masts.",
        },
        normalized_claim={"subject": "speaker:bob", "predicate": "asserts", "object": "Hope had masts."},
        member_count=2,
        member_claim_ids=["claim_1", "claim_2"],
        member_normalized_claim_ids=["nclaim_1", "nclaim_2"],
        source_ids=["src_1"],
        evidence_ids=["ev_1", "ev_2"],
    )

    assert record.schema_version == "claim.dedupe.v1"
    assert record.source_count == 1
    assert record.evidence_count == 2
    assert record.cross_source is False
    assert record.duplicate_level == "same_normalized_proposition"
    assert SCHEMA_REGISTRY["claim_duplicates"] is ClaimDuplicateGroupRecord

    with pytest.raises(ValidationError):
        ClaimDuplicateGroupRecord(
            dedupe_id="dedupe_2",
            normalized_proposition={
                "subject": "speaker:bob",
                "predicate": "asserts",
                "object": "Hope had masts.",
            },
            normalized_claim={"subject": "speaker:bob", "predicate": "asserts", "object": "Hope had masts."},
            member_count=2,
            member_claim_ids=["claim_1"],
            member_normalized_claim_ids=["nclaim_1", "nclaim_2"],
            source_ids=["src_1"],
            evidence_ids=["ev_1", "ev_2"],
        )


def test_gold_evaluation_record_requires_valid_rates():
    record = GoldEvaluationRecord(
        evaluation_id="gold_eval_1",
        gold_path="gold.json",
        gold_claims=1,
        expected_accepted=1,
        produced_accepted=1,
        accepted_matches=1,
        accepted_false_positives=0,
        accepted_missing=0,
        accepted_precision=1.0,
        accepted_recall=1.0,
        expected_quarantined=0,
        produced_quarantined=0,
        quarantine_matches=0,
        quarantine_false_positives=0,
        quarantine_missing=0,
        quarantine_precision=None,
        quarantine_recall=None,
        evidence_exact_match_rate=1.0,
        attribution_preservation_rate=1.0,
        uncertainty_preservation_rate=1.0,
        negation_preservation_rate=1.0,
        quantity_preservation_rate=1.0,
        unsupported_entity_rate=0.0,
        quarantine_rate=0.0,
    )

    assert record.schema_version == "gold.eval.v1"
    assert record.evidence_exact_match_rate == 1.0
    assert SCHEMA_REGISTRY["gold_eval"] is GoldEvaluationRecord

    with pytest.raises(ValidationError):
        GoldEvaluationRecord(
            evaluation_id="gold_eval_2",
            gold_path="gold.json",
            gold_claims=1,
            expected_accepted=1,
            produced_accepted=1,
            accepted_matches=1,
            accepted_false_positives=0,
            accepted_missing=0,
            accepted_precision=1.5,
            accepted_recall=1.0,
            expected_quarantined=0,
            produced_quarantined=0,
            quarantine_matches=0,
            quarantine_false_positives=0,
            quarantine_missing=0,
        )

    with pytest.raises(ValidationError):
        GoldEvaluationRecord(
            evaluation_id="gold_eval_3",
            gold_path="gold.json",
            gold_claims=1,
            expected_accepted=1,
            produced_accepted=1,
            accepted_matches=1,
            accepted_false_positives=0,
            accepted_missing=0,
            accepted_precision=1.0,
            accepted_recall=1.0,
            expected_quarantined=0,
            produced_quarantined=0,
            quarantine_matches=0,
            quarantine_false_positives=0,
            quarantine_missing=0,
            evidence_exact_match_rate=1.5,
        )
