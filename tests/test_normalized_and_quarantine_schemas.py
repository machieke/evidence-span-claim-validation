import pytest
from pydantic import ValidationError

from evidence_pipeline.schemas.claims import (
    ClaimValidationSummary,
    EntityResolution,
    NormalizedClaimRecord,
    ValidatedClaimRecord,
)
from evidence_pipeline.schemas.validation import QuarantineRecord, ValidationRecord


def test_normalized_claim_links_to_validated_claim_and_evidence():
    record = NormalizedClaimRecord(
        normalized_claim_id="nclaim_1",
        claim_id="claim_1",
        source_id="src_1",
        evidence_id="ev_1",
        normalized_claim={
            "subject": "entity:vessel_hope",
            "predicate": "appears_condition",
            "object": "condition:older_diesel_engine",
        },
    )

    assert record.schema_version == "claim.normalized.v1"


def test_normalized_claim_requires_core_proposition_keys():
    with pytest.raises(ValidationError):
        NormalizedClaimRecord(
            normalized_claim_id="nclaim_1",
            claim_id="claim_1",
            source_id="src_1",
            evidence_id="ev_1",
            normalized_claim={
                "subject": "entity:vessel_hope",
                "predicate": "appears_condition",
            },
        )


def test_entity_resolution_requires_auditable_identifiers():
    with pytest.raises(ValidationError):
        EntityResolution(surface="Hope", canonical_id="", confidence=0.9, basis="exact_name_match")


def test_quarantine_record_has_machine_readable_reasons():
    record = QuarantineRecord(
        quarantine_id="q_1",
        record_type="claim",
        record_id="claim_1",
        claim_id="claim_1",
        stage="validate_claims",
        reason_codes=["evidence_not_exact_substring"],
    )

    assert record.reason_codes == ["evidence_not_exact_substring"]


def test_quarantine_record_rejects_empty_reasons():
    with pytest.raises(ValidationError):
        QuarantineRecord(
            quarantine_id="q_1",
            record_type="claim",
            record_id="claim_1",
            claim_id="claim_1",
            stage="validate_claims",
        )


def test_rejected_validation_record_requires_errors():
    with pytest.raises(ValidationError):
        ValidationRecord(
            validation_id="val_1",
            claim_id="claim_1",
            stage="validate_claims",
            status="quarantined",
        )

    accepted = ValidationRecord(
        validation_id="val_2",
        claim_id="claim_2",
        stage="validate_claims",
        status="accepted_extracted",
    )
    assert accepted.errors == []


def test_accepted_validated_claim_requires_valid_summary():
    with pytest.raises(ValidationError):
        ValidatedClaimRecord(
            claim_id="claim_1",
            source_id="src_1",
            source_modality="chat",
            evidence_id="ev_1",
            source_faithful_claim="The speaker asserted: Hope had three masts.",
            evidence_text="Hope had three masts.",
            modality="asserted",
            truth_status="speaker_asserted_unverified",
            support_status="accepted_extracted",
            validation=ClaimValidationSummary(deterministic_valid=False),
        )
