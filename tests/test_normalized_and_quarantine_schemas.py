from evidence_pipeline.schemas.claims import NormalizedClaimRecord
from evidence_pipeline.schemas.validation import QuarantineRecord


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
