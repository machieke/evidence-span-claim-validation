from evidence_pipeline.reports.summary import _normalized_confidence_rate


def test_normalized_confidence_rate_counts_each_normalized_row():
    claims_validated = [
        {
            "claim_id": "claim_1",
            "support_status": "accepted_extracted",
            "confidence": 0.8,
        }
    ]
    claims_normalized = [
        {
            "claim_id": "claim_1",
            "normalized_claim": {
                "qualifiers": {
                    "confidence": 0.8,
                    "confidence_basis": "validated_claim_confidence",
                }
            },
        },
        {
            "claim_id": "claim_1",
            "normalized_claim": {
                "qualifiers": {
                    "confidence": 0.7,
                    "confidence_basis": "validated_claim_confidence",
                }
            },
        },
    ]

    assert _normalized_confidence_rate(claims_validated, claims_normalized) == "50.0%"
