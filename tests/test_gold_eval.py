import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl
from evidence_pipeline.schemas.claims import ClaimValidationSummary, ValidatedClaimRecord
from evidence_pipeline.schemas.validation import QuarantineRecord


runner = CliRunner()


def test_gold_eval_reports_quarantine_precision_and_recall(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/claims.validated.jsonl"),
            ValidatedClaimRecord(
                claim_id="claim_accepted",
                source_id="src_1",
                source_modality="chat",
                evidence_id="ev_accepted",
                source_faithful_claim="The speaker asserted: Accepted claim.",
                evidence_text="Accepted claim.",
                modality="asserted",
                truth_status="speaker_asserted_unverified",
                support_status="accepted_extracted",
                validation=ClaimValidationSummary(deterministic_valid=True),
            ),
        )
        append_jsonl(
            Path("data/jsonl/quarantine.jsonl"),
            QuarantineRecord(
                quarantine_id="q_match",
                record_type="claim",
                record_id="claim_quarantined_match",
                claim_id="claim_quarantined_match",
                stage="validate_claims",
                reason_codes=["evidence_not_exact_substring"],
                payload={"evidence_id": "ev_quarantined", "evidence_text": "Quarantined claim."},
            ),
        )
        append_jsonl(
            Path("data/jsonl/quarantine.jsonl"),
            QuarantineRecord(
                quarantine_id="q_extra",
                record_type="claim",
                record_id="claim_quarantined_extra",
                claim_id="claim_quarantined_extra",
                stage="validate_claims",
                reason_codes=["unsupported_entities_introduced"],
                payload={"evidence_id": "ev_extra", "evidence_text": "Extra quarantine."},
            ),
        )
        Path("gold.json").write_text(
            json.dumps(
                {
                    "claims": [
                        {
                            "evidence_id": "ev_accepted",
                            "evidence_text": "Accepted claim.",
                            "expected_status": "accepted",
                        },
                        {
                            "evidence_id": "ev_quarantined",
                            "evidence_text": "Quarantined claim.",
                            "expected_status": "quarantined",
                        },
                        {
                            "evidence_id": "ev_missing_quarantine",
                            "evidence_text": "Missing quarantine.",
                            "expected_status": "quarantined",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = runner.invoke(app, ["eval-gold", "gold.json"])

        assert result.exit_code == 0, result.stdout
        assert "quarantine_precision=0.5" in result.stdout
        assert "quarantine_recall=0.5" in result.stdout
        report = Path("data/reports/gold_eval.md").read_text(encoding="utf-8")
        assert "| Quarantine precision | 50.0% |" in report
        assert "| Quarantine recall | 50.0% |" in report
        assert "| Quarantine false positives | 1 |" in report
        assert "| Quarantine missing | 1 |" in report
        assert "ev_missing_quarantine" in report
        assert "ev_extra" in report
