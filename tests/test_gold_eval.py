import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
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

        metrics = [
            payload
            for _, payload in read_jsonl(Path("data/reports/gold_eval.jsonl"))
        ]
        assert len(metrics) == 1
        assert metrics[0]["accepted_precision"] == 1.0
        assert metrics[0]["quarantine_precision"] == 0.5
        assert metrics[0]["quarantine_recall"] == 0.5

        summary = runner.invoke(app, ["report"])
        assert summary.exit_code == 0, summary.stdout
        summary_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| gold_eval | 1 |" in summary_text

        export = runner.invoke(app, ["export-sqlite"])
        assert export.exit_code == 0, export.stdout
        with sqlite3.connect(Path("data/reports/pipeline.sqlite")) as connection:
            gold_eval_count = connection.execute("SELECT COUNT(*) FROM gold_eval").fetchone()[0]
            artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("gold_eval",),
            ).fetchone()[0]
            payload_json = connection.execute(
                "SELECT payload_json FROM gold_eval WHERE record_key = ?",
                (metrics[0]["evaluation_id"],),
            ).fetchone()[0]

        exported_payload = json.loads(payload_json)
        assert gold_eval_count == 1
        assert artifact_count == 1
        assert exported_payload["quarantine_recall"] == 0.5
