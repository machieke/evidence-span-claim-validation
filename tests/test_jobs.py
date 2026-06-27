from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.cli import app
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord


runner = CliRunner()


def test_core_stage_commands_write_idempotent_job_records(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_msg_1",
                source_id="src_chat_1",
                source_modality="chat",
                evidence_type="message_span",
                text="Hope had three masts.",
                provenance={"message_id": "msg_1", "sender_id": "alice"},
            ),
        )
        append_jsonl(
            Path("data/jsonl/spans.jsonl"),
            SpanRecord(
                span_id="span_1",
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

        commands = [
            ["extract-claims", "--modality", "chat"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs] == ["extract_claims", "validate_claims", "normalize_claims"]
        assert all(job["status"] == "succeeded" for job in jobs)
        assert all(job["attempts"] == 1 for job in jobs)
        assert all(job["config_hash"].startswith("cfg_") for job in jobs)
        assert jobs[0]["input_record_ids"] == ["modality:chat"]
        assert jobs[0]["metrics"] == {"claims_created": 1, "claims_skipped": 0}
        assert jobs[1]["metrics"] == {
            "claims_accepted": 1,
            "claims_quarantined": 0,
            "claims_skipped": 0,
        }
        assert jobs[2]["metrics"] == {"claims_normalized": 1, "claims_skipped": 0}

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 3 |" in report_text
        assert "| extract_claims | 1 |" in report_text
        assert "| validate_claims | 1 |" in report_text
        assert "| normalize_claims | 1 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
