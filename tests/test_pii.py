import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.chat import ChatMessageRecord


runner = CliRunner()


def test_detect_pii_writes_redacted_findings_without_raw_matches(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/chat_messages.jsonl"),
            ChatMessageRecord(
                message_id="msg_1",
                source_id="src_chat_1",
                conversation_id="conv_1",
                sender_id="alice",
                text="Email alice@example.com or call 415-555-1212.",
            ),
        )

        first = runner.invoke(app, ["detect-pii", "--artifact", "chat_messages"])
        second = runner.invoke(app, ["detect-pii", "--artifact", "chat_messages"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "findings=2" in first.stdout

        output_path = Path("data/reports/pii_findings.jsonl")
        output_text = output_path.read_text(encoding="utf-8")
        findings = [payload for _, payload in read_jsonl(output_path)]
        assert len(findings) == 2
        assert {finding["pii_type"] for finding in findings} == {"email", "phone"}
        assert all(finding["artifact"] == "chat_messages" for finding in findings)
        assert all("match_hash" in finding for finding in findings)
        assert "alice@example.com" not in output_text
        assert "415-555-1212" not in output_text
        assert "a***@example.com" in output_text
        assert "***-***-1212" in output_text

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| pii_findings | 2 |" in report_text
        assert "## PII Findings By Type" in report_text
        assert "| email | 1 |" in report_text
        assert "| phone | 1 |" in report_text

        export = runner.invoke(app, ["export-sqlite"])
        assert export.exit_code == 0, export.stdout
        with sqlite3.connect(Path("data/reports/pipeline.sqlite")) as connection:
            finding_count = connection.execute("SELECT COUNT(*) FROM pii_findings").fetchone()[0]
            artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("pii_findings",),
            ).fetchone()[0]
            payload_json = connection.execute(
                "SELECT payload_json FROM pii_findings WHERE record_key = ?",
                (findings[0]["finding_id"],),
            ).fetchone()[0]

        exported_payload = json.loads(payload_json)
        assert finding_count == 2
        assert artifact_count == 2
        assert exported_payload["pii_type"] in {"email", "phone"}
        assert "alice@example.com" not in payload_json
        assert "415-555-1212" not in payload_json

        invalid = runner.invoke(app, ["detect-pii", "--artifact", "images"])
        assert invalid.exit_code != 0
        assert "PII detection supports artifacts" in invalid.stdout


def test_redact_pii_writes_redacted_copy_without_mutating_source(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        source_path = Path("data/jsonl/chat_messages.jsonl")
        append_jsonl(
            source_path,
            ChatMessageRecord(
                message_id="msg_1",
                source_id="src_chat_1",
                conversation_id="conv_1",
                sender_id="alice",
                text="Email alice@example.com or call 415-555-1212.",
            ),
        )

        first = runner.invoke(app, ["redact-pii", "--artifact", "chat_messages"])
        second = runner.invoke(app, ["redact-pii", "--artifact", "chat_messages"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "records=1" in first.stdout
        assert "replacements=2" in first.stdout

        output_path = Path("data/reports/chat_messages.redacted.jsonl")
        output_text = output_path.read_text(encoding="utf-8")
        redacted_records = [payload for _, payload in read_jsonl(output_path)]
        assert len(redacted_records) == 1
        assert redacted_records[0]["text"] == "Email [EMAIL] or call [PHONE]."
        assert "alice@example.com" not in output_text
        assert "415-555-1212" not in output_text

        source_text = source_path.read_text(encoding="utf-8")
        assert "alice@example.com" in source_text
        assert "415-555-1212" in source_text

        invalid = runner.invoke(app, ["redact-pii", "--artifact", "all"])
        assert invalid.exit_code != 0
        assert "requires one artifact at a time" in invalid.stdout
