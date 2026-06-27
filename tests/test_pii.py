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

        invalid = runner.invoke(app, ["detect-pii", "--artifact", "images"])
        assert invalid.exit_code != 0
        assert "PII detection supports artifacts" in invalid.stdout
