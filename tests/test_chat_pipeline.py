import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def _write_chat_export(path: Path) -> None:
    payload = {
        "conversation_id": "conv_1",
        "thread_id": "thread_1",
        "metadata": {"platform": "fixture"},
        "messages": [
            {
                "id": "msg_1",
                "sender_id": "user_a",
                "sender_display_name": "Alice",
                "sender_role": "user",
                "timestamp": "2026-06-24T08:00:00Z",
                "text": "Did Hope have masts?",
            },
            {
                "id": "msg_2",
                "sender_id": "user_b",
                "sender_display_name": "Bob",
                "sender_role": "external",
                "timestamp": "2026-06-24T08:01:00Z",
                "text": "I saw Hope yesterday. It had three masts.",
            },
            {
                "id": "msg_3",
                "sender_id": "user_a",
                "sender_display_name": "Alice",
                "sender_role": "user",
                "timestamp": "2026-06-24T08:02:00Z",
                "text": "Thanks",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_chat_pipeline_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        export = Path("chat.json")
        _write_chat_export(export)

        commands = [
            ["ingest-chat", "chat.json"],
            ["build-chat-evidence"],
            ["chunk-chat", "--previous-messages", "1"],
            ["detect-chat-spans"],
            ["extract-claims", "--modality", "chat"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        assert len(list(read_jsonl(Path("data/jsonl/sources.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/chat_messages.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/chunks.jsonl")))) == 3

        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        assert [span["text"] for span in spans] == [
            "Did Hope have masts?",
            "I saw Hope yesterday.",
            "It had three masts.",
        ]
        assert "question_speech_act" in spans[0]["risk_flags"]
        assert "context_dependent_coreference" in spans[2]["risk_flags"]
        assert len(list(read_jsonl(Path("data/jsonl/claims.raw.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 3

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "# Evidence Pipeline Extraction Summary" in report_text
        assert "| claims_validated | 3 |" in report_text
        assert "| claims_normalized | 3 |" in report_text

        validate = runner.invoke(app, ["validate-artifacts"])
        assert validate.exit_code == 0, validate.stdout
