import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_run_chat_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("chat.json").write_text(
            json.dumps(
                {
                    "conversation_id": "conv_runner",
                    "messages": [
                        {
                            "id": "msg_1",
                            "sender_id": "alice",
                            "timestamp": "2026-06-24T08:00:00Z",
                            "text": "Did Hope have masts?",
                        },
                        {
                            "id": "msg_2",
                            "sender_id": "bob",
                            "timestamp": "2026-06-24T08:01:00Z",
                            "text": "Hope had three masts.",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )

        first = runner.invoke(app, ["run-chat", "chat.json", "--previous-messages", "1"])
        second = runner.invoke(app, ["run-chat", "chat.json", "--previous-messages", "1"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "messages_created=2" in first.stdout
        assert "messages_created=0" in second.stdout
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 2
        assert Path("data/reports/extraction_summary.md").exists()


def test_run_images_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (16, 16), color=(32, 64, 128)).save(image_path)

        first = runner.invoke(app, ["run-images", "image.png", "--patch-size", "16", "--stride", "16"])
        second = runner.invoke(app, ["run-images", "image.png", "--patch-size", "16", "--stride", "16"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "images_created=1" in first.stdout
        assert "images_created=0" in second.stdout
        assert len(list(read_jsonl(Path("data/jsonl/images.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/image_regions.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 1
        assert Path("data/reports/extraction_summary.md").exists()
