import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_generic_stage_commands_dispatch_chat_pipeline(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("chat.json").write_text(
            json.dumps(
                {
                    "conversation_id": "conv_generic",
                    "messages": [
                        {
                            "id": "msg_1",
                            "sender_id": "alice",
                            "timestamp": "2026-06-24T08:00:00Z",
                            "text": "Hope had three masts.",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        ingest = runner.invoke(app, ["ingest-chat", "chat.json"])
        assert ingest.exit_code == 0, ingest.stdout

        commands = [
            ["build-evidence", "--modality", "chat"],
            ["chunk", "--modality", "chat", "--previous-messages", "0"],
            ["detect-spans", "--modality", "chat"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/chunks.jsonl")))) == 1
        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        assert [span["text"] for span in spans] == ["Hope had three masts."]

        invalid = runner.invoke(app, ["chunk", "--modality", "video"])
        assert invalid.exit_code != 0
        assert "chunk supports" in invalid.stdout


def test_generic_stage_commands_dispatch_image_ocr_pipeline(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("sign.png")
        Image.new("RGB", (32, 32), color=(255, 255, 255)).save(image_path)
        ingest = runner.invoke(app, ["ingest-images", "sign.png"])
        assert ingest.exit_code == 0, ingest.stdout
        image = next(payload for _, payload in read_jsonl(Path("data/jsonl/images.jsonl")))
        Path("ocr.json").write_text(
            json.dumps(
                {
                    "ocr": [
                        {
                            "image_id": image["image_id"],
                            "text": "Dock 4 is closed.",
                            "bbox": [1, 1, 20, 8],
                            "ocr_confidence": 0.95,
                            "ocr_model": "fixture_ocr_v1",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        ocr = runner.invoke(app, ["ingest-image-ocr", "ocr.json"])
        assert ocr.exit_code == 0, ocr.stdout

        commands = [
            ["chunk", "--modality", "image"],
            ["detect-spans", "--modality", "image"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        chunks = [payload for _, payload in read_jsonl(Path("data/jsonl/chunks.jsonl"))]
        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        assert len(chunks) == 1
        assert [span["text"] for span in spans] == ["Dock 4 is closed."]
