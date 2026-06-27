import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_image_ocr_text_flows_through_text_claim_validation(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("sign.png")
        Image.new("RGB", (64, 32), color=(255, 255, 255)).save(image_path)

        ingest = runner.invoke(app, ["ingest-images", "sign.png"])
        assert ingest.exit_code == 0, ingest.stdout
        image = next(payload for _, payload in read_jsonl(Path("data/jsonl/images.jsonl")))

        ocr_payload = [
            {
                "image_id": image["image_id"],
                "text": "Dock 4 is closed at 5.",
                "bbox": [4, 4, 40, 10],
                "ocr_confidence": 0.92,
                "ocr_model": "fixture_ocr_v1",
            },
            {
                "image_id": image["image_id"],
                "text": "Gate 2 is open.",
                "bbox": [4, 18, 30, 8],
                "ocr_confidence": 0.5,
                "ocr_model": "fixture_ocr_v1",
            },
        ]
        Path("ocr.json").write_text(json.dumps({"ocr": ocr_payload}), encoding="utf-8")

        commands = [
            ["ingest-image-ocr", "ocr.json"],
            ["chunk-image-ocr"],
            ["detect-image-ocr-spans"],
            ["extract-claims", "--modality", "image"],
            ["validate-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        evidence = [payload for _, payload in read_jsonl(Path("data/jsonl/evidence.jsonl"))]
        assert [record["evidence_type"] for record in evidence] == ["ocr_text_span", "ocr_text_span"]
        assert evidence[0]["risk_flags"] == []
        assert evidence[1]["risk_flags"] == ["low_ocr_confidence"]

        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        assert [span["text"] for span in spans] == ["Dock 4 is closed at 5.", "Gate 2 is open."]

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 2
        assert all(claim["claim_type"] == "ocr_text_claim" for claim in raw_claims)
        assert all(claim["model"]["model"] == "image_ocr.rules.v1" for claim in raw_claims)
        assert [claim["evidence_text"] for claim in raw_claims] == [
            "Dock 4 is closed at 5.",
            "Gate 2 is open.",
        ]

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert [record["status"] for record in validations] == ["accepted_extracted", "quarantined"]
        assert validations[0]["errors"] == []
        assert validations[1]["errors"] == ["low_ocr_confidence"]

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert len(quarantined) == 1
        assert quarantined[0]["reason_codes"] == ["low_ocr_confidence"]

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
