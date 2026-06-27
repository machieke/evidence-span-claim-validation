from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_image_region_color_classification_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (16, 16), color=(220, 20, 20)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["embed-image-regions"],
            ["classify-image-regions"],
            ["validate-claims"],
            ["normalize-claims"],
            ["export-graph"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 1
        claim = raw_claims[0]
        assert claim["claim_type"] == "named_visual_classification"
        assert claim["object"] == "red"
        assert claim["predicate"] == "classified_as"
        assert claim["modality"] == "model_observation"
        assert claim["truth_status"] == "model_observation_unverified"
        assert claim["attribution"] == {"agent": "dominant_color_classifier_v1", "type": "model"}
        assert "color_only_classification" in claim["risk_flags"]

        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 1

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
