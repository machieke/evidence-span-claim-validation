from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_image_region_pipeline_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (32, 32), color=(128, 64, 32)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["extract-claims", "--modality", "image"],
            ["validate-claims"],
            ["normalize-claims"],
            ["export-graph"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        assert len(list(read_jsonl(Path("data/jsonl/sources.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/images.jsonl")))) == 1

        regions = [payload for _, payload in read_jsonl(Path("data/jsonl/image_regions.jsonl"))]
        assert len(regions) == 4
        assert sorted(region["bbox"] for region in regions) == [
            [0, 0, 16, 16],
            [0, 16, 16, 16],
            [16, 0, 16, 16],
            [16, 16, 16, 16],
        ]
        for region in regions:
            assert Path(region["crop_path"]).exists()

        evidence = [payload for _, payload in read_jsonl(Path("data/jsonl/evidence.jsonl"))]
        assert len(evidence) == 4
        assert all(record["source_modality"] == "image" for record in evidence)
        assert all(record["evidence_type"] == "visual_region" for record in evidence)
        assert all("text" not in record for record in evidence)

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 4
        assert all(record["source_modality"] == "image" for record in raw_claims)
        assert all(record["claim_type"] == "visual_region_proposal" for record in raw_claims)
        assert all(record["modality"] == "model_observation" for record in raw_claims)
        assert all(record["truth_status"] == "model_observation_unverified" for record in raw_claims)
        assert all("evidence_text" not in record for record in raw_claims)

        validated_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.validated.jsonl"))]
        assert len(validated_claims) == 4
        assert all(record["support_status"] == "accepted_extracted" for record in validated_claims)

        normalized_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.normalized.jsonl"))]
        assert len(normalized_claims) == 4

        graph_edges = [payload for _, payload in read_jsonl(Path("data/reports/claim_graph.jsonl"))]
        assert len(graph_edges) == 4

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| images | 1 |" in report_text
        assert "| image_regions | 4 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
