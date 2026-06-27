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

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| images | 1 |" in report_text
        assert "| image_regions | 4 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
