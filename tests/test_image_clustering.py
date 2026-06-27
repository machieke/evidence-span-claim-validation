from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_image_region_clustering_emits_cluster_claims(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (32, 16), color=(200, 20, 20)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["embed-image-regions"],
            ["cluster-image-regions", "--distance-threshold", "0.01", "--min-cluster-size", "2"],
            ["build-image-cluster-evidence"],
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

        embeddings = [payload for _, payload in read_jsonl(Path("data/jsonl/image_region_embeddings.jsonl"))]
        assert len(embeddings) == 2
        assert all(record["embedding_model"] == "color_rgb_mean_std_v1" for record in embeddings)
        assert all(record["embedding_dim"] == 6 for record in embeddings)

        clusters = [payload for _, payload in read_jsonl(Path("data/jsonl/image_feature_clusters.jsonl"))]
        assert len(clusters) == 1
        assert clusters[0]["cluster_size"] == 2
        assert len(clusters[0]["member_region_ids"]) == 2
        assert clusters[0]["status"] == "unnamed"

        evidence = [payload for _, payload in read_jsonl(Path("data/jsonl/evidence.jsonl"))]
        assert len(evidence) == 3
        assert sorted(record["evidence_type"] for record in evidence) == [
            "visual_cluster",
            "visual_region",
            "visual_region",
        ]

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 3
        assert {record["claim_type"] for record in raw_claims} == {
            "unnamed_visual_feature_cluster",
            "visual_region_proposal",
        }

        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 3

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
