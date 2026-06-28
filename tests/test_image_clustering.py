from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


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
        cluster_evidence = next(record for record in evidence if record["evidence_type"] == "visual_cluster")
        assert len(cluster_evidence["provenance"]["representative_crop_paths"]) == 2
        assert all(
            Path(crop_path).exists()
            for crop_path in cluster_evidence["provenance"]["representative_crop_paths"]
        )

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 3
        assert {record["claim_type"] for record in raw_claims} == {
            "unnamed_visual_feature_cluster",
            "visual_region_proposal",
        }

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        cluster_validation = next(
            record for record in validations if record["claim_id"].startswith("claim_vf_")
        )
        assert cluster_validation["errors"] == [
            "image_cluster_too_small",
            "image_cluster_insufficient_cross_source",
        ]
        assert quarantined[0]["reason_codes"] == cluster_validation["errors"]
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 2
        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs] == [
            "ingest_images",
            "propose_image_regions",
            "build_image_evidence",
            "embed_image_regions",
            "cluster_image_regions",
            "build_image_cluster_evidence",
            "extract_claims",
            "validate_claims",
            "normalize_claims",
            "export_graph",
        ]
        assert jobs[4]["model_id"] == "connected_components_color_distance_v1+color_rgb_mean_std_v1"
        assert jobs[4]["metrics"] == {"clustered_regions": 2, "clusters_created": 1, "clusters_skipped": 0}

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout


def test_large_cross_source_visual_cluster_is_accepted(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        evidence = EvidenceRecord(
            evidence_id="ev_cluster_1",
            source_id="src_img_1",
            source_modality="image",
            evidence_type="visual_cluster",
            text=None,
            provenance={
                "feature_cluster_id": "cluster_1",
                "cluster_size": 5,
                "cohesion_score": 0.91,
                "source_ids": ["src_img_1", "src_img_2", "src_img_3"],
                "member_region_ids": ["r1", "r2", "r3", "r4", "r5"],
            },
        )
        claim = RawClaimRecord(
            claim_id="claim_vf_cluster_1",
            source_id="src_img_1",
            source_modality="image",
            evidence_id="ev_cluster_1",
            claim_type="unnamed_visual_feature_cluster",
            source_faithful_claim="Regions r1, r2, r3, r4, r5 were clustered as visually similar.",
            subject="cluster_1",
            predicate="has_member_regions",
            object=["r1", "r2", "r3", "r4", "r5"],
            attributes={"cohesion_score": 0.91},
            modality="model_observation",
            attribution={"type": "model", "agent": "clusterer_v1"},
            truth_status="model_observation_unverified",
            confidence=0.91,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=1" in result.stdout
        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert validations[0]["status"] == "accepted_extracted"
        assert validations[0]["errors"] == []
