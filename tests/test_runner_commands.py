import json
from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def _write_runner_chat_export(path: Path) -> None:
    path.write_text(
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


def test_run_chat_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_runner_chat_export(Path("chat.json"))

        first = runner.invoke(app, ["run-chat", "chat.json", "--previous-messages", "1"])
        second = runner.invoke(app, ["run-chat", "chat.json", "--previous-messages", "1"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "messages_created=2" in first.stdout
        assert "messages_created=0" in second.stdout
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 2
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 2
        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs] == [
            "ingest_chat",
            "build_chat_evidence",
            "chunk_chat",
            "detect_chat_spans",
            "extract_claims",
            "validate_claims",
            "normalize_claims",
            "export_graph",
        ]
        assert [job["model_id"] for job in jobs] == [
            "chat.ingest.v1",
            "chat_evidence.builder.v1",
            "chat_chunker.thread_window.v1",
            "chat_rules_v1",
            "rules.v1",
            "deterministic.v7",
            "normalizer.v1",
            "graph.export.v1",
        ]
        assert len({job.get("source_id") for job in jobs if job.get("source_id") is not None}) == 1
        assert jobs[2]["input_record_ids"][:2] == ["policy:max_tokens=1200", "policy:previous_messages=1"]
        assert Path("data/reports/extraction_summary.md").exists()
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 8 |" in report_text


def test_finalize_run_writes_acceptance_outputs_idempotently(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_runner_chat_export(Path("chat.json"))
        run = runner.invoke(app, ["run-chat", "chat.json", "--previous-messages", "1"])
        assert run.exit_code == 0, run.stdout

        first = runner.invoke(app, ["finalize-run"])
        second = runner.invoke(app, ["finalize-run"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "passed=True" in first.stdout
        assert "failed_checks=0" in first.stdout
        assert "artifact_failures=0" in first.stdout
        assert "sqlite=data/reports/pipeline.sqlite" in first.stdout

        checks = [payload for _, payload in read_jsonl(Path("data/reports/acceptance_check.jsonl"))]
        assert checks
        assert {check["status"] for check in checks} == {"passed"}
        assert Path("data/reports/pipeline.sqlite").exists()

        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "## Acceptance Checks" in report_text
        assert "| acceptance_check |" in report_text

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs].count("export_graph") == 1
        assert [job["stage"] for job in jobs].count("acceptance_check") == 1
        assert [job["stage"] for job in jobs].count("export_sqlite") == 1

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout


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
        assert "embeddings_created=1" in first.stdout
        assert "embeddings_created=0" in second.stdout
        assert "clusters_created=0" in first.stdout
        assert "clusters_created=0" in second.stdout
        assert "claims_created=1" in first.stdout
        assert "claims_created=0" in second.stdout
        assert len(list(read_jsonl(Path("data/jsonl/images.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/image_regions.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/image_region_embeddings.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/image_feature_clusters.jsonl")))) == 0
        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/claims.raw.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 1
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
        assert [job["model_id"] for job in jobs] == [
            "image.ingest.v1",
            "image_region_proposal.grid.v1",
            "image_region_evidence.builder.v1",
            "color_rgb_mean_std_v1",
            "connected_components_color_distance_v1+color_rgb_mean_std_v1",
            "image_cluster_evidence.builder.v1",
            "image_region.rules.v1+image_cluster.rules.v1",
            "deterministic.v7",
            "normalizer.v1",
            "graph.export.v1",
        ]
        assert len({job.get("source_id") for job in jobs if job.get("source_id") is not None}) == 1
        assert jobs[1]["metrics"] == {"regions_created": 1, "regions_skipped": 0}
        assert jobs[4]["metrics"] == {"clustered_regions": 0, "clusters_created": 0, "clusters_skipped": 1}
        assert Path("data/reports/extraction_summary.md").exists()
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 10 |" in report_text
