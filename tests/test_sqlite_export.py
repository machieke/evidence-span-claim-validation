import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


runner = CliRunner()


def test_export_sqlite_writes_artifact_tables_and_counts(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        evidence = EvidenceRecord(
            evidence_id="ev_1",
            source_id="src_1",
            source_modality="chat",
            evidence_type="message_span",
            text="Hope had three masts.",
            provenance={"message_id": "msg_1", "sender_id": "alice"},
        )
        claim = RawClaimRecord(
            claim_id="claim_1",
            source_id="src_1",
            source_modality="chat",
            evidence_id="ev_1",
            source_faithful_claim="The speaker asserted: Hope had three masts.",
            modality="asserted",
            evidence_text="Hope had three masts.",
            attribution={"type": "speaker", "agent": "alice"},
            truth_status="speaker_asserted_unverified",
            confidence=0.9,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)
        append_jsonl(
            Path("data/reports/claim_graph.jsonl"),
            {
                "edge_id": "edge_1",
                "normalized_claim_id": "nclaim_1",
                "claim_id": "claim_1",
                "source_id": "src_1",
                "evidence_id": "ev_1",
                "subject": "speaker:alice",
                "predicate": "asserts",
                "object": "Hope had three masts.",
                "truth_status": "speaker_asserted_unverified",
                "schema_version": "graph.edge.v1",
            },
        )
        append_jsonl(
            Path("data/reports/model_routing.jsonl"),
            {
                "routing_id": "route_1",
                "stage": "validate_claims",
                "record_type": "claim_raw",
                "record_id": "claim_1",
                "source_id": "src_1",
                "source_modality": "chat",
                "model_role": "validation",
                "selected_tier": "default",
                "selected_model": "cheap_validator_model",
                "reasons": [],
                "score": 0.9,
                "schema_version": "model.routing.v1",
            },
        )

        first = runner.invoke(app, ["export-sqlite"])
        second = runner.invoke(app, ["export-sqlite"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "records=4" in first.stdout

        database_path = Path("data/reports/pipeline.sqlite")
        assert database_path.exists()
        with sqlite3.connect(database_path) as connection:
            evidence_count = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
            claims_count = connection.execute("SELECT COUNT(*) FROM claims_raw").fetchone()[0]
            graph_count = connection.execute("SELECT COUNT(*) FROM claim_graph").fetchone()[0]
            routing_count = connection.execute("SELECT COUNT(*) FROM model_routing").fetchone()[0]
            jobs_count = connection.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
            review_count = connection.execute("SELECT COUNT(*) FROM review_decisions").fetchone()[0]
            audit_count = connection.execute("SELECT COUNT(*) FROM audit_events").fetchone()[0]
            artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("claims_raw",),
            ).fetchone()[0]
            graph_artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("claim_graph",),
            ).fetchone()[0]
            routing_artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("model_routing",),
            ).fetchone()[0]
            payload_json = connection.execute(
                "SELECT payload_json FROM claims_raw WHERE record_key = ?",
                ("claim_1",),
            ).fetchone()[0]
            graph_payload_json = connection.execute(
                "SELECT payload_json FROM claim_graph WHERE record_key = ?",
                ("edge_1",),
            ).fetchone()[0]
            routing_payload_json = connection.execute(
                "SELECT payload_json FROM model_routing WHERE record_key = ?",
                ("route_1",),
            ).fetchone()[0]

        assert evidence_count == 1
        assert claims_count == 1
        assert graph_count == 1
        assert routing_count == 1
        assert jobs_count == 0
        assert review_count == 0
        assert audit_count == 0
        assert artifact_count == 1
        assert graph_artifact_count == 1
        assert routing_artifact_count == 1
        assert json.loads(payload_json)["claim_id"] == "claim_1"
        assert json.loads(graph_payload_json)["edge_id"] == "edge_1"
        assert json.loads(routing_payload_json)["routing_id"] == "route_1"
