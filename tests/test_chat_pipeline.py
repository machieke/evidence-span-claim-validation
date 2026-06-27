import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def _write_chat_export(path: Path) -> None:
    payload = {
        "conversation_id": "conv_1",
        "thread_id": "thread_1",
        "metadata": {"platform": "fixture"},
        "messages": [
            {
                "id": "msg_1",
                "sender_id": "user_a",
                "sender_display_name": "Alice",
                "sender_role": "user",
                "timestamp": "2026-06-24T08:00:00Z",
                "text": "Did Hope have masts?",
            },
            {
                "id": "msg_2",
                "sender_id": "user_b",
                "sender_display_name": "Bob",
                "sender_role": "external",
                "timestamp": "2026-06-24T08:01:00Z",
                "text": "I saw Hope yesterday. It had three masts.",
            },
            {
                "id": "msg_3",
                "sender_id": "user_a",
                "sender_display_name": "Alice",
                "sender_role": "user",
                "timestamp": "2026-06-24T08:02:00Z",
                "text": "Thanks",
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_chat_pipeline_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        export = Path("chat.json")
        _write_chat_export(export)

        commands = [
            ["ingest-chat", "chat.json"],
            ["build-chat-evidence"],
            ["chunk-chat", "--previous-messages", "1"],
            ["detect-chat-spans"],
            ["extract-claims", "--modality", "chat"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        assert len(list(read_jsonl(Path("data/jsonl/sources.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/chat_messages.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/evidence.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/chunks.jsonl")))) == 3

        spans = [payload for _, payload in read_jsonl(Path("data/jsonl/spans.jsonl"))]
        assert [span["text"] for span in spans] == [
            "Did Hope have masts?",
            "I saw Hope yesterday.",
            "It had three masts.",
        ]
        assert "question_speech_act" in spans[0]["risk_flags"]
        assert "context_dependent_coreference" in spans[2]["risk_flags"]
        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 3
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 3

        trace = runner.invoke(app, ["trace-claim", raw_claims[0]["claim_id"]])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert trace_payload["found"] is True
        assert trace_payload["raw_claim"]["claim_id"] == raw_claims[0]["claim_id"]
        assert trace_payload["evidence"]["evidence_id"] == raw_claims[0]["evidence_id"]
        assert trace_payload["source"]["source_modality"] == "chat"
        assert [job["stage"] for job in trace_payload["jobs"]] == [
            "extract_claims",
            "validate_claims",
            "normalize_claims",
        ]

        graph = runner.invoke(app, ["export-graph"])
        assert graph.exit_code == 0, graph.stdout
        edges = [payload for _, payload in read_jsonl(Path("data/reports/claim_graph.jsonl"))]
        assert len(edges) == 3
        assert all(edge["schema_version"] == "graph.edge.v1" for edge in edges)
        assert {edge["predicate"] for edge in edges} >= {"asserts", "asks_whether"}
        assert all(edge["truth_status"] == "speaker_asserted_unverified" for edge in edges)
        assert all(edge["attribution"]["type"] == "speaker" for edge in edges)
        graph_trace = runner.invoke(app, ["trace-claim", raw_claims[0]["claim_id"]])
        assert graph_trace.exit_code == 0, graph_trace.stdout
        graph_trace_payload = json.loads(graph_trace.stdout)
        assert [edge["claim_id"] for edge in graph_trace_payload["graph_edges"]] == [raw_claims[0]["claim_id"]]

        validated_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.validated.jsonl"))]
        gold_claims = [
            {
                "evidence_id": claim["evidence_id"],
                "evidence_text": claim["evidence_text"],
                "expected_status": "accepted",
            }
            for claim in validated_claims
        ]
        gold_claims.append(
            {
                "evidence_id": "ev_missing",
                "evidence_text": "Missing expected claim.",
                "expected_status": "accepted",
            }
        )
        Path("gold.json").write_text(json.dumps({"claims": gold_claims}), encoding="utf-8")
        gold = runner.invoke(app, ["eval-gold", "gold.json"])
        assert gold.exit_code == 0, gold.stdout
        gold_report = Path("data/reports/gold_eval.md").read_text(encoding="utf-8")
        assert "| Accepted precision | 100.0% |" in gold_report
        assert "| Accepted recall | 75.0% |" in gold_report

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "# Evidence Pipeline Extraction Summary" in report_text
        assert "| claim_graph | 3 |" in report_text
        assert "| claims_validated | 3 |" in report_text
        assert "| claims_normalized | 3 |" in report_text

        validate = runner.invoke(app, ["validate-artifacts"])
        assert validate.exit_code == 0, validate.stdout
