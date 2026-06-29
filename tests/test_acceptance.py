import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord


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
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_chat_pipeline() -> None:
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
        result = runner.invoke(app, command)
        assert result.exit_code == 0, result.stdout


def test_acceptance_check_passes_for_complete_chat_pipeline(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_chat_export(Path("chat.json"))
        _run_chat_pipeline()

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout

        first = runner.invoke(app, ["acceptance-check"])
        second = runner.invoke(app, ["acceptance-check"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "failed_checks=0" in first.stdout
        assert "passed=True" in first.stdout

        checks = [payload for _, payload in read_jsonl(Path("data/reports/acceptance_check.jsonl"))]
        check_by_id = {check["check_id"]: check for check in checks}
        assert check_by_id["source_records_registered"]["status"] == "passed"
        assert check_by_id["accepted_text_claims_exact_evidence"]["status"] == "passed"
        assert check_by_id["accepted_chat_audio_claims_attributed"]["status"] == "passed"
        assert check_by_id["normalized_claims_from_accepted_claims"]["status"] == "passed"
        assert check_by_id["summary_report_exists"]["status"] == "passed"

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        acceptance_jobs = [job for job in jobs if job["stage"] == "acceptance_check"]
        assert len(acceptance_jobs) == 1
        assert acceptance_jobs[0]["model_id"] == "acceptance.check.v1"
        assert acceptance_jobs[0]["metrics"] == {
            "checks": len(checks),
            "failed_checks": 0,
        }

        summary = runner.invoke(app, ["report"])
        assert summary.exit_code == 0, summary.stdout
        summary_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "## Acceptance Checks" in summary_text
        assert "| passed |" in summary_text

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout

        export = runner.invoke(app, ["export-sqlite"])
        assert export.exit_code == 0, export.stdout
        with sqlite3.connect(Path("data/reports/pipeline.sqlite")) as connection:
            acceptance_count = connection.execute("SELECT COUNT(*) FROM acceptance_check").fetchone()[0]
        assert acceptance_count == len(checks)


def test_acceptance_check_fails_for_normalized_non_accepted_claim(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        _write_chat_export(Path("chat.json"))
        _run_chat_pipeline()
        append_jsonl(
            Path("data/jsonl/claims.normalized.jsonl"),
            NormalizedClaimRecord(
                normalized_claim_id="nclaim_bad",
                claim_id="claim_missing",
                source_id="src_chat_1",
                evidence_id="ev_missing",
                normalized_claim={
                    "subject": "entity:hope",
                    "predicate": "had",
                    "object": "three masts",
                },
            ),
        )

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout

        result = runner.invoke(app, ["acceptance-check"])

        assert result.exit_code == 1, result.stdout
        assert "passed=False" in result.stdout
        checks = [payload for _, payload in read_jsonl(Path("data/reports/acceptance_check.jsonl"))]
        failed = {check["check_id"]: check for check in checks if check["status"] == "failed"}
        assert set(failed) == {"normalized_claims_from_accepted_claims"}
        assert failed["normalized_claims_from_accepted_claims"]["details"] == [
            {
                "claim_id": "claim_missing",
                "normalized_claim_id": "nclaim_bad",
                "reason": "normalized_from_non_accepted_claim",
            }
        ]

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
