import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.sources import SourceRecord


runner = CliRunner()


def test_retention_plan_reports_old_raw_sources_without_deleting(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        old_file = Path("old_chat.json")
        current_file = Path("current_chat.json")
        old_file.write_text("old", encoding="utf-8")
        current_file.write_text("current", encoding="utf-8")

        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_old",
                source_modality="chat",
                source_file=str(old_file),
                ingested_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            ),
        )
        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_current",
                source_modality="chat",
                source_file=str(current_file),
                ingested_at=datetime.now(timezone.utc),
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_old_source",
                source_id="src_old",
                source_modality="chat",
                evidence_id="ev_old_source",
                source_faithful_claim="The speaker asserted: Old source contains a claim.",
                modality="asserted",
                evidence_text="Old source contains a claim.",
                attribution={"type": "speaker", "agent": "alice"},
                truth_status="speaker_asserted_unverified",
                confidence=0.9,
            ),
        )

        result = runner.invoke(app, ["retention-plan"])
        second = runner.invoke(app, ["retention-plan"])
        assert result.exit_code == 0, result.stdout
        assert second.exit_code == 0, second.stdout
        assert "candidates=1" in result.stdout

        output_path = Path("data/reports/retention_plan.jsonl")
        candidates = [payload for _, payload in read_jsonl(output_path)]
        assert len(candidates) == 1
        assert candidates[0]["source_id"] == "src_old"
        assert candidates[0]["action"] == "delete_raw_source"
        assert candidates[0]["dry_run"] is True
        assert candidates[0]["retention_days"] == 365
        assert old_file.exists()
        assert current_file.exists()

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "retention_plan"
        assert jobs[0]["model_id"] == "retention.plan.v1"
        assert jobs[0]["input_record_ids"] == ["sources"]
        assert jobs[0]["metrics"] == {"candidates": 1}

        audit_events = [payload for _, payload in read_jsonl(Path("data/jsonl/audit_events.jsonl"))]
        assert len(audit_events) == 1
        assert audit_events[0]["action"] == "retention_plan"
        assert audit_events[0]["target_type"] == "retention_candidate"
        assert audit_events[0]["target_id"] == candidates[0]["retention_id"]
        assert audit_events[0]["source_id"] == "src_old"
        assert audit_events[0]["details"]["reason_code"] == "raw_source_retention_exceeded"
        assert audit_events[0]["details"]["dry_run"] is True

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| audit_events | 1 |" in report_text
        assert "## Jobs By Stage" in report_text
        assert "| retention_plan | 1 |" in report_text
        assert "## Audit Events" in report_text
        assert "## Retention Plan Reasons" in report_text
        assert "| raw_source_retention_exceeded | 1 |" in report_text

        trace = runner.invoke(app, ["trace-claim", "claim_old_source"])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert (
            trace_payload["retention_plan"][0]["retention_id"]
            == candidates[0]["retention_id"]
        )
        assert trace_payload["audit_events"][0]["target_id"] == candidates[0]["retention_id"]
        assert [job["stage"] for job in trace_payload["jobs"]] == ["retention_plan"]

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert "data/reports/retention_plan.jsonl: checked 1 records" in artifact_check.stdout


def test_retention_plan_uses_configured_retention_days(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        Path("pipeline.yaml").write_text(
            """
retention:
  raw_source_retention_days: 30
""",
            encoding="utf-8",
        )
        current_time = datetime.now(timezone.utc)
        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_expired",
                source_modality="chat",
                source_file="expired.json",
                ingested_at=current_time - timedelta(days=31),
            ),
        )
        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_recent",
                source_modality="chat",
                source_file="recent.json",
                ingested_at=current_time - timedelta(days=29),
            ),
        )

        result = runner.invoke(app, ["retention-plan", "--config", "pipeline.yaml"])

        assert result.exit_code == 0, result.stdout
        assert "candidates=1" in result.stdout
        candidates = [
            payload
            for _, payload in read_jsonl(Path("data/reports/retention_plan.jsonl"))
        ]
        assert [candidate["source_id"] for candidate in candidates] == ["src_expired"]
        assert candidates[0]["retention_days"] == 30
