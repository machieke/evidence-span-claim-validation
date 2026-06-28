import json
import sqlite3
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.audio import AudioUtteranceRecord
from evidence_pipeline.schemas.chat import ChatMessageRecord
from evidence_pipeline.schemas.claims import RawClaimRecord


runner = CliRunner()


def test_detect_pii_writes_redacted_findings_without_raw_matches(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/chat_messages.jsonl"),
            ChatMessageRecord(
                message_id="msg_1",
                source_id="src_chat_1",
                conversation_id="conv_1",
                sender_id="alice",
                text="Email alice@example.com or call 415-555-1212.",
            ),
        )

        first = runner.invoke(app, ["detect-pii", "--artifact", "chat_messages"])
        second = runner.invoke(app, ["detect-pii", "--artifact", "chat_messages"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "findings=2" in first.stdout

        output_path = Path("data/reports/pii_findings.jsonl")
        output_text = output_path.read_text(encoding="utf-8")
        findings = [payload for _, payload in read_jsonl(output_path)]
        assert len(findings) == 2
        assert {finding["pii_type"] for finding in findings} == {"email", "phone"}
        assert all(finding["artifact"] == "chat_messages" for finding in findings)
        assert all("match_hash" in finding for finding in findings)
        assert "alice@example.com" not in output_text
        assert "415-555-1212" not in output_text
        assert "a***@example.com" in output_text
        assert "***-***-1212" in output_text

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "detect_pii"
        assert jobs[0]["model_id"] == "pii.regex.v1"
        assert jobs[0]["input_record_ids"] == ["artifact:chat_messages"]
        assert jobs[0]["metrics"] == {"findings": 2}

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| detect_pii | 1 |" in report_text
        assert "| pii_findings | 2 |" in report_text
        assert "## PII Findings By Type" in report_text
        assert "| email | 1 |" in report_text
        assert "| phone | 1 |" in report_text

        export = runner.invoke(app, ["export-sqlite"])
        assert export.exit_code == 0, export.stdout
        with sqlite3.connect(Path("data/reports/pipeline.sqlite")) as connection:
            finding_count = connection.execute("SELECT COUNT(*) FROM pii_findings").fetchone()[0]
            artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("pii_findings",),
            ).fetchone()[0]
            payload_json = connection.execute(
                "SELECT payload_json FROM pii_findings WHERE record_key = ?",
                (findings[0]["finding_id"],),
            ).fetchone()[0]

        exported_payload = json.loads(payload_json)
        assert finding_count == 2
        assert artifact_count == 2
        assert exported_payload["pii_type"] in {"email", "phone"}
        assert "alice@example.com" not in payload_json
        assert "415-555-1212" not in payload_json

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert "data/reports/pii_findings.jsonl: checked 2 records" in artifact_check.stdout

        invalid = runner.invoke(app, ["detect-pii", "--artifact", "images"])
        assert invalid.exit_code != 0
        assert "PII detection supports artifacts" in invalid.stdout


def test_redact_pii_writes_redacted_copy_without_mutating_source(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        source_path = Path("data/jsonl/chat_messages.jsonl")
        append_jsonl(
            source_path,
            ChatMessageRecord(
                message_id="msg_1",
                source_id="src_chat_1",
                conversation_id="conv_1",
                sender_id="alice",
                text="Email alice@example.com or call 415-555-1212.",
            ),
        )

        first = runner.invoke(app, ["redact-pii", "--artifact", "chat_messages"])
        second = runner.invoke(app, ["redact-pii", "--artifact", "chat_messages"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "records=1" in first.stdout
        assert "replacements=2" in first.stdout
        assert "redactions=1" in first.stdout
        assert "manifest=data/reports/pii_redactions.jsonl" in first.stdout

        output_path = Path("data/reports/chat_messages.redacted.jsonl")
        output_text = output_path.read_text(encoding="utf-8")
        redacted_records = [payload for _, payload in read_jsonl(output_path)]
        assert len(redacted_records) == 1
        assert redacted_records[0]["text"] == "Email [EMAIL] or call [PHONE]."
        assert "alice@example.com" not in output_text
        assert "415-555-1212" not in output_text

        manifest_path = Path("data/reports/pii_redactions.jsonl")
        manifest_text = manifest_path.read_text(encoding="utf-8")
        redactions = [payload for _, payload in read_jsonl(manifest_path)]
        assert len(redactions) == 1
        assert redactions[0]["artifact"] == "chat_messages"
        assert redactions[0]["record_id"] == "msg_1"
        assert redactions[0]["fields"] == ["text"]
        assert redactions[0]["replacement_count"] == 2
        assert redactions[0]["output_path"] == "data/reports/chat_messages.redacted.jsonl"
        assert "alice@example.com" not in manifest_text
        assert "415-555-1212" not in manifest_text

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "redact_pii"
        assert jobs[0]["model_id"] == "pii.regex.v1"
        assert jobs[0]["input_record_ids"] == ["artifact:chat_messages"]
        assert jobs[0]["metrics"] == {
            "records_written": 1,
            "redactions": 1,
            "replacements": 2,
        }

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| redact_pii | 1 |" in report_text
        assert "| pii_redactions | 1 |" in report_text
        assert "## PII Redactions By Artifact" in report_text
        assert "| chat_messages | 1 |" in report_text

        export = runner.invoke(app, ["export-sqlite"])
        assert export.exit_code == 0, export.stdout
        with sqlite3.connect(Path("data/reports/pipeline.sqlite")) as connection:
            redaction_count = connection.execute("SELECT COUNT(*) FROM pii_redactions").fetchone()[0]
            artifact_count = connection.execute(
                "SELECT record_count FROM artifact_counts WHERE artifact_name = ?",
                ("pii_redactions",),
            ).fetchone()[0]
            payload_json = connection.execute(
                "SELECT payload_json FROM pii_redactions WHERE record_key = ?",
                (redactions[0]["redaction_id"],),
            ).fetchone()[0]

        exported_payload = json.loads(payload_json)
        assert redaction_count == 1
        assert artifact_count == 1
        assert exported_payload["replacement_count"] == 2
        assert "alice@example.com" not in payload_json
        assert "415-555-1212" not in payload_json

        source_text = source_path.read_text(encoding="utf-8")
        assert "alice@example.com" in source_text
        assert "415-555-1212" in source_text

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert "data/reports/pii_redactions.jsonl: checked 1 records" in artifact_check.stdout

        invalid = runner.invoke(app, ["redact-pii", "--artifact", "all"])
        assert invalid.exit_code != 0
        assert "requires one artifact at a time" in invalid.stdout


def test_audio_transcript_pii_detection_and_redaction(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        source_path = Path("data/jsonl/audio_utterances.jsonl")
        append_jsonl(
            source_path,
            AudioUtteranceRecord(
                utterance_id="utt_1",
                source_id="src_audio_1",
                speaker="SPEAKER_00",
                start=0.0,
                end=4.0,
                text="Reach me at alice@example.com, 415-555-1212, or 123-45-6789.",
                asr_confidence=0.96,
                diarization_confidence=0.9,
            ),
        )

        detection = runner.invoke(app, ["detect-pii", "--artifact", "audio_utterances"])
        assert detection.exit_code == 0, detection.stdout
        assert "findings=3" in detection.stdout
        findings_path = Path("data/reports/pii_findings.jsonl")
        findings_text = findings_path.read_text(encoding="utf-8")
        findings = [payload for _, payload in read_jsonl(findings_path)]
        assert {finding["pii_type"] for finding in findings} == {"email", "phone", "ssn"}
        assert all(finding["artifact"] == "audio_utterances" for finding in findings)
        assert all(finding["record_id"] == "utt_1" for finding in findings)
        assert "alice@example.com" not in findings_text
        assert "415-555-1212" not in findings_text
        assert "123-45-6789" not in findings_text

        redaction = runner.invoke(app, ["redact-pii", "--artifact", "audio_utterances"])
        assert redaction.exit_code == 0, redaction.stdout
        assert "records=1" in redaction.stdout
        assert "replacements=3" in redaction.stdout
        redacted_path = Path("data/reports/audio_utterances.redacted.jsonl")
        redacted_text = redacted_path.read_text(encoding="utf-8")
        redacted_records = [payload for _, payload in read_jsonl(redacted_path)]
        assert redacted_records[0]["text"] == "Reach me at [EMAIL], [PHONE], or [SSN]."
        assert "alice@example.com" not in redacted_text
        assert "415-555-1212" not in redacted_text
        assert "123-45-6789" not in redacted_text

        source_text = source_path.read_text(encoding="utf-8")
        assert "alice@example.com" in source_text
        assert "415-555-1212" in source_text
        assert "123-45-6789" in source_text


def test_trace_claim_includes_claim_pii_reports(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_pii",
                source_id="src_chat_1",
                source_modality="chat",
                evidence_id="ev_claim_pii",
                source_faithful_claim="The speaker asserted: Call Alice at 415-555-1212.",
                modality="asserted",
                evidence_text="Call Alice at 415-555-1212.",
                attribution={"type": "speaker", "agent": "alice"},
                truth_status="speaker_asserted_unverified",
                confidence=0.9,
            ),
        )

        detection = runner.invoke(app, ["detect-pii", "--artifact", "claims_raw"])
        redaction = runner.invoke(app, ["redact-pii", "--artifact", "claims_raw"])
        assert detection.exit_code == 0, detection.stdout
        assert redaction.exit_code == 0, redaction.stdout

        findings = [
            payload
            for _, payload in read_jsonl(Path("data/reports/pii_findings.jsonl"))
        ]
        redactions = [
            payload
            for _, payload in read_jsonl(Path("data/reports/pii_redactions.jsonl"))
        ]
        trace = runner.invoke(app, ["trace-claim", "claim_pii"])

        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert [finding["finding_id"] for finding in trace_payload["pii_findings"]] == [
            finding["finding_id"] for finding in findings
        ]
        assert trace_payload["pii_redactions"][0]["redaction_id"] == redactions[0]["redaction_id"]
