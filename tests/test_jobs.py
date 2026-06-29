from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.cli import app
from evidence_pipeline.jobs import record_job_result
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord


runner = CliRunner()


def test_register_source_writes_idempotent_job_record(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        Path("source.txt").write_text("Hope had three masts.", encoding="utf-8")

        first = runner.invoke(app, ["register-source", "source.txt", "--modality", "chat"])
        second = runner.invoke(app, ["register-source", "source.txt", "--modality", "chat"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert first.stdout == second.stdout
        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "register_source"
        assert jobs[0]["model_id"] == "source.registration.v1"
        assert jobs[0]["input_record_ids"] == [first.stdout.strip()]
        assert jobs[0]["metrics"] == {"sources_created": 1, "sources_skipped": 0}


def test_core_stage_commands_write_idempotent_job_records(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_msg_1",
                source_id="src_chat_1",
                source_modality="chat",
                evidence_type="message_span",
                text="Hope had three masts.",
                provenance={"conversation_id": "conv_1", "message_id": "msg_1", "sender_id": "alice"},
            ),
        )
        append_jsonl(
            Path("data/jsonl/spans.jsonl"),
            SpanRecord(
                span_id="span_1",
                source_id="src_chat_1",
                source_modality="chat",
                evidence_id="ev_msg_1",
                text="Hope had three masts.",
                char_start=0,
                char_end=21,
                label="claim_bearing",
                score=0.9,
            ),
        )

        commands = [
            ["extract-claims", "--modality", "chat"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs] == ["extract_claims", "validate_claims", "normalize_claims"]
        assert all(job["status"] == "succeeded" for job in jobs)
        assert all(job["attempts"] == 1 for job in jobs)
        assert all(job["config_hash"].startswith("cfg_") for job in jobs)
        assert [job["model_id"] for job in jobs] == [
            "rules.v1",
            "deterministic.v5",
            "normalizer.v1",
        ]
        assert all(job["model_hash"].startswith("model_") for job in jobs)
        assert all(job.get("prompt_id") is None for job in jobs)
        assert all(job.get("prompt_hash") is None for job in jobs)
        assert jobs[0]["input_record_ids"] == ["modality:chat"]
        assert jobs[0]["metrics"] == {"claims_created": 1, "claims_skipped": 0}
        assert jobs[1]["metrics"] == {
            "claims_accepted": 1,
            "claims_quarantined": 0,
            "claims_skipped": 0,
        }
        assert jobs[2]["metrics"] == {"claims_normalized": 1, "claims_skipped": 0}

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 3 |" in report_text
        assert "| extract_claims | 1 |" in report_text
        assert "| validate_claims | 1 |" in report_text
        assert "| normalize_claims | 1 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout


def test_extract_claims_can_append_in_batches(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        for index, text in enumerate(
            [
                "Hope had three masts.",
                "Hope had a diesel engine.",
                "Hope was docked yesterday.",
            ],
            start=1,
        ):
            append_jsonl(
                Path("data/jsonl/evidence.jsonl"),
                EvidenceRecord(
                    evidence_id=f"ev_msg_{index}",
                    source_id="src_chat_1",
                    source_modality="chat",
                    evidence_type="message_span",
                    text=text,
                    provenance={"message_id": f"msg_{index}", "sender_id": "alice"},
                ),
            )
            append_jsonl(
                Path("data/jsonl/spans.jsonl"),
                SpanRecord(
                    span_id=f"span_{index}",
                    source_id="src_chat_1",
                    source_modality="chat",
                    evidence_id=f"ev_msg_{index}",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    label="claim_bearing",
                    score=0.9,
                ),
            )

        first = runner.invoke(
            app,
            ["extract-claims", "--modality", "chat", "--batch-size", "2"],
        )
        second = runner.invoke(
            app,
            ["extract-claims", "--modality", "chat", "--batch-size", "2"],
        )
        invalid = runner.invoke(
            app,
            ["extract-claims", "--modality", "chat", "--batch-size", "0"],
        )

        assert first.exit_code == 0, first.stdout
        assert "claims_created=3" in first.stdout
        assert "claims_skipped=0" in first.stdout
        assert "batches_processed=2" in first.stdout
        assert second.exit_code == 0, second.stdout
        assert "claims_created=0" in second.stdout
        assert "claims_skipped=3" in second.stdout
        assert "batches_processed=0" in second.stdout
        assert invalid.exit_code != 0
        assert "batch size must be at least 1" in invalid.stdout

        claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(claims) == 3
        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["metrics"] == {
            "batches_processed": 2,
            "claims_created": 3,
            "claims_skipped": 0,
        }
        assert jobs[0]["metadata"] == {"batch_size": 2, "modality": "chat"}


def test_record_job_result_persists_auditable_model_and_prompt_ids(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        config = PipelineConfig()

        first = record_job_result(
            config,
            stage="extract_claims",
            source_id="src_1",
            input_record_ids=["span_1"],
            model_id="llm.extractor.v2",
            prompt_id="extract_claims.chat.v3",
            metrics={"claims_created": 1},
        )
        second = record_job_result(
            config,
            stage="extract_claims",
            source_id="src_1",
            input_record_ids=["span_1"],
            model_id="llm.extractor.v2",
            prompt_id="extract_claims.chat.v3",
            metrics={"claims_created": 1},
        )

        assert first.created is True
        assert second.created is False
        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["job_id"] == first.job_id
        assert jobs[0]["model_id"] == "llm.extractor.v2"
        assert jobs[0]["prompt_id"] == "extract_claims.chat.v3"
        assert jobs[0]["model_hash"].startswith("model_")
        assert jobs[0]["prompt_hash"].startswith("prompt_")
