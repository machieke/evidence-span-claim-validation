import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.config import PipelineConfig, load_config
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.sources import SourceRecord
from evidence_pipeline.validation.privacy import (
    PrivacyPolicyError,
    model_invocation_privacy_decision,
    require_model_invocation_allowed,
)


runner = CliRunner()


def _claim(claim_id: str, source_id: str, provider: str, text: str) -> RawClaimRecord:
    return RawClaimRecord(
        claim_id=claim_id,
        source_id=source_id,
        source_modality="chat",
        evidence_id=f"ev_{claim_id}",
        source_faithful_claim=f"The speaker asserted: {text}",
        modality="asserted",
        evidence_text=text,
        attribution={"type": "speaker", "agent": "alice"},
        truth_status="speaker_asserted_unverified",
        confidence=0.9,
        model={"provider": provider, "model": f"{provider}_model"},
    )


def test_check_privacy_flags_external_provider_for_sensitive_source(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_sensitive",
                source_modality="chat",
                source_file="sensitive.json",
                metadata={"local_only": True, "contains_pii": "yes"},
            ),
        )
        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_public",
                source_modality="chat",
                source_file="public.json",
                metadata={},
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            _claim("claim_local", "src_sensitive", "deterministic", "Alice called Bob."),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            _claim("claim_external", "src_sensitive", "openai", "Call Alice at 415-555-1212."),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            _claim("claim_public_external", "src_public", "openai", "Public launch is Tuesday."),
        )

        failed = runner.invoke(app, ["check-privacy"])
        assert failed.exit_code == 1, failed.stdout
        assert "claims_checked=3" in failed.stdout
        assert "violations=1" in failed.stdout

        report_only = runner.invoke(app, ["check-privacy", "--report-only"])
        assert report_only.exit_code == 0, report_only.stdout

        output_path = Path("data/reports/privacy_policy_violations.jsonl")
        output_text = output_path.read_text(encoding="utf-8")
        violations = [payload for _, payload in read_jsonl(output_path)]
        assert len(violations) == 1
        assert violations[0]["source_id"] == "src_sensitive"
        assert violations[0]["claim_id"] == "claim_external"
        assert violations[0]["provider"] == "openai"
        assert violations[0]["reason_code"] == "non_local_provider_for_sensitive_source"
        assert violations[0]["sensitive_metadata_keys"] == ["contains_pii", "local_only"]
        assert "Call Alice at 415-555-1212." not in output_text

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "check_privacy"
        assert jobs[0]["model_id"] == "privacy.local_only.v1"
        assert jobs[0]["input_record_ids"] == ["claims_raw", "sources"]
        assert jobs[0]["metrics"] == {"claims_checked": 3, "violations": 1}

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| check_privacy | 1 |" in report_text
        assert "| privacy_policy_violations | 1 |" in report_text
        assert "## Privacy Policy Violations" in report_text
        assert "| non_local_provider_for_sensitive_source | 1 |" in report_text

        trace = runner.invoke(app, ["trace-claim", "claim_external"])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert trace_payload["privacy_policy_violations"][0]["violation_id"] == violations[0]["violation_id"]

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert (
            "data/reports/privacy_policy_violations.jsonl: checked 1 records"
            in artifact_check.stdout
        )


def test_model_invocation_privacy_guard_blocks_external_provider_for_sensitive_source(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_sensitive",
                source_modality="chat",
                source_file="sensitive.json",
                metadata={"local_only": True, "contains_pii": "yes"},
            ),
        )

        local_decision = require_model_invocation_allowed(
            PipelineConfig(),
            "src_sensitive",
            "deterministic",
            "rules.v1",
        )
        assert local_decision.allowed is True

        blocked = model_invocation_privacy_decision(
            PipelineConfig(),
            "src_sensitive",
            "openai",
            "external_model",
        )
        assert blocked.allowed is False
        assert blocked.reason_code == "non_local_provider_for_sensitive_source"
        assert blocked.sensitive_metadata_keys == ["contains_pii", "local_only"]

        with pytest.raises(PrivacyPolicyError, match="model invocation blocked"):
            require_model_invocation_allowed(
                PipelineConfig(),
                "src_sensitive",
                "openai",
                "external_model",
            )


def test_check_privacy_uses_configured_local_providers(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        Path("pipeline.yaml").write_text(
            """
privacy:
  local_model_providers:
    - deterministic
    - openai
""",
            encoding="utf-8",
        )
        append_jsonl(
            Path("data/jsonl/sources.jsonl"),
            SourceRecord(
                source_id="src_sensitive",
                source_modality="chat",
                source_file="sensitive.json",
                metadata={"local_only": True},
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            _claim("claim_openai", "src_sensitive", "openai", "Alice called Bob."),
        )

        result = runner.invoke(app, ["check-privacy", "--config", "pipeline.yaml"])

        assert result.exit_code == 0, result.stdout
        assert "claims_checked=1" in result.stdout
        assert "violations=0" in result.stdout
        violations = [
            payload
            for _, payload in read_jsonl(Path("data/reports/privacy_policy_violations.jsonl"))
        ]
        assert violations == []

        decision = require_model_invocation_allowed(
            load_config(Path("pipeline.yaml")),
            "src_sensitive",
            "openai",
            "openai_model",
        )
        assert decision.allowed is True
