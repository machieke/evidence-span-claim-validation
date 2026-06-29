import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord


runner = CliRunner()


def test_repair_claims_suggests_exact_evidence_text(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        evidence = EvidenceRecord(
            evidence_id="ev_1",
            source_id="src_1",
            source_modality="pdf",
            evidence_type="text_span",
            text="The vessel Hope appears old.",
            provenance={"page": 1, "block_id": "block_1"},
        )
        span = SpanRecord(
            span_id="span_1",
            source_id="src_1",
            source_modality="pdf",
            evidence_id="ev_1",
            text="The vessel Hope appears old.",
            label="claim_bearing",
            score=0.8,
        )
        claim = RawClaimRecord(
            claim_id="claim_1",
            source_id="src_1",
            source_modality="pdf",
            span_id="span_1",
            evidence_id="ev_1",
            source_faithful_claim="The document states that the vessel Hope appears old.",
            modality="uncertain_observation",
            evidence_text="The vessel   Hope appears old.",
            attribution={"type": "document", "agent": "src_1"},
            truth_status="source_asserted_unverified",
            confidence=0.8,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/spans.jsonl"), span)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        result = runner.invoke(app, ["repair-claims", "--only", "evidence_not_exact_substring"])

        assert result.exit_code == 0, result.stdout
        suggestions = [payload for _, payload in read_jsonl(Path("data/reports/claim_repairs.jsonl"))]
        assert len(suggestions) == 1
        assert suggestions[0]["claim_id"] == "claim_1"
        assert suggestions[0]["original_evidence_text"] == "The vessel   Hope appears old."
        assert suggestions[0]["suggested_evidence_text"] == "The vessel Hope appears old."
        assert suggestions[0]["support_scope"] == "span"

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "repair_claims"
        assert jobs[0]["model_id"] == "claim.repair_suggestion.v1"
        assert jobs[0]["input_record_ids"] == ["claims_raw", "evidence", "spans"]
        assert jobs[0]["metrics"] == {"suggestions": 1}

        invalid = runner.invoke(app, ["repair-claims", "--only", "unsupported_entities_introduced"])
        assert invalid.exit_code != 0
        assert "repair-claims supports reason codes" in invalid.stdout

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| repair_claims | 1 |" in report_text
        assert "| Evidence repair suggestion rate | 100.0% |" in report_text
        assert "| Evidence repair suggestions | 1 |" in report_text

        trace = runner.invoke(app, ["trace-claim", "claim_1"])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert trace_payload["repair_suggestions"][0]["repair_id"] == suggestions[0]["repair_id"]

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert "data/reports/claim_repairs.jsonl: checked 1 records" in artifact_check.stdout


def test_repair_claims_suggests_normalized_substring_slice(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        evidence = EvidenceRecord(
            evidence_id="ev_1",
            source_id="src_1",
            source_modality="pdf",
            evidence_type="text_span",
            text="The report says \u201cHope\u201d sailed at 09:00 on Monday.",
            provenance={"page": 1, "block_id": "block_1"},
        )
        claim = RawClaimRecord(
            claim_id="claim_1",
            source_id="src_1",
            source_modality="pdf",
            evidence_id="ev_1",
            source_faithful_claim='The document states that "Hope" sailed at 09:00.',
            modality="asserted",
            evidence_text='"Hope" sailed at 09:00',
            attribution={"type": "document", "agent": "src_1"},
            truth_status="source_asserted_unverified",
            confidence=0.8,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        result = runner.invoke(app, ["repair-claims"])

        assert result.exit_code == 0, result.stdout
        suggestions = [payload for _, payload in read_jsonl(Path("data/reports/claim_repairs.jsonl"))]
        assert len(suggestions) == 1
        assert suggestions[0]["original_evidence_text"] == '"Hope" sailed at 09:00'
        assert suggestions[0]["suggested_evidence_text"] == "\u201cHope\u201d sailed at 09:00"
        assert suggestions[0]["support_scope"] == "evidence"


def test_apply_repairs_creates_audited_repaired_raw_claim(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        evidence = EvidenceRecord(
            evidence_id="ev_1",
            source_id="src_1",
            source_modality="pdf",
            evidence_type="text_span",
            text="The vessel Hope appears old.",
            provenance={"page": 1, "block_id": "block_1"},
        )
        span = SpanRecord(
            span_id="span_1",
            source_id="src_1",
            source_modality="pdf",
            evidence_id="ev_1",
            text="The vessel Hope appears old.",
            label="claim_bearing",
            score=0.8,
        )
        claim = RawClaimRecord(
            claim_id="claim_1",
            source_id="src_1",
            source_modality="pdf",
            span_id="span_1",
            evidence_id="ev_1",
            source_faithful_claim="The document states that the vessel Hope appears old.",
            modality="uncertain_observation",
            evidence_text="The vessel   Hope appears old.",
            attribution={"type": "document", "agent": "src_1"},
            truth_status="source_asserted_unverified",
            confidence=0.8,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/spans.jsonl"), span)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        repair = runner.invoke(app, ["repair-claims"])
        first = runner.invoke(app, ["apply-repairs", "--actor-id", "repairer_1"])
        second = runner.invoke(app, ["apply-repairs", "--actor-id", "repairer_1"])

        assert repair.exit_code == 0, repair.stdout
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "repairs_applied=1 repairs_skipped=0 repairs_failed=0" in first.stdout
        assert "repairs_applied=0 repairs_skipped=1 repairs_failed=0" in second.stdout

        claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(claims) == 2
        repaired = next(record for record in claims if record["claim_id"] != "claim_1")
        assert repaired["evidence_text"] == "The vessel Hope appears old."
        assert repaired["attributes"]["repair"]["original_claim_id"] == "claim_1"
        assert repaired["attributes"]["repair"]["reason_codes"] == ["evidence_not_exact_substring"]
        assert "evidence_text_repaired" in repaired["risk_flags"]

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert len(validations) == 1
        assert validations[0]["stage"] == "apply_repairs"
        assert validations[0]["status"] == "repaired"
        assert validations[0]["claim_id"] == repaired["claim_id"]
        assert validations[0]["validator_version"] == "claim.repair_application.v1"

        audit_events = [payload for _, payload in read_jsonl(Path("data/jsonl/audit_events.jsonl"))]
        assert [event["status"] for event in audit_events] == ["created", "skipped"]
        assert all(event["action"] == "apply_repair" for event in audit_events)
        assert all(event["actor_id"] == "repairer_1" for event in audit_events)
        assert all(event["claim_id"] == repaired["claim_id"] for event in audit_events)

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert [job["stage"] for job in jobs] == ["repair_claims", "apply_repairs"]
        assert jobs[1]["model_id"] == "claim.repair_application.v1"
        assert jobs[1]["metrics"] == {"repairs_applied": 1, "repairs_failed": 0, "repairs_skipped": 0}
        assert repaired["claim_id"] in jobs[1]["input_record_ids"]

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| Repair application success rate | 100.0% |" in report_text

        trace = runner.invoke(app, ["trace-claim", repaired["claim_id"]])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert [job["stage"] for job in trace_payload["jobs"]] == ["apply_repairs"]
        assert [event["status"] for event in trace_payload["audit_events"]] == ["created", "skipped"]

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
