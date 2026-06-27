from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord
from evidence_pipeline.schemas.spans import SpanRecord


runner = CliRunner()


def _seed_claim_validation_fixture() -> None:
    evidence = EvidenceRecord(
        evidence_id="ev_msg_1",
        source_id="src_chat_1",
        source_modality="chat",
        evidence_type="message_span",
        text="I saw Hope yesterday. It had three masts.",
        provenance={
            "conversation_id": "conv_1",
            "message_id": "msg_1",
            "sender_id": "user_b",
            "sender_display_name": "Bob",
            "char_start": 0,
            "char_end": 41,
        },
    )
    span = SpanRecord(
        span_id="span_1",
        chunk_id="chunk_1",
        source_id="src_chat_1",
        source_modality="chat",
        evidence_id="ev_msg_1",
        text="It had three masts.",
        char_start=22,
        char_end=41,
        label="claim_bearing",
        score=0.8,
    )
    accepted_claim = RawClaimRecord(
        claim_id="claim_accepted",
        source_id="src_chat_1",
        source_modality="chat",
        span_id="span_1",
        evidence_id="ev_msg_1",
        source_faithful_claim="The speaker asserted that it had three masts.",
        subject="it",
        predicate="had",
        object="three masts",
        modality="asserted",
        evidence_text="It had three masts.",
        attribution={"type": "speaker", "agent": "user_b"},
        truth_status="speaker_asserted_unverified",
        confidence=0.82,
    )
    rejected_claim = RawClaimRecord(
        claim_id="claim_rejected",
        source_id="src_chat_1",
        source_modality="chat",
        span_id="span_1",
        evidence_id="ev_msg_1",
        source_faithful_claim="The speaker asserted that it had four masts.",
        subject="it",
        predicate="had",
        object="four masts",
        modality="asserted",
        evidence_text="It had four masts.",
        attribution={"type": "speaker", "agent": "user_b"},
        truth_status="speaker_asserted_unverified",
        confidence=0.82,
    )

    append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
    append_jsonl(Path("data/jsonl/spans.jsonl"), span)
    append_jsonl(Path("data/jsonl/claims.raw.jsonl"), accepted_claim)
    append_jsonl(Path("data/jsonl/claims.raw.jsonl"), rejected_claim)


def test_validate_claims_writes_accepted_validation_and_quarantine(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        _seed_claim_validation_fixture()

        first = runner.invoke(app, ["validate-claims"])
        second = runner.invoke(app, ["validate-claims"])

        assert first.exit_code == 0, first.stdout
        assert "claims_accepted=1" in first.stdout
        assert "claims_quarantined=1" in first.stdout
        assert second.exit_code == 0, second.stdout
        assert "claims_skipped=2" in second.stdout

        validated = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.validated.jsonl"))]
        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]

        assert [claim["claim_id"] for claim in validated] == ["claim_accepted"]
        assert len(validations) == 2
        assert [record["claim_id"] for record in quarantined] == ["claim_rejected"]
        assert quarantined[0]["reason_codes"] == ["evidence_not_exact_substring"]

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout


def test_validate_claims_quarantines_unsupported_named_entities(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        evidence = EvidenceRecord(
            evidence_id="ev_msg_1",
            source_id="src_chat_1",
            source_modality="chat",
            evidence_type="message_span",
            text="Hope had three masts.",
            provenance={
                "conversation_id": "conv_1",
                "message_id": "msg_1",
                "sender_id": "user_b",
                "sender_display_name": "Bob",
                "char_start": 0,
                "char_end": 21,
            },
        )
        span = SpanRecord(
            span_id="span_1",
            chunk_id="chunk_1",
            source_id="src_chat_1",
            source_modality="chat",
            evidence_id="ev_msg_1",
            text="Hope had three masts.",
            char_start=0,
            char_end=21,
            label="claim_bearing",
            score=0.8,
        )
        claim = RawClaimRecord(
            claim_id="claim_introduced_entity",
            source_id="src_chat_1",
            source_modality="chat",
            span_id="span_1",
            evidence_id="ev_msg_1",
            source_faithful_claim="The speaker asserted that Hope had three masts near Boston.",
            subject="Hope",
            predicate="had",
            object="three masts near Boston",
            modality="asserted",
            evidence_text="Hope had three masts.",
            attribution={"type": "speaker", "agent": "user_b"},
            truth_status="speaker_asserted_unverified",
            confidence=0.9,
        )

        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/spans.jsonl"), span)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=0" in result.stdout
        assert "claims_quarantined=1" in result.stdout

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]

        assert validations[0]["errors"] == ["unsupported_entities_introduced"]
        assert validations[0]["warnings"] == ["unsupported_entities_introduced"]
        assert validations[0]["metadata"]["validation"]["introduced_entities"] == ["Boston"]
        assert quarantined[0]["reason_codes"] == ["unsupported_entities_introduced"]
        assert quarantined[0]["warnings"] == ["unsupported_entities_introduced"]

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "## Validation Errors" in report_text
        assert "## Validation Warnings" in report_text
        assert "| unsupported_entities_introduced | 1 |" in report_text
        assert "| Unsupported entity validation rate | 100.0% |" in report_text
