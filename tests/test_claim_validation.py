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


def test_validate_claims_quarantines_missing_chat_provenance(tmp_path: Path):
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
                provenance={},
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_missing_chat_provenance",
                source_id="src_chat_1",
                source_modality="chat",
                evidence_id="ev_msg_1",
                source_faithful_claim="The speaker asserted that Hope had three masts.",
                subject="Hope",
                predicate="had",
                object="three masts",
                modality="asserted",
                evidence_text="Hope had three masts.",
                attribution={"type": "speaker", "agent": "user_b"},
                truth_status="speaker_asserted_unverified",
                confidence=0.82,
            ),
        )

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=0" in result.stdout
        assert "claims_quarantined=1" in result.stdout

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert validations[0]["errors"] == [
            "missing_conversation_provenance",
            "missing_message_provenance",
            "missing_sender_provenance",
        ]
        assert validations[0]["validator_version"] == "deterministic.v5"

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert quarantined[0]["reason_codes"] == [
            "missing_conversation_provenance",
            "missing_message_provenance",
            "missing_sender_provenance",
        ]


def test_validate_claims_quarantines_missing_audio_provenance(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_audio_1",
                source_id="src_audio_1",
                source_modality="audio",
                evidence_type="utterance_span",
                text="Hope departed at 09:00.",
                provenance={},
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_missing_audio_provenance",
                source_id="src_audio_1",
                source_modality="audio",
                evidence_id="ev_audio_1",
                source_faithful_claim="The speaker asserted that Hope departed at 09:00.",
                subject="Hope",
                predicate="departed_at",
                object="09:00",
                modality="asserted",
                evidence_text="Hope departed at 09:00.",
                attribution={"type": "speaker", "agent": "SPEAKER_00"},
                truth_status="speaker_asserted_unverified",
                confidence=0.82,
            ),
        )

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=0" in result.stdout
        assert "claims_quarantined=1" in result.stdout

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert validations[0]["errors"] == [
            "missing_utterance_provenance",
            "missing_speaker_provenance",
            "missing_audio_timestamp_provenance",
        ]
        assert validations[0]["warnings"] == [
            "missing_asr_confidence_provenance",
            "missing_diarization_confidence_provenance",
        ]
        assert validations[0]["validator_version"] == "deterministic.v5"

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert quarantined[0]["reason_codes"] == [
            "missing_utterance_provenance",
            "missing_speaker_provenance",
            "missing_audio_timestamp_provenance",
        ]
        assert quarantined[0]["warnings"] == [
            "missing_asr_confidence_provenance",
            "missing_diarization_confidence_provenance",
        ]


def test_validate_claims_quarantines_missing_visual_region_provenance(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_img_1",
                source_id="src_img_1",
                source_modality="image",
                evidence_type="visual_region",
                text=None,
                provenance={},
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_missing_visual_region_provenance",
                source_id="src_img_1",
                source_modality="image",
                evidence_id="ev_img_1",
                claim_type="visual_region_proposal",
                source_faithful_claim="Region region_1 was proposed as a visual region by grid_16_stride16.",
                subject="region_1",
                predicate="proposed_visual_region",
                object={"bbox": [0, 0, 16, 16]},
                modality="model_observation",
                attribution={"type": "model", "agent": "grid_16_stride16"},
                truth_status="model_observation_unverified",
                confidence=0.5,
            ),
        )

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=0" in result.stdout
        assert "claims_quarantined=1" in result.stdout

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert validations[0]["errors"] == [
            "missing_image_provenance",
            "missing_region_provenance",
            "missing_image_bbox_provenance",
            "missing_region_proposal_provenance",
        ]
        assert validations[0]["warnings"] == [
            "missing_region_crop_provenance",
            "unsupported_entities_introduced",
        ]
        assert validations[0]["validator_version"] == "deterministic.v5"

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert quarantined[0]["reason_codes"] == validations[0]["errors"]
        assert quarantined[0]["warnings"] == [
            "missing_region_crop_provenance",
            "unsupported_entities_introduced",
        ]


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


def test_validate_claims_quarantines_quantity_word_mismatch(tmp_path: Path):
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
            claim_id="claim_quantity_mismatch",
            source_id="src_chat_1",
            source_modality="chat",
            span_id="span_1",
            evidence_id="ev_msg_1",
            source_faithful_claim="The speaker asserted that Hope had four masts.",
            subject="Hope",
            predicate="had",
            object="four masts",
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

        assert validations[0]["errors"] == ["quantity_mismatch"]
        assert validations[0]["metadata"]["validation"]["quantities_preserved"] is False
        assert quarantined[0]["reason_codes"] == ["quantity_mismatch"]

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| Quantity preservation rate | 0.0% |" in report_text
        assert "| Attribution preservation rate | 100.0% |" in report_text
        assert "| Negation preservation rate | 100.0% |" in report_text
        assert "| Uncertainty preservation rate | 100.0% |" in report_text


def test_validate_claims_enforces_pdf_provenance_requirements(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_pdf_bad",
                source_id="src_pdf_bad",
                source_modality="pdf",
                evidence_type="text_span",
                text="The vessel Hope appears old.",
                provenance={"extractor": "pymupdf"},
            ),
        )
        append_jsonl(
            Path("data/jsonl/evidence.jsonl"),
            EvidenceRecord(
                evidence_id="ev_pdf_good",
                source_id="src_pdf_good",
                source_modality="pdf",
                evidence_type="text_span",
                text="The surveyor found no active fuel leak.",
                provenance={"page": 2, "block_id": "pdf_block_2", "extractor": "pypdf"},
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_pdf_bad",
                source_id="src_pdf_bad",
                source_modality="pdf",
                evidence_id="ev_pdf_bad",
                source_faithful_claim="The document states: The vessel Hope appears old.",
                subject="vessel Hope",
                predicate="appears",
                object="old",
                modality="uncertain_observation",
                evidence_text="The vessel Hope appears old.",
                attribution={"type": "document", "agent": "src_pdf_bad"},
                truth_status="source_asserted_unverified",
                confidence=0.9,
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.raw.jsonl"),
            RawClaimRecord(
                claim_id="claim_pdf_good",
                source_id="src_pdf_good",
                source_modality="pdf",
                evidence_id="ev_pdf_good",
                source_faithful_claim="The document states: The surveyor found no active fuel leak.",
                subject="surveyor",
                predicate="found",
                object="no active fuel leak",
                modality="negated",
                evidence_text="The surveyor found no active fuel leak.",
                attribution={"type": "document", "agent": "src_pdf_good"},
                truth_status="source_asserted_unverified",
                confidence=0.9,
            ),
        )

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=1" in result.stdout
        assert "claims_quarantined=1" in result.stdout

        validations = {
            payload["claim_id"]: payload
            for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))
        }
        assert validations["claim_pdf_bad"]["errors"] == [
            "missing_page_provenance",
            "missing_block_provenance",
            "missing_bbox_provenance",
        ]
        assert validations["claim_pdf_bad"]["validator_version"] == "deterministic.v5"
        assert validations["claim_pdf_good"]["status"] == "accepted_extracted"

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert quarantined[0]["reason_codes"] == [
            "missing_page_provenance",
            "missing_block_provenance",
            "missing_bbox_provenance",
        ]
