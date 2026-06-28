from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import ClaimValidationSummary, RawClaimRecord, ValidatedClaimRecord


runner = CliRunner()


def _validated_from_raw(claim: RawClaimRecord) -> ValidatedClaimRecord:
    return ValidatedClaimRecord(
        claim_id=claim.claim_id,
        source_id=claim.source_id,
        source_modality=claim.source_modality,
        span_id=claim.span_id,
        evidence_id=claim.evidence_id,
        source_faithful_claim=claim.source_faithful_claim,
        evidence_text=claim.evidence_text,
        modality=claim.modality,
        truth_status=claim.truth_status,
        support_status="accepted_extracted",
        validation=ClaimValidationSummary(deterministic_valid=True),
        risk_flags=claim.risk_flags,
    )


def test_normalize_claims_uses_image_classification_predicate_registry(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        claim = RawClaimRecord(
            claim_id="claim_img_1",
            source_id="src_img_1",
            source_modality="image",
            evidence_id="ev_img_1",
            claim_type="named_visual_classification",
            source_faithful_claim="Model dominant_color_classifier_v1 classified region region_1 as red.",
            subject="region_1",
            predicate="classified_as",
            object="red",
            modality="model_observation",
            attribution={"type": "model", "agent": "dominant_color_classifier_v1"},
            truth_status="model_observation_unverified",
            confidence=0.93,
        )
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)
        append_jsonl(Path("data/jsonl/claims.validated.jsonl"), _validated_from_raw(claim))

        result = runner.invoke(app, ["normalize-claims"])

        assert result.exit_code == 0, result.stdout
        normalized = next(payload for _, payload in read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))
        assert normalized["normalized_claim"]["subject"] == "image_region:region_1"
        assert normalized["normalized_claim"]["predicate"] == "classified_as"
        assert normalized["normalized_claim"]["object"] == "red"
        assert normalized["normalization"]["predicate_mapping"] == {
            "surface": "classified_as",
            "canonical": "classified_as",
        }
        assert normalized["normalization"]["metadata"]["predicate_registry_version"] == "predicate.registry.v1"
        assert {item["basis"] for item in normalized["normalization"]["entity_resolution"]} == {
            "attribution_agent",
            "claim_subject",
        }


def test_normalize_claims_preserves_controlled_raw_predicates(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        claim = RawClaimRecord(
            claim_id="claim_pdf_1",
            source_id="src_pdf_1",
            source_modality="pdf",
            evidence_id="ev_pdf_1",
            source_faithful_claim="The survey report observes that vessel Hope appears old.",
            subject="vessel Hope",
            predicate="reports_observation",
            object="appears old",
            modality="asserted",
            evidence_text="vessel Hope appears old",
            attribution={"type": "document", "agent": "src_pdf_1"},
            truth_status="source_asserted_unverified",
            confidence=0.9,
        )
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)
        append_jsonl(Path("data/jsonl/claims.validated.jsonl"), _validated_from_raw(claim))

        result = runner.invoke(app, ["normalize-claims"])

        assert result.exit_code == 0, result.stdout
        normalized = next(payload for _, payload in read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))
        assert normalized["normalized_claim"]["subject"] == "entity:vessel_hope"
        assert normalized["normalized_claim"]["predicate"] == "reports_observation"
        assert normalized["normalization"]["predicate_mapping"] == {
            "surface": "reports_observation",
            "canonical": "reports_observation",
        }
        assert [item["basis"] for item in normalized["normalization"]["entity_resolution"]] == [
            "claim_subject",
            "attribution_agent",
        ]
