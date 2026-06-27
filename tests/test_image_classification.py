from pathlib import Path

from PIL import Image
from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.evidence import EvidenceRecord


runner = CliRunner()


def test_image_region_color_classification_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (16, 16), color=(220, 20, 20)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["embed-image-regions"],
            ["classify-image-regions"],
            ["validate-claims"],
            ["normalize-claims"],
            ["export-graph"],
        ]
        for command in commands:
            first = runner.invoke(app, command)
            second = runner.invoke(app, command)
            assert first.exit_code == 0, first.stdout
            assert second.exit_code == 0, second.stdout

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 1
        claim = raw_claims[0]
        assert claim["claim_type"] == "named_visual_classification"
        assert claim["object"] == "red"
        assert claim["predicate"] == "classified_as"
        assert claim["modality"] == "model_observation"
        assert claim["truth_status"] == "model_observation_unverified"
        assert claim["attribution"] == {"agent": "dominant_color_classifier_v1", "type": "model"}
        assert "color_only_classification" in claim["risk_flags"]

        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 1
        assert len(list(read_jsonl(Path("data/reports/claim_graph.jsonl")))) == 1

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout


def test_low_confidence_image_region_classification_is_quarantined(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (16, 16), color=(128, 128, 128)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["embed-image-regions"],
            ["classify-image-regions"],
            ["validate-claims"],
            ["normalize-claims"],
        ]
        for command in commands:
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.stdout

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert raw_claims[0]["claim_type"] == "named_visual_classification"
        assert raw_claims[0]["object"] == "gray"
        assert raw_claims[0]["attributes"]["classifier"]["confidence"] < 0.85

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]

        assert validations[0]["status"] == "quarantined"
        assert validations[0]["errors"] == ["image_label_low_confidence"]
        assert quarantined[0]["reason_codes"] == ["image_label_low_confidence"]
        assert len(list(read_jsonl(Path("data/jsonl/claims.validated.jsonl")))) == 0
        assert len(list(read_jsonl(Path("data/jsonl/claims.normalized.jsonl")))) == 0


def test_review_accepts_low_confidence_image_region_classification(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (16, 16), color=(128, 128, 128)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["embed-image-regions"],
            ["classify-image-regions"],
        ]
        for command in commands:
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.stdout

        claim = next(payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl")))
        review = runner.invoke(
            app,
            [
                "review-claim",
                claim["claim_id"],
                "--decision",
                "accept",
                "--reviewer-id",
                "reviewer_1",
                "--reason-code",
                "human_confirmed_label",
            ],
        )
        validation = runner.invoke(app, ["validate-claims"])

        assert review.exit_code == 0, review.stdout
        assert validation.exit_code == 0, validation.stdout
        assert "claims_accepted=1" in validation.stdout
        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert validations[0]["status"] == "accepted_extracted"
        assert validations[0]["errors"] == []


def test_review_rejects_high_confidence_image_region_classification(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        image_path = Path("image.png")
        Image.new("RGB", (16, 16), color=(220, 20, 20)).save(image_path)

        commands = [
            ["ingest-images", "image.png"],
            ["propose-image-regions", "--patch-size", "16", "--stride", "16"],
            ["build-image-evidence"],
            ["embed-image-regions"],
            ["classify-image-regions"],
        ]
        for command in commands:
            result = runner.invoke(app, command)
            assert result.exit_code == 0, result.stdout

        claim = next(payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl")))
        review = runner.invoke(
            app,
            [
                "review-claim",
                claim["claim_id"],
                "--decision",
                "reject",
                "--reviewer-id",
                "reviewer_1",
                "--reason-code",
                "wrong_label",
            ],
        )
        validation = runner.invoke(app, ["validate-claims"])

        assert review.exit_code == 0, review.stdout
        assert validation.exit_code == 0, validation.stdout
        assert "claims_quarantined=1" in validation.stdout
        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert validations[0]["errors"] == ["human_review_rejected_label"]
        assert quarantined[0]["reason_codes"] == ["human_review_rejected_label"]


def test_human_confirmed_image_classification_bypasses_confidence_gate(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        evidence = EvidenceRecord(
            evidence_id="ev_img_1",
            source_id="src_img_1",
            source_modality="image",
            evidence_type="visual_region",
            text=None,
            provenance={"region_id": "region_1", "bbox": [0, 0, 16, 16]},
        )
        claim = RawClaimRecord(
            claim_id="claim_img_label_human_confirmed",
            source_id="src_img_1",
            source_modality="image",
            evidence_id="ev_img_1",
            claim_type="named_visual_classification",
            source_faithful_claim="Reviewer reviewer_1 confirmed region region_1 as gray.",
            subject="region_1",
            predicate="classified_as",
            object="gray",
            attributes={"classifier": {"confidence": 0.2}, "human_confirmed": True},
            modality="model_observation",
            attribution={"type": "human_reviewer", "agent": "reviewer_1"},
            truth_status="human_confirmed",
            confidence=0.2,
        )
        append_jsonl(Path("data/jsonl/evidence.jsonl"), evidence)
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), claim)

        result = runner.invoke(app, ["validate-claims"])

        assert result.exit_code == 0, result.stdout
        assert "claims_accepted=1" in result.stdout
        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert validations[0]["status"] == "accepted_extracted"
        assert validations[0]["errors"] == []
