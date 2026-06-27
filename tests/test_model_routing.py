from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.spans import SpanRecord


runner = CliRunner()


def _span(span_id: str, score: float) -> SpanRecord:
    return SpanRecord(
        span_id=span_id,
        source_id="src_chat_1",
        source_modality="chat",
        evidence_id=f"ev_{span_id}",
        text="Hope had three masts.",
        char_start=0,
        char_end=21,
        label="claim_bearing",
        score=score,
    )


def _claim(claim_id: str, source_modality: str, confidence: float) -> RawClaimRecord:
    if source_modality == "image":
        return RawClaimRecord(
            claim_id=claim_id,
            source_id="src_img_1",
            source_modality="image",
            evidence_id=f"ev_{claim_id}",
            claim_type="named_visual_classification",
            source_faithful_claim="Model classifier_v1 classified region region_1 as red.",
            subject="region_1",
            predicate="classified_as",
            object="red",
            modality="model_observation",
            attribution={"type": "model", "agent": "classifier_v1"},
            truth_status="model_observation_unverified",
            confidence=confidence,
        )
    return RawClaimRecord(
        claim_id=claim_id,
        source_id="src_chat_1",
        source_modality="chat",
        evidence_id=f"ev_{claim_id}",
        source_faithful_claim="The speaker asserted: Hope had three masts.",
        modality="asserted",
        evidence_text="Hope had three masts.",
        attribution={"type": "speaker", "agent": "alice"},
        truth_status="speaker_asserted_unverified",
        confidence=confidence,
    )


def test_route_models_writes_default_and_strong_recommendations(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        Path("models.yaml").write_text(
            """
models:
  extraction_default: cheap_extract
  extraction_strong: strong_extract
  validation_default: cheap_validate
  validation_strong: strong_validate
routing:
  use_strong_extractor_if:
    span_score_lt: 0.70
    risk_flags_any:
      - context_dependent_coreference
  use_strong_validator_if:
    raw_claim_confidence_lt: 0.65
    modality:
      - image
    risk_flags_any:
      - speaker_uncertain
""",
            encoding="utf-8",
        )
        append_jsonl(Path("data/jsonl/spans.jsonl"), _span("span_low", 0.5))
        append_jsonl(Path("data/jsonl/spans.jsonl"), _span("span_high", 0.9))
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), _claim("claim_low", "chat", 0.5))
        append_jsonl(Path("data/jsonl/claims.raw.jsonl"), _claim("claim_image", "image", 0.9))

        result = runner.invoke(app, ["route-models", "--models-config", "models.yaml"])
        assert result.exit_code == 0, result.stdout
        assert "recommendations=4" in result.stdout

        output_path = Path("data/reports/model_routing.jsonl")
        output_text = output_path.read_text(encoding="utf-8")
        recommendations = {payload["record_id"]: payload for _, payload in read_jsonl(output_path)}
        assert recommendations["span_low"]["selected_tier"] == "strong"
        assert recommendations["span_low"]["selected_model"] == "strong_extract"
        assert recommendations["span_low"]["reasons"] == ["span_score_lt:0.7"]
        assert recommendations["span_high"]["selected_tier"] == "default"
        assert recommendations["span_high"]["selected_model"] == "cheap_extract"
        assert recommendations["claim_low"]["selected_tier"] == "strong"
        assert recommendations["claim_low"]["selected_model"] == "strong_validate"
        assert recommendations["claim_low"]["reasons"] == ["raw_claim_confidence_lt:0.65"]
        assert recommendations["claim_image"]["selected_tier"] == "strong"
        assert recommendations["claim_image"]["reasons"] == ["modality:image"]
        assert "Hope had three masts." not in output_text

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "## Model Routing By Tier" in report_text
        assert "| default | 1 |" in report_text
        assert "| strong | 3 |" in report_text
        assert "## Model Routing By Role" in report_text
        assert "| extraction | 2 |" in report_text
        assert "| validation | 2 |" in report_text

        invalid = runner.invoke(app, ["route-models", "--stage", "embedding"])
        assert invalid.exit_code != 0
        assert "model routing supports stages" in invalid.stdout
