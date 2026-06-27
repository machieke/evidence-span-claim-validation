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

        result = runner.invoke(app, ["repair-claims"])

        assert result.exit_code == 0, result.stdout
        suggestions = [payload for _, payload in read_jsonl(Path("data/reports/claim_repairs.jsonl"))]
        assert len(suggestions) == 1
        assert suggestions[0]["claim_id"] == "claim_1"
        assert suggestions[0]["original_evidence_text"] == "The vessel   Hope appears old."
        assert suggestions[0]["suggested_evidence_text"] == "The vessel Hope appears old."
        assert suggestions[0]["support_scope"] == "span"
