from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord


runner = CliRunner()


def test_export_metta_writes_normalized_claim_expressions(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        append_jsonl(
            Path("data/jsonl/claims.normalized.jsonl"),
            NormalizedClaimRecord(
                normalized_claim_id="nclaim_1",
                claim_id="claim_1",
                source_id="src_1",
                evidence_id="ev_1",
                normalized_claim={
                    "subject": "speaker:alice",
                    "predicate": "asserts",
                    "object": "Hope had three masts.",
                    "qualifiers": {
                        "modality": "asserted",
                        "truth_status": "speaker_asserted_unverified",
                        "attribution": {"type": "speaker", "agent": "alice"},
                        "source_faithful_claim": "The speaker asserted: Hope had three masts.",
                        "confidence": 0.82,
                        "confidence_basis": "validated_claim_confidence",
                    },
                },
            ),
        )

        first = runner.invoke(app, ["export-metta"])
        second = runner.invoke(app, ["export-metta"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "claims=1" in first.stdout

        output = Path("data/reports/claims.metta").read_text(encoding="utf-8")
        assert "; schema: metta.claim_export.v1" in output
        assert '(claim "nclaim_1" "claim_1" "src_1" "ev_1" "speaker:alice" "asserts"' in output
        assert '"Hope had three masts."' in output
        assert '\\"truth_status\\":\\"speaker_asserted_unverified\\"' in output
        assert '(claim-modality "nclaim_1" "asserted"' in output
        assert '(claim-truth-status "nclaim_1" "speaker_asserted_unverified"' in output
        assert '\\"type\\":\\"speaker\\"' in output
        assert '(claim-source-faithful "nclaim_1" "The speaker asserted: Hope had three masts."' in output
        assert '(claim-confidence "nclaim_1" 0.82' in output
        assert '(claim-confidence-basis "nclaim_1" "validated_claim_confidence"' in output

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "export_metta"
        assert jobs[0]["model_id"] == "metta.claim_export.v1"
        assert jobs[0]["input_record_ids"] == ["claims_normalized"]
        assert jobs[0]["metrics"] == {"claims": 1}
