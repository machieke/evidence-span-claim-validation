from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord


runner = CliRunner()


def _normalized_claim(claim_id: str, evidence_id: str, object_value: str) -> NormalizedClaimRecord:
    return NormalizedClaimRecord(
        normalized_claim_id=f"n_{claim_id}",
        claim_id=claim_id,
        source_id="src_1",
        evidence_id=evidence_id,
        normalized_claim={
            "subject": "speaker:bob",
            "predicate": "asserts",
            "object": object_value,
            "qualifiers": {"truth_status": "speaker_asserted_unverified"},
        },
    )


def test_dedupe_claims_groups_duplicate_normalized_claims(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        append_jsonl(Path("data/jsonl/claims.normalized.jsonl"), _normalized_claim("claim_1", "ev_1", "Hope had masts."))
        append_jsonl(Path("data/jsonl/claims.normalized.jsonl"), _normalized_claim("claim_2", "ev_2", "Hope had masts."))
        append_jsonl(Path("data/jsonl/claims.normalized.jsonl"), _normalized_claim("claim_3", "ev_3", "Hope had an engine."))

        result = runner.invoke(app, ["dedupe-claims"])

        assert result.exit_code == 0, result.stdout
        groups = [payload for _, payload in read_jsonl(Path("data/reports/claim_duplicates.jsonl"))]
        assert len(groups) == 1
        assert groups[0]["member_count"] == 2
        assert groups[0]["member_claim_ids"] == ["claim_1", "claim_2"]
