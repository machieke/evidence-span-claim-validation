import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord


runner = CliRunner()


def _normalized_claim(
    claim_id: str,
    evidence_id: str,
    object_value: str,
    source_id: str = "src_1",
    subject: str = "speaker:bob",
    predicate: str = "asserts",
    qualifiers: dict = None,
) -> NormalizedClaimRecord:
    return NormalizedClaimRecord(
        normalized_claim_id=f"n_{claim_id}",
        claim_id=claim_id,
        source_id=source_id,
        evidence_id=evidence_id,
        normalized_claim={
            "subject": subject,
            "predicate": predicate,
            "object": object_value,
            "qualifiers": qualifiers or {"truth_status": "speaker_asserted_unverified"},
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

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "dedupe_claims"
        assert jobs[0]["model_id"] == "claim.dedupe.v1"
        assert jobs[0]["input_record_ids"] == ["claims_normalized"]
        assert jobs[0]["metrics"] == {"groups": 1}

        trace = runner.invoke(app, ["trace-claim", "claim_1"])
        assert trace.exit_code == 0, trace.stdout
        trace_payload = json.loads(trace.stdout)
        assert trace_payload["duplicate_groups"][0]["dedupe_id"] == groups[0]["dedupe_id"]

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| dedupe_claims | 1 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
        assert "data/reports/claim_duplicates.jsonl: checked 1 records" in artifact_check.stdout


def test_dedupe_claims_groups_cross_source_normalized_propositions(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0
        append_jsonl(
            Path("data/jsonl/claims.normalized.jsonl"),
            _normalized_claim(
                "claim_1",
                "ev_1",
                "mast",
                source_id="src_1",
                subject="entity:vessel_hope",
                predicate="has_feature",
                qualifiers={
                    "truth_status": "source_asserted_unverified",
                    "attribution": {"type": "document", "agent": "src_1"},
                    "source_faithful_claim": "Document one states Hope has a mast.",
                },
            ),
        )
        append_jsonl(
            Path("data/jsonl/claims.normalized.jsonl"),
            _normalized_claim(
                "claim_2",
                "ev_2",
                "mast",
                source_id="src_2",
                subject="entity:vessel_hope",
                predicate="has_feature",
                qualifiers={
                    "truth_status": "source_asserted_unverified",
                    "attribution": {"type": "document", "agent": "src_2"},
                    "source_faithful_claim": "Document two states Hope has a mast.",
                },
            ),
        )

        result = runner.invoke(app, ["dedupe-claims"])

        assert result.exit_code == 0, result.stdout
        groups = [payload for _, payload in read_jsonl(Path("data/reports/claim_duplicates.jsonl"))]
        assert len(groups) == 1
        assert groups[0]["member_count"] == 2
        assert groups[0]["source_ids"] == ["src_1", "src_2"]
        assert groups[0]["normalized_proposition"] == {
            "subject": "entity:vessel_hope",
            "predicate": "has_feature",
            "object": "mast",
            "qualifiers": {"truth_status": "source_asserted_unverified"},
        }
        assert "attribution" not in groups[0]["normalized_proposition"]["qualifiers"]
