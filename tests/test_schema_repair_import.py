import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def test_import_raw_claims_repairs_valid_candidates_and_quarantines_invalid(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        init = runner.invoke(app, ["init"])
        assert init.exit_code == 0

        candidates = [
            {
                "source_id": "src_chat_1",
                "source_modality": "chat",
                "evidence_id": "ev_1",
                "claim": "The speaker asserted: Hope had three masts.",
                "modality": "asserted",
                "evidence": "Hope had three masts.",
                "attribution": "alice",
                "truth_status": "speaker_asserted_unverified",
                "confidence": "82%",
            },
            {
                "claim": "This candidate is missing required provenance.",
                "confidence": 0.9,
            },
        ]
        Path("candidates.json").write_text(json.dumps({"claims": candidates}), encoding="utf-8")

        first = runner.invoke(app, ["import-raw-claims", "candidates.json"])
        second = runner.invoke(app, ["import-raw-claims", "candidates.json"])
        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "claims_imported=1" in first.stdout
        assert "claims_repaired=1" in first.stdout
        assert "claims_quarantined=1" in first.stdout
        assert "claims_skipped=2" in second.stdout

        raw_claims = [payload for _, payload in read_jsonl(Path("data/jsonl/claims.raw.jsonl"))]
        assert len(raw_claims) == 1
        assert raw_claims[0]["claim_id"].startswith("claim_import_")
        assert raw_claims[0]["source_faithful_claim"] == "The speaker asserted: Hope had three masts."
        assert raw_claims[0]["evidence_text"] == "Hope had three masts."
        assert raw_claims[0]["attribution"] == {"agent": "alice", "type": "speaker"}
        assert raw_claims[0]["confidence"] == 0.82

        validations = [payload for _, payload in read_jsonl(Path("data/jsonl/validations.jsonl"))]
        assert {record["status"] for record in validations} == {"repaired", "schema_invalid"}
        repaired_validation = next(record for record in validations if record["status"] == "repaired")
        assert "rename_claim_to_source_faithful_claim" in repaired_validation["warnings"]
        assert "scale_confidence_percent" in repaired_validation["warnings"]

        quarantined = [payload for _, payload in read_jsonl(Path("data/jsonl/quarantine.jsonl"))]
        assert len(quarantined) == 1
        assert quarantined[0]["record_type"] == "claim_candidate"
        assert quarantined[0]["reason_codes"] == ["schema_invalid_after_repair"]
        assert "repaired_payload" in quarantined[0]["payload"]

        jobs = [payload for _, payload in read_jsonl(Path("data/jsonl/jobs.jsonl"))]
        assert len(jobs) == 1
        assert jobs[0]["stage"] == "import_raw_claims"
        assert jobs[0]["model_id"] == "schema_repair.v1"
        assert jobs[0]["input_record_ids"] == ["candidates.json"]
        assert jobs[0]["metrics"] == {
            "claims_imported": 1,
            "claims_quarantined": 1,
            "claims_repaired": 1,
            "claims_skipped": 0,
        }

        report = runner.invoke(app, ["report"])
        assert report.exit_code == 0, report.stdout
        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| jobs | 1 |" in report_text
        assert "| import_raw_claims | 1 |" in report_text

        artifact_check = runner.invoke(app, ["validate-artifacts"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
