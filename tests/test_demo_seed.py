import json
from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app
from evidence_pipeline.jsonl import read_jsonl


runner = CliRunner()


def _rows(path: str):
    return [payload for _, payload in read_jsonl(Path(path))]


def test_seed_demo_artifacts_finalizes_acceptance_ready_dataset(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        first = runner.invoke(app, ["seed-demo-artifacts"])
        second = runner.invoke(app, ["seed-demo-artifacts"])

        assert first.exit_code == 0, first.stdout
        assert second.exit_code == 0, second.stdout
        assert "records_created=" in first.stdout
        assert "records_created=0" in second.stdout
        assert "gold=data/reports/demo_gold.json" in first.stdout
        assert "gold_claims=10" in first.stdout

        assert len(_rows("data/jsonl/sources.jsonl")) == 25
        assert len(_rows("data/jsonl/chat_messages.jsonl")) == 10
        assert len(_rows("data/jsonl/pdf_blocks.jsonl")) == 3
        assert len(_rows("data/jsonl/audio_utterances.jsonl")) == 3
        assert len(_rows("data/jsonl/images.jsonl")) == 20
        assert len(_rows("data/jsonl/image_regions.jsonl")) == 20
        assert len(_rows("data/jsonl/quarantine.jsonl")) == 1
        gold_payload = json.loads(Path("data/reports/demo_gold.json").read_text(encoding="utf-8"))
        assert len(gold_payload["claims"]) == 10

        finalize = runner.invoke(app, ["finalize-run", "--gold", "data/reports/demo_gold.json"])
        assert finalize.exit_code == 0, finalize.stdout
        assert "passed=True" in finalize.stdout
        assert "failed_checks=0" in finalize.stdout
        assert "gold_eval=data/reports/gold_eval.md" in finalize.stdout
        assert "gold_claims=10" in finalize.stdout

        checks = _rows("data/reports/acceptance_check.jsonl")
        assert checks
        assert {check["status"] for check in checks} == {"passed"}

        report_text = Path("data/reports/extraction_summary.md").read_text(encoding="utf-8")
        assert "| sources | 25 |" in report_text
        assert "| chat_messages | 10 |" in report_text
        assert "| pdf_blocks | 3 |" in report_text
        assert "| audio_utterances | 3 |" in report_text
        assert "| images | 20 |" in report_text
        assert "| quarantine | 1 |" in report_text
        assert "| Gold accepted precision | 100.0% |" in report_text
        assert "| Gold quarantine recall | 100.0% |" in report_text

        jobs = _rows("data/jsonl/jobs.jsonl")
        assert [job["stage"] for job in jobs].count("seed_demo_artifacts") == 1
        assert [job["stage"] for job in jobs].count("eval_gold") == 1
        assert [job["stage"] for job in jobs].count("acceptance_check") == 1

        artifact_check = runner.invoke(app, ["validate-artifacts", "--include-reports"])
        assert artifact_check.exit_code == 0, artifact_check.stdout
