from pathlib import Path

from typer.testing import CliRunner

from evidence_pipeline.cli import app


runner = CliRunner()


def test_init_creates_canonical_artifacts(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        assert Path("data/jsonl/sources.jsonl").exists()
        assert Path("data/jsonl/evidence.jsonl").exists()
        assert Path("data/reports").exists()


def test_register_source_is_idempotent(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        source = Path("input.txt")
        source.write_text("hello", encoding="utf-8")

        first = runner.invoke(app, ["register-source", "input.txt", "--modality", "chat"])
        second = runner.invoke(app, ["register-source", "input.txt", "--modality", "chat"])

        assert first.exit_code == 0
        assert second.exit_code == 0
        assert first.stdout == second.stdout
        rows = Path("data/jsonl/sources.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(rows) == 1


def test_validate_jsonl_reports_valid_count(tmp_path: Path):
    with runner.isolated_filesystem(temp_dir=tmp_path):
        source = Path("input.txt")
        source.write_text("hello", encoding="utf-8")
        register = runner.invoke(app, ["register-source", "input.txt", "--modality", "chat"])
        assert register.exit_code == 0

        result = runner.invoke(app, ["validate-jsonl", "data/jsonl/sources.jsonl", "--schema", "source"])

        assert result.exit_code == 0
        assert "valid 1 records" in result.stdout
