from pathlib import Path

from evidence_pipeline.jsonl import append_jsonl, read_jsonl
from evidence_pipeline.schemas.sources import SourceRecord


def test_append_and_read_jsonl_round_trip(tmp_path: Path):
    path = tmp_path / "sources.jsonl"
    record = SourceRecord(
        source_id="src_1",
        source_modality="chat",
        source_file="data/raw/chat/export.json",
        sha256="abc",
    )

    append_jsonl(path, record)
    rows = list(read_jsonl(path))

    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][1]["source_id"] == "src_1"
    assert rows[0][1]["schema_version"] == "source.v1"
