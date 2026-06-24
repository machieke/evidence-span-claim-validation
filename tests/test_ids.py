from pathlib import Path

from evidence_pipeline.ids import sha256_file, stable_id


def test_stable_id_is_deterministic_for_structured_data():
    first = stable_id("src", {"b": 2, "a": 1})
    second = stable_id("src", {"a": 1, "b": 2})

    assert first == second
    assert first.startswith("src_")


def test_sha256_file(tmp_path: Path):
    path = tmp_path / "source.txt"
    path.write_text("hello\n", encoding="utf-8")

    assert sha256_file(path) == "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
