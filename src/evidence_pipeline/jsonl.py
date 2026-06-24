from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Set, Tuple, Type

from pydantic import BaseModel, ValidationError


class JSONLDecodeError(ValueError):
    def __init__(self, path: Path, line_number: int, message: str) -> None:
        super().__init__(f"{path}:{line_number}: {message}")
        self.path = path
        self.line_number = line_number


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, record: Any) -> None:
    ensure_parent(path)
    if isinstance(record, BaseModel):
        payload = record.model_dump(mode="json", exclude_none=True)
    else:
        payload = record
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        handle.write("\n")


def read_jsonl(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                decoded = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise JSONLDecodeError(path, line_number, str(exc))
            if not isinstance(decoded, dict):
                raise JSONLDecodeError(path, line_number, "expected JSON object")
            yield line_number, decoded


def read_jsonl_records(path: Path, model: Type[BaseModel]) -> Iterator[Tuple[int, BaseModel]]:
    for line_number, payload in read_jsonl(path):
        try:
            yield line_number, model.model_validate(payload)
        except ValidationError as exc:
            raise JSONLDecodeError(path, line_number, str(exc))


def write_jsonl(path: Path, records: Iterable[Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            if isinstance(record, BaseModel):
                payload = record.model_dump(mode="json", exclude_none=True)
            else:
                payload = record
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
            handle.write("\n")


def find_record(path: Path, key: str, value: Any) -> Optional[Dict[str, Any]]:
    for _, payload in read_jsonl(path):
        if payload.get(key) == value:
            return payload
    return None


def existing_values(path: Path, key: str) -> Set[Any]:
    return {payload[key] for _, payload in read_jsonl(path) if key in payload}
