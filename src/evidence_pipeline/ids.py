from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def stable_id(prefix: str, value: Any, length: int = 24) -> str:
    """Create a deterministic, compact identifier from structured content."""
    digest = hashlib.sha256(_canonical_json(value)).hexdigest()[:length]
    return f"{prefix}_{digest}"


def random_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
