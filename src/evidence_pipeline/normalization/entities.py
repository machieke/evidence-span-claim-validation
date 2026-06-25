from __future__ import annotations

import re


_NON_ID_RE = re.compile(r"[^a-z0-9]+")


def canonical_id(prefix: str, surface: str) -> str:
    normalized = surface.strip().lower()
    slug = _NON_ID_RE.sub("_", normalized).strip("_")
    if not slug:
        slug = "unknown"
    return f"{prefix}:{slug}"
