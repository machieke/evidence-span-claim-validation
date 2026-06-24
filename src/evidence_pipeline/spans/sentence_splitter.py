from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TextSegment:
    text: str
    char_start: int
    char_end: int


_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def split_sentences(text: str) -> Iterator[TextSegment]:
    for match in _SENTENCE_RE.finditer(text):
        start, end = match.span()
        segment = match.group(0)
        leading = len(segment) - len(segment.lstrip())
        trailing = len(segment.rstrip())
        adjusted_start = start + leading
        adjusted_end = start + trailing
        if adjusted_end <= adjusted_start:
            continue
        yield TextSegment(text=text[adjusted_start:adjusted_end], char_start=adjusted_start, char_end=adjusted_end)
