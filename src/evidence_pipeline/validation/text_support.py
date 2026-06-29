from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Set


UNCERTAINTY_MARKERS = {
    "appears",
    "seems",
    "may",
    "might",
    "likely",
    "allegedly",
    "reportedly",
    "possibly",
    "suggests",
    "maybe",
}

NEGATION_MARKERS = {
    "not",
    "no",
    "never",
    "without",
    "neither",
    "failed to",
    "lacks",
    "cannot",
    "can't",
    "won't",
    "didn't",
    "doesn't",
}

NUMBER_WORDS = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
}

_NUMBER_WORD_PATTERN = "|".join(sorted(NUMBER_WORDS))
_UNIT_PATTERN = (
    r"masts?|engines?|meters?|metres?|feet|foot|ft|inches|inch|in|"
    r"cm|mm|km|miles?|mi|kg|g|lbs?|pounds?|oz|tons?|"
    r"hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|months?|years?|"
    r"liters?|litres?|gallons?|percent|%"
)
_NUMERIC_VALUE_PATTERN = r"(?:\d+(?:\.\d+)?|" + _NUMBER_WORD_PATTERN + r")"
_NUMBER_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}-\d{2}-\d{2}|"
    r"\d{1,2}/\d{1,2}/\d{2,4}|"
    r"\d{1,2}:\d{2}(?:\s?[ap]m)?|"
    r"yesterday|today|tomorrow|last week|next week|"
    r"(?:"
    + _NUMERIC_VALUE_PATTERN
    + r")\s*(?:-|to)\s*(?:"
    + _NUMERIC_VALUE_PATTERN
    + r")(?:\s+"
    + _UNIT_PATTERN
    + r")?|"
    r"(?:"
    + _NUMERIC_VALUE_PATTERN
    + r")\s+"
    + _UNIT_PATTERN
    + r"|"
    r"\d+(?:\.\d+)?%?|"
    + _NUMBER_WORD_PATTERN
    + r")\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b")
_WHITESPACE_RE = re.compile(r"\s+")
_COMMON_ENTITY_FALSE_POSITIVES = {
    "I",
    "The",
    "A",
    "An",
    "This",
    "That",
    "It",
    "They",
    "He",
    "She",
    "OCR",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
}


@dataclass(frozen=True)
class EvidenceMatch:
    exact: bool
    normalized: bool


def normalize_text_for_matching(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def evidence_substring_match(evidence_text: str, support_texts: Iterable[str]) -> EvidenceMatch:
    support_list = [text for text in support_texts if text]
    if any(evidence_text in support for support in support_list):
        return EvidenceMatch(exact=True, normalized=True)

    normalized_evidence = normalize_text_for_matching(evidence_text)
    normalized = any(normalized_evidence in normalize_text_for_matching(support) for support in support_list)
    return EvidenceMatch(exact=False, normalized=normalized)


def contains_any_marker(text: str, markers: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(re.search(r"\b" + re.escape(marker) + r"\b", lowered) for marker in markers)


def contains_negation(text: str) -> bool:
    return contains_any_marker(text, NEGATION_MARKERS)


def contains_uncertainty(text: str) -> bool:
    return contains_any_marker(text, UNCERTAINTY_MARKERS)


def extract_quantities(text: str) -> Set[str]:
    return {match.group(0).lower() for match in _NUMBER_DATE_RE.finditer(text)}


def extract_named_entities(text: str) -> Set[str]:
    entities = {match.group(0).strip() for match in _ENTITY_RE.finditer(text)}
    return {entity for entity in entities if entity not in _COMMON_ENTITY_FALSE_POSITIVES}


def unsupported_entities(claim_text: str, support_texts: Iterable[str], allowed_texts: Iterable[str]) -> List[str]:
    supported = set()
    for text in support_texts:
        supported.update(extract_named_entities(text))
    for text in allowed_texts:
        supported.update(extract_named_entities(text))

    introduced = []
    for entity in sorted(extract_named_entities(claim_text)):
        if entity not in supported:
            introduced.append(entity)
    return introduced
