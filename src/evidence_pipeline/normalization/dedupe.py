from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.claims import NormalizedClaimRecord
from evidence_pipeline.schemas.reports import ClaimDuplicateGroupRecord

DEDUPE_VERSION = "claim.dedupe.v1"


@dataclass
class DedupeResult:
    output_path: Path
    group_count: int


DEDUPE_QUALIFIER_KEYS = {"modality", "truth_status"}


def _normalized_proposition(record: NormalizedClaimRecord) -> Dict[str, object]:
    normalized = record.normalized_claim
    proposition = {
        "subject": normalized.get("subject"),
        "predicate": normalized.get("predicate"),
        "object": normalized.get("object"),
    }
    qualifiers = normalized.get("qualifiers")
    if isinstance(qualifiers, dict):
        stable_qualifiers = {
            key: qualifiers[key]
            for key in sorted(DEDUPE_QUALIFIER_KEYS)
            if key in qualifiers
        }
        if stable_qualifiers:
            proposition["qualifiers"] = stable_qualifiers
    return proposition


def _dedupe_key(record: NormalizedClaimRecord) -> str:
    return json.dumps(_normalized_proposition(record), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _omitted_qualifier_keys(records: List[NormalizedClaimRecord]) -> List[str]:
    keys = set()
    for record in records:
        qualifiers = record.normalized_claim.get("qualifiers")
        if not isinstance(qualifiers, dict):
            continue
        keys.update(str(key) for key in qualifiers if key not in DEDUPE_QUALIFIER_KEYS)
    return sorted(keys)


def _duplicate_level(records: List[NormalizedClaimRecord]) -> str:
    source_count = len({record.source_id for record in records})
    evidence_count = len({record.evidence_id for record in records})
    if len(records) == 1:
        return "singleton"
    if source_count > 1:
        return "cross_source_corroboration_candidate"
    if evidence_count > 1:
        return "same_source_distinct_evidence"
    return "same_evidence_duplicate"


def _group_record(key: str, records: List[NormalizedClaimRecord]) -> Dict[str, object]:
    return ClaimDuplicateGroupRecord(
        dedupe_id=stable_id("dedupe", {"normalized_claim": key}),
        normalized_proposition=json.loads(key),
        normalized_claim=records[0].normalized_claim,
        member_count=len(records),
        member_claim_ids=[record.claim_id for record in records],
        member_normalized_claim_ids=[record.normalized_claim_id for record in records],
        source_ids=sorted({record.source_id for record in records}),
        evidence_ids=[record.evidence_id for record in records],
        duplicate_level=_duplicate_level(records),
        omitted_qualifier_keys=_omitted_qualifier_keys(records),
    ).model_dump(mode="json")


def dedupe_normalized_claims(
    config: PipelineConfig,
    output_path: Optional[Path] = None,
    include_singletons: bool = False,
) -> DedupeResult:
    if output_path is None:
        output_path = config.paths.reports_dir / "claim_duplicates.jsonl"
    paths = config.jsonl_paths()
    grouped: Dict[str, List[NormalizedClaimRecord]] = defaultdict(list)
    for _, record in read_jsonl_records(paths["claims_normalized"], NormalizedClaimRecord):
        grouped[_dedupe_key(record)].append(record)

    groups = [
        _group_record(key, records)
        for key, records in sorted(grouped.items())
        if include_singletons or len(records) > 1
    ]
    write_jsonl(output_path, groups)
    return DedupeResult(output_path=output_path, group_count=len(groups))
