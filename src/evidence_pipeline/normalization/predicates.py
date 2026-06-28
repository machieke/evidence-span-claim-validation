from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


PREDICATE_REGISTRY_VERSION = "predicate.registry.v1"


@dataclass(frozen=True)
class PredicateDefinition:
    canonical: str
    description: str


PREDICATE_REGISTRY: Dict[str, PredicateDefinition] = {
    "asserts": PredicateDefinition("asserts", "Source or speaker asserts a proposition."),
    "reports": PredicateDefinition("reports", "Source reports a proposition without direct-observation framing."),
    "reports_observation": PredicateDefinition("reports_observation", "Source reports a direct observation."),
    "asserts_uncertain": PredicateDefinition("asserts_uncertain", "Source expresses uncertainty about a proposition."),
    "asks_whether": PredicateDefinition("asks_whether", "Source asks whether a proposition is true."),
    "negates": PredicateDefinition("negates", "Source denies or negates a proposition."),
    "classified_as": PredicateDefinition("classified_as", "Model or reviewer labels a visual region."),
    "has_member_regions": PredicateDefinition("has_member_regions", "Visual feature cluster has member regions."),
    "proposes_visual_region": PredicateDefinition("proposes_visual_region", "Model proposes a visual evidence region."),
    "model_observation": PredicateDefinition("model_observation", "Model emits an unverified visual observation."),
}

PREDICATE_BY_MODALITY = {
    "asserted": "asserts",
    "direct_observation": "reports_observation",
    "uncertain_observation": "asserts_uncertain",
    "reported": "reports",
    "reported_direct_observation": "reports_observation",
    "reported_uncertain": "asserts_uncertain",
    "negated": "negates",
    "hypothetical": "asks_whether",
    "question_asked": "asks_whether",
    "model_observation": "model_observation",
}

PREDICATE_BY_CLAIM_TYPE = {
    "named_visual_classification": "classified_as",
    "unnamed_visual_feature_cluster": "has_member_regions",
    "visual_region_proposal": "proposes_visual_region",
}


def predicate_for_modality(modality: str) -> str:
    return PREDICATE_BY_MODALITY.get(modality, "asserts")


def predicate_for_claim(modality: str, claim_type: Optional[str] = None, raw_predicate: Optional[str] = None) -> str:
    if claim_type in PREDICATE_BY_CLAIM_TYPE:
        return PREDICATE_BY_CLAIM_TYPE[claim_type]
    if raw_predicate in PREDICATE_REGISTRY:
        return raw_predicate
    return predicate_for_modality(modality)


def predicate_definition(predicate: str) -> PredicateDefinition:
    return PREDICATE_REGISTRY.get(predicate, PREDICATE_REGISTRY["asserts"])
