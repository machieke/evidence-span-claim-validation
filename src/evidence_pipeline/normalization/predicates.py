from __future__ import annotations


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


def predicate_for_modality(modality: str) -> str:
    return PREDICATE_BY_MODALITY.get(modality, "asserts")
