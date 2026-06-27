from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import Field, model_validator

from evidence_pipeline.schemas.base import SourceModality, StrictModel

ClaimModality = Literal[
    "asserted",
    "direct_observation",
    "uncertain_observation",
    "reported",
    "reported_direct_observation",
    "reported_uncertain",
    "negated",
    "hypothetical",
    "question_asked",
    "model_observation",
]

TruthStatus = Literal[
    "source_asserted_unverified",
    "speaker_asserted_unverified",
    "model_observation_unverified",
    "human_confirmed",
    "unknown",
]

SupportStatus = Literal[
    "raw_extracted",
    "schema_valid",
    "deterministic_valid",
    "semantic_valid",
    "accepted_extracted",
    "needs_review",
    "quarantined",
    "normalized",
    "exported",
]


class Attribution(StrictModel):
    type: Literal["speaker", "document", "model", "human_reviewer", "unknown"]
    agent: Optional[str] = None


class ModelInfo(StrictModel):
    provider: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None


class RawClaimRecord(StrictModel):
    claim_id: str
    source_id: str
    source_modality: SourceModality
    span_id: Optional[str] = None
    evidence_id: str
    claim_type: str = "attributed_text_claim"
    source_faithful_claim: str
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[Union[str, List[Any], Dict[str, Any]]] = None
    quantity: Optional[Union[int, float, str]] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    modality: ClaimModality
    evidence_text: Optional[str] = None
    context_dependent: bool = False
    context_used: Optional[str] = None
    attribution: Attribution
    truth_status: TruthStatus
    confidence: float
    model: ModelInfo = Field(default_factory=ModelInfo)
    support_status: SupportStatus = "raw_extracted"
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "claim.raw.v1"

    @model_validator(mode="after")
    def validate_claim(self) -> "RawClaimRecord":
        if not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
        if self.source_modality != "image" and (self.evidence_text is None or not self.evidence_text.strip()):
            raise ValueError("text-like claims require evidence_text")
        if self.context_dependent and not self.context_used:
            raise ValueError("context_dependent claims require context_used")
        return self


class ClaimValidationSummary(StrictModel):
    deterministic_valid: bool
    evidence_exact_match: Optional[bool] = None
    negation_preserved: Optional[bool] = None
    uncertainty_preserved: Optional[bool] = None
    attribution_preserved: Optional[bool] = None
    quantities_preserved: Optional[bool] = None
    introduced_entities: List[str] = Field(default_factory=list)
    validator_version: str = "deterministic.v1"


class ValidatedClaimRecord(StrictModel):
    claim_id: str
    source_id: str
    source_modality: SourceModality
    span_id: Optional[str] = None
    evidence_id: str
    source_faithful_claim: str
    evidence_text: Optional[str] = None
    normalized_claim: Optional[Dict[str, Any]] = None
    modality: ClaimModality
    truth_status: TruthStatus
    support_status: SupportStatus
    validation: ClaimValidationSummary
    risk_flags: List[str] = Field(default_factory=list)
    schema_version: str = "claim.validated.v1"

    @model_validator(mode="after")
    def validate_status(self) -> "ValidatedClaimRecord":
        if self.support_status not in {"accepted_extracted", "needs_review", "quarantined", "semantic_valid", "deterministic_valid"}:
            raise ValueError("validated claims require a validation-stage support_status")
        if self.support_status == "accepted_extracted" and not self.validation.deterministic_valid:
            raise ValueError("accepted extracted claims require deterministic_valid validation")
        return self


class EntityResolution(StrictModel):
    surface: str
    canonical_id: str
    confidence: float
    basis: str

    @model_validator(mode="after")
    def validate_confidence(self) -> "EntityResolution":
        if not (0 <= self.confidence <= 1):
            raise ValueError("confidence must be between 0 and 1")
        return self


class PredicateMapping(StrictModel):
    surface: str
    canonical: str


class NormalizationDetails(StrictModel):
    entity_resolution: List[EntityResolution] = Field(default_factory=list)
    predicate_mapping: Optional[PredicateMapping] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class NormalizedClaimRecord(StrictModel):
    normalized_claim_id: str
    claim_id: str
    source_id: str
    evidence_id: str
    normalized_claim: Dict[str, Any]
    normalization: NormalizationDetails = Field(default_factory=NormalizationDetails)
    schema_version: str = "claim.normalized.v1"

    @model_validator(mode="after")
    def validate_normalized_claim_shape(self) -> "NormalizedClaimRecord":
        required_keys = {"subject", "predicate", "object"}
        missing = sorted(required_keys - set(self.normalized_claim))
        if missing:
            raise ValueError(f"normalized_claim missing required keys: {', '.join(missing)}")
        return self
