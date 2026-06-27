from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import read_jsonl_records, write_jsonl
from evidence_pipeline.schemas.claims import RawClaimRecord
from evidence_pipeline.schemas.spans import SpanRecord

ROUTING_STAGES = {"all", "extraction", "validation"}


class ModelNames(BaseModel):
    extraction_default: str = "cheap_structured_json_model"
    extraction_strong: str = "strong_reasoning_model"
    validation_default: str = "cheap_validator_model"
    validation_strong: str = "strong_validator_model"


class ExtractionRoutingRules(BaseModel):
    span_score_lt: Optional[float] = 0.60
    risk_flags_any: List[str] = Field(default_factory=lambda: ["context_dependent_coreference"])


class ValidationRoutingRules(BaseModel):
    raw_claim_confidence_lt: Optional[float] = 0.65
    modality: List[str] = Field(default_factory=lambda: ["image"])
    risk_flags_any: List[str] = Field(default_factory=lambda: ["speaker_uncertain", "weak_cluster_margin"])


class RoutingRules(BaseModel):
    use_strong_extractor_if: ExtractionRoutingRules = Field(default_factory=ExtractionRoutingRules)
    use_strong_validator_if: ValidationRoutingRules = Field(default_factory=ValidationRoutingRules)


class ModelsConfig(BaseModel):
    models: ModelNames = Field(default_factory=ModelNames)
    routing: RoutingRules = Field(default_factory=RoutingRules)


@dataclass
class ModelRoutingReportResult:
    output_path: Path
    recommendation_count: int


def load_models_config(path: Path = Path("configs/models.yaml")) -> ModelsConfig:
    if not path.exists():
        return ModelsConfig()
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ModelsConfig.model_validate(payload)


def _matching_risk_flags(record_flags: List[str], configured_flags: List[str]) -> List[str]:
    configured = set(configured_flags)
    return sorted(flag for flag in record_flags if flag in configured)


def _routing_id(stage: str, record_type: str, record_id: str, selected_model: str, reasons: List[str]) -> str:
    return stable_id(
        "route",
        {
            "stage": stage,
            "record_type": record_type,
            "record_id": record_id,
            "selected_model": selected_model,
            "reasons": reasons,
        },
    )


def _span_recommendation(span: SpanRecord, models_config: ModelsConfig) -> dict:
    rules = models_config.routing.use_strong_extractor_if
    reasons = []
    if rules.span_score_lt is not None and span.score is not None and span.score < rules.span_score_lt:
        reasons.append(f"span_score_lt:{rules.span_score_lt}")
    matching_flags = _matching_risk_flags(span.risk_flags, rules.risk_flags_any)
    reasons.extend(f"risk_flag:{flag}" for flag in matching_flags)

    selected_tier = "strong" if reasons else "default"
    selected_model = (
        models_config.models.extraction_strong
        if selected_tier == "strong"
        else models_config.models.extraction_default
    )
    return {
        "routing_id": _routing_id("extract_claims", "span", span.span_id, selected_model, reasons),
        "stage": "extract_claims",
        "record_type": "span",
        "record_id": span.span_id,
        "source_id": span.source_id,
        "source_modality": span.source_modality,
        "model_role": "extraction",
        "selected_tier": selected_tier,
        "selected_model": selected_model,
        "reasons": reasons,
        "score": span.score,
        "schema_version": "model.routing.v1",
    }


def _claim_recommendation(claim: RawClaimRecord, models_config: ModelsConfig) -> dict:
    rules = models_config.routing.use_strong_validator_if
    reasons = []
    if rules.raw_claim_confidence_lt is not None and claim.confidence < rules.raw_claim_confidence_lt:
        reasons.append(f"raw_claim_confidence_lt:{rules.raw_claim_confidence_lt}")
    if claim.source_modality in set(rules.modality):
        reasons.append(f"modality:{claim.source_modality}")
    matching_flags = _matching_risk_flags(claim.risk_flags, rules.risk_flags_any)
    reasons.extend(f"risk_flag:{flag}" for flag in matching_flags)

    selected_tier = "strong" if reasons else "default"
    selected_model = (
        models_config.models.validation_strong
        if selected_tier == "strong"
        else models_config.models.validation_default
    )
    return {
        "routing_id": _routing_id("validate_claims", "claim_raw", claim.claim_id, selected_model, reasons),
        "stage": "validate_claims",
        "record_type": "claim_raw",
        "record_id": claim.claim_id,
        "source_id": claim.source_id,
        "source_modality": claim.source_modality,
        "model_role": "validation",
        "selected_tier": selected_tier,
        "selected_model": selected_model,
        "reasons": reasons,
        "score": claim.confidence,
        "schema_version": "model.routing.v1",
    }


def write_model_routing_report(
    config: PipelineConfig,
    models_config_path: Path = Path("configs/models.yaml"),
    output_path: Optional[Path] = None,
    stage: str = "all",
) -> ModelRoutingReportResult:
    if stage not in ROUTING_STAGES:
        expected = ", ".join(sorted(ROUTING_STAGES))
        raise ValueError(f"model routing supports stages: {expected}")
    if output_path is None:
        output_path = config.paths.reports_dir / "model_routing.jsonl"

    models_config = load_models_config(models_config_path)
    paths = config.jsonl_paths()
    recommendations = []
    if stage in {"all", "extraction"}:
        for _, span in read_jsonl_records(paths["spans"], SpanRecord):
            recommendations.append(_span_recommendation(span, models_config))
    if stage in {"all", "validation"}:
        for _, claim in read_jsonl_records(paths["claims_raw"], RawClaimRecord):
            recommendations.append(_claim_recommendation(claim, models_config))

    write_jsonl(output_path, recommendations)
    return ModelRoutingReportResult(output_path=output_path, recommendation_count=len(recommendations))
