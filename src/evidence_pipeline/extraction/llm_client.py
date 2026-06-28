from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Protocol, Type, TypeVar

from pydantic import BaseModel, ValidationError


class JsonExtractionError(ValueError):
    """Raised when a JSON extraction provider cannot return a valid object."""


@dataclass(frozen=True)
class JsonExtractionRequest:
    prompt: str
    schema_name: str
    schema: Mapping[str, Any]
    provider: str
    model: str
    prompt_version: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JsonExtractionResult:
    payload: Dict[str, Any]
    provider: str
    model: str
    prompt_version: Optional[str] = None
    usage: Mapping[str, Any] = field(default_factory=dict)


class JsonExtractor(Protocol):
    def extract_json(self, request: JsonExtractionRequest) -> JsonExtractionResult:
        """Return one JSON object matching the requested schema."""
        ...


class DeterministicJsonExtractor:
    """Offline provider used by deterministic rules and tests.

    The provider returns a prebuilt payload from request metadata or a fixture map,
    then the shared adapter validation path handles schema enforcement.
    """

    def __init__(self, payloads: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        self._payloads = {key: dict(value) for key, value in (payloads or {}).items()}

    def extract_json(self, request: JsonExtractionRequest) -> JsonExtractionResult:
        payload = request.metadata.get("payload")
        if payload is None and request.prompt_version is not None:
            payload = self._payloads.get(request.prompt_version)
        if payload is None:
            payload = self._payloads.get(request.schema_name)
        if isinstance(payload, BaseModel):
            payload = payload.model_dump(mode="json", exclude_none=True)
        if not isinstance(payload, Mapping):
            raise JsonExtractionError(
                f"deterministic provider has no object payload for {request.schema_name}"
            )
        return JsonExtractionResult(
            payload=dict(payload),
            provider=request.provider,
            model=request.model,
            prompt_version=request.prompt_version,
            usage={"provider": "deterministic"},
        )


ModelT = TypeVar("ModelT", bound=BaseModel)


def extract_json(
    extractor: JsonExtractor,
    request: JsonExtractionRequest,
    output_model: Type[ModelT],
) -> ModelT:
    result = extractor.extract_json(request)
    try:
        return output_model.model_validate(result.payload)
    except ValidationError as exc:
        raise JsonExtractionError(
            f"{request.schema_name} payload failed schema validation: {exc}"
        ) from exc
