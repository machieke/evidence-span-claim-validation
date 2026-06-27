from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image

from evidence_pipeline.config import PipelineConfig
from evidence_pipeline.ids import stable_id
from evidence_pipeline.jsonl import append_jsonl, existing_values, read_jsonl_records
from evidence_pipeline.schemas.image import (
    ImageFeatureClusterRecord,
    ImageRegionEmbeddingRecord,
    ImageRegionRecord,
)

COLOR_EMBEDDING_MODEL = "color_rgb_mean_std_v1"
COLOR_CLUSTERING_METHOD = "connected_components_color_distance_v1"


@dataclass
class ImageRegionEmbeddingResult:
    created: int
    skipped: int


@dataclass
class ImageFeatureClusterResult:
    created: int
    skipped: int
    clustered_regions: int


def _embedding_id(region: ImageRegionRecord, embedding_model: str) -> str:
    return stable_id("img_emb", {"region_id": region.region_id, "embedding_model": embedding_model})


def _cluster_id(
    member_region_ids: Sequence[str],
    embedding_model: str,
    clustering_method: str,
) -> str:
    return stable_id(
        "img_cluster",
        {
            "member_region_ids": sorted(member_region_ids),
            "embedding_model": embedding_model,
            "clustering_method": clustering_method,
        },
    )


def _color_embedding(crop_path: Path) -> List[float]:
    with Image.open(crop_path) as image:
        thumbnail = image.convert("RGB").resize((16, 16))
        pixels = list(thumbnail.getdata())
    channel_values = [[pixel[index] / 255.0 for pixel in pixels] for index in range(3)]
    means = [sum(values) / len(values) for values in channel_values]
    stds = [
        math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
        for values, mean in zip(channel_values, means)
    ]
    return [round(value, 6) for value in means + stds]


def build_image_region_embeddings(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    embedding_model: str = COLOR_EMBEDDING_MODEL,
) -> ImageRegionEmbeddingResult:
    paths = config.jsonl_paths()
    existing_ids = existing_values(paths["image_region_embeddings"], "embedding_id")
    created = 0
    skipped = 0

    for _, region in read_jsonl_records(paths["image_regions"], ImageRegionRecord):
        if source_id is not None and region.source_id != source_id:
            continue
        embedding_id = _embedding_id(region, embedding_model)
        if embedding_id in existing_ids:
            skipped += 1
            continue
        if region.crop_path is None or not Path(region.crop_path).exists():
            skipped += 1
            continue
        vector = _color_embedding(Path(region.crop_path))
        append_jsonl(
            paths["image_region_embeddings"],
            ImageRegionEmbeddingRecord(
                embedding_id=embedding_id,
                region_id=region.region_id,
                image_id=region.image_id,
                source_id=region.source_id,
                embedding_model=embedding_model,
                embedding_dim=len(vector),
                vector=vector,
                preprocessing={"crop_resize": 16, "normalize": "rgb_0_1"},
                risk_flags=region.risk_flags,
            ),
        )
        existing_ids.add(embedding_id)
        created += 1

    return ImageRegionEmbeddingResult(created=created, skipped=skipped)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _connected_components(
    embeddings: Sequence[ImageRegionEmbeddingRecord],
    distance_threshold: float,
) -> List[List[ImageRegionEmbeddingRecord]]:
    neighbors: Dict[str, Set[str]] = {record.region_id: set() for record in embeddings}
    by_region_id = {record.region_id: record for record in embeddings}

    for index, left in enumerate(embeddings):
        for right in embeddings[index + 1 :]:
            if _distance(left.vector, right.vector) <= distance_threshold:
                neighbors[left.region_id].add(right.region_id)
                neighbors[right.region_id].add(left.region_id)

    components: List[List[ImageRegionEmbeddingRecord]] = []
    seen: Set[str] = set()
    for record in embeddings:
        if record.region_id in seen:
            continue
        stack = [record.region_id]
        component_ids: List[str] = []
        seen.add(record.region_id)
        while stack:
            current = stack.pop()
            component_ids.append(current)
            for neighbor in sorted(neighbors[current]):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                stack.append(neighbor)
        components.append([by_region_id[region_id] for region_id in sorted(component_ids)])
    return components


def _pairwise_distances(records: Sequence[ImageRegionEmbeddingRecord]) -> List[float]:
    distances: List[float] = []
    for index, left in enumerate(records):
        for right in records[index + 1 :]:
            distances.append(_distance(left.vector, right.vector))
    return distances


def _centroid(records: Sequence[ImageRegionEmbeddingRecord]) -> List[float]:
    dimensions = len(records[0].vector)
    return [
        sum(record.vector[index] for record in records) / len(records)
        for index in range(dimensions)
    ]


def _representatives(records: Sequence[ImageRegionEmbeddingRecord], limit: int = 2) -> List[str]:
    centroid = _centroid(records)
    ranked = sorted(records, key=lambda record: (_distance(record.vector, centroid), record.region_id))
    return [record.region_id for record in ranked[:limit]]


def _nearest_external_distance(
    component: Sequence[ImageRegionEmbeddingRecord],
    all_embeddings: Sequence[ImageRegionEmbeddingRecord],
) -> float:
    component_ids = {record.region_id for record in component}
    external = [record for record in all_embeddings if record.region_id not in component_ids]
    if not external:
        return 1.0
    return min(_distance(member.vector, other.vector) for member in component for other in external)


def _cluster_scores(
    component: Sequence[ImageRegionEmbeddingRecord],
    all_embeddings: Sequence[ImageRegionEmbeddingRecord],
) -> Tuple[float, float]:
    internal_distances = _pairwise_distances(component)
    average_internal_distance = sum(internal_distances) / len(internal_distances) if internal_distances else 0.0
    max_internal_distance = max(internal_distances) if internal_distances else 0.0
    nearest_external_distance = _nearest_external_distance(component, all_embeddings)
    cohesion_score = max(0.0, min(1.0, 1.0 - average_internal_distance))
    margin = max(0.0, min(1.0, nearest_external_distance - max_internal_distance))
    return round(cohesion_score, 6), round(margin, 6)


def _cluster_risk_flags(component: Sequence[ImageRegionEmbeddingRecord], margin: float) -> List[str]:
    flags = {flag for record in component for flag in record.risk_flags}
    source_ids = {record.source_id for record in component}
    if len(component) < 3:
        flags.add("small_cluster")
    if len(source_ids) == 1:
        flags.add("same_source_cluster")
    if margin < 0.05:
        flags.add("weak_cluster_margin")
    return sorted(flags)


def cluster_image_regions(
    config: PipelineConfig,
    source_id: Optional[str] = None,
    embedding_model: str = COLOR_EMBEDDING_MODEL,
    distance_threshold: float = 0.08,
    min_cluster_size: int = 2,
) -> ImageFeatureClusterResult:
    paths = config.jsonl_paths()
    if distance_threshold < 0:
        raise ValueError("distance_threshold must be non-negative")
    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1")

    embeddings = [
        record
        for _, record in read_jsonl_records(paths["image_region_embeddings"], ImageRegionEmbeddingRecord)
        if record.embedding_model == embedding_model and (source_id is None or record.source_id == source_id)
    ]
    embeddings.sort(key=lambda record: record.region_id)
    existing_ids = existing_values(paths["image_feature_clusters"], "feature_cluster_id")
    created = 0
    skipped = 0
    clustered_regions = 0

    for component in _connected_components(embeddings, distance_threshold):
        if len(component) < min_cluster_size:
            skipped += len(component)
            continue
        member_region_ids = [record.region_id for record in component]
        feature_cluster_id = _cluster_id(member_region_ids, embedding_model, COLOR_CLUSTERING_METHOD)
        if feature_cluster_id in existing_ids:
            skipped += len(component)
            continue
        cohesion_score, margin = _cluster_scores(component, embeddings)
        append_jsonl(
            paths["image_feature_clusters"],
            ImageFeatureClusterRecord(
                feature_cluster_id=feature_cluster_id,
                embedding_model=embedding_model,
                clustering_method=COLOR_CLUSTERING_METHOD,
                member_region_ids=member_region_ids,
                cluster_size=len(member_region_ids),
                cohesion_score=cohesion_score,
                nearest_neighbor_margin=margin,
                representative_region_ids=_representatives(component),
                source_ids=sorted({record.source_id for record in component}),
                risk_flags=_cluster_risk_flags(component, margin),
            ),
        )
        existing_ids.add(feature_cluster_id)
        created += 1
        clustered_regions += len(component)

    return ImageFeatureClusterResult(created=created, skipped=skipped, clustered_regions=clustered_regions)
