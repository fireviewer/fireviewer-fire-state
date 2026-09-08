"""Deterministic CPU fusion of heterogeneous wildfire spatial observations."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import numpy as np
import rasterio  # type: ignore[import-untyped]
from pyproj import Transformer
from rasterio.features import rasterize, shapes  # type: ignore[import-untyped]
from rasterio.shutil import copy as raster_copy  # type: ignore[import-untyped]
from rasterio.transform import Affine, from_origin  # type: ignore[import-untyped]
from rasterio.warp import Resampling, reproject  # type: ignore[import-untyped]
from shapely import get_num_coordinates, make_valid, normalize, set_precision
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    mapping,
    shape,
)
from shapely.geometry.polygon import orient
from shapely.ops import transform as geometry_transform
from shapely.ops import unary_union

from fireviewer_contracts.backend.fire_state_schemas import (
    DailyFireQuality,
    DailyFireStateV2,
    FireProbabilityGridReference,
    FireSpatialProvenanceGridReference,
    FusionProfileV1,
    PerimeterEstimateV3,
    SensorContributionV1,
    SpatialObservationV2,
)
from fireviewer_contracts.backend.hashing import sha256_hex
from fireviewer_contracts.backend.part4_framing_schemas import FramingArtifactReference, Part4SpatialContextV1
from fireviewer_fire_state.part4_fusion_profiles import (
    DEFAULT_FUSION_PROFILE_ID,
    load_fusion_profile,
)
from fireviewer_fire_state.storage import ObjectStorageError, ObjectStore

FUSION_ALGORITHM_ID = "fireviewer.part4.spatiotemporal-probability-fusion"
FUSION_ALGORITHM_VERSION = "3.3.0"
BASELINE_ALGORITHM_VERSION = "3.2.1"
_PARIS = ZoneInfo("Europe/Paris")
_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)


@dataclass(frozen=True, slots=True)
class FusedFireState:
    state: DailyFireStateV2
    affected_probability: np.ndarray[Any, np.dtype[np.float32]] | None
    active_probability: np.ndarray[Any, np.dtype[np.float32]] | None
    observable_probability: np.ndarray[Any, np.dtype[np.float32]] | None
    observed_support: np.ndarray[Any, np.dtype[np.float32]] | None
    multi_source_support: np.ndarray[Any, np.dtype[np.float32]] | None
    prior_interpolated_support: np.ndarray[Any, np.dtype[np.float32]] | None
    uncertainty_support: np.ndarray[Any, np.dtype[np.float32]] | None
    transform: Affine | None
    lineage_grids: dict[str, np.ndarray[Any, Any]] = field(default_factory=dict)
    lineage_metadata: dict[str, Any] = field(default_factory=dict)


_DIRECT_GEOMETRY_OBSERVATIONS = frozenset(
    {
        "active_probability",
        "burned_probability",
        "modelled_perimeter",
        "official_perimeter",
    }
)
_MULTI_SOURCE_INELIGIBLE_OBSERVATIONS = frozenset({"modelled_perimeter"})


def _polygonal(value: object) -> Polygon | MultiPolygon | None:
    candidate = make_valid(cast(Any, value))
    if isinstance(candidate, Polygon | MultiPolygon):
        return candidate
    if isinstance(candidate, GeometryCollection):
        polygons = [item for item in candidate.geoms if isinstance(item, Polygon | MultiPolygon)]
        if polygons:
            merged = unary_union(polygons)
            if isinstance(merged, Polygon | MultiPolygon):
                return merged
    return None


def _projected_observation_geometry(item: SpatialObservationV2) -> Polygon | MultiPolygon | None:
    if item.geometry_geojson is None:
        return None
    try:
        source = shape(item.geometry_geojson)
    except (TypeError, ValueError):
        return None
    projected = geometry_transform(_TO_L93.transform, source)
    if isinstance(projected, Point):
        radius = max(item.horizontal_accuracy_m or item.resolution_m or 100.0, 1.0)
        projected = projected.buffer(radius)
    polygon = _polygonal(projected)
    if polygon is None or polygon.is_empty:
        return None
    return _polygonal(set_precision(polygon, grid_size=0.1, mode="valid_output"))


def _projected_geojson(value: dict[str, Any] | None) -> Polygon | MultiPolygon | None:
    if value is None:
        return None
    try:
        source = shape(value)
    except (TypeError, ValueError):
        return None
    return _polygonal(geometry_transform(_TO_L93.transform, source))


def _boundary_uncertainty_band(geometry: Any, distance_m: float, cell_size_m: float = 20) -> Any:
    """Buffer a polygon boundary without noding every detailed boundary segment.

    Dilation minus erosion is the same two-sided distance band, including holes
    and narrow components. The original affected geometry is never simplified.
    Buffering a dense MultiLineString directly can require gigabytes of GEOS
    intermediates even for a small incident.
    """
    if get_num_coordinates(geometry) <= 50_000 or distance_m <= cell_size_m:
        return geometry.buffer(distance_m).difference(geometry.buffer(-distance_m))
    # The same signed radius composition as the search domain keeps complex
    # contour intermediates bounded. GEOS approximates circular arcs in both
    # paths; the source polygon and the declared radius remain unchanged.
    step = min(distance_m, cell_size_m)
    outer = geometry.buffer(step).buffer(distance_m - step)
    inner = geometry.buffer(-step).buffer(-(distance_m - step))
    return outer.difference(inner)


def _grid_for_bounds(
    bounds: tuple[float, float, float, float],
    profile: FusionProfileV1,
    minimum_resolution: float = 0,
) -> tuple[Affine, int, int, float]:
    min_x, min_y, max_x, max_y = bounds
    if not all(math.isfinite(item) for item in bounds) or max_x <= min_x or max_y <= min_y:
        raise ValueError("fire state AOI bounds are invalid")
    for resolution in profile.grid.resolutions_m:
        if resolution < minimum_resolution:
            continue
        west = math.floor((min_x - profile.grid.aoi_margin_m) / resolution) * resolution
        south = math.floor((min_y - profile.grid.aoi_margin_m) / resolution) * resolution
        east = math.ceil((max_x + profile.grid.aoi_margin_m) / resolution) * resolution
        north = math.ceil((max_y + profile.grid.aoi_margin_m) / resolution) * resolution
        width = max(1, round((east - west) / resolution))
        height = max(1, round((north - south) / resolution))
        if width * height <= profile.grid.max_cells:
            return from_origin(west, north, resolution, resolution), width, height, resolution
    raise ValueError("fire state AOI exceeds the four-million-cell budget")


def _mask(
    geometry: Polygon | MultiPolygon,
    *,
    transform: Affine,
    width: int,
    height: int,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    result = rasterize(
        [(mapping(geometry), 1)],
        out_shape=(height, width),
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)
    return np.asarray(result, dtype=np.bool_)


def boundary_uncertainty_mask(
    geometry: Polygon | MultiPolygon,
    distance_m: float,
    *,
    transform: Affine,
    width: int,
    height: int,
) -> np.ndarray[Any, np.dtype[np.bool_]]:
    """Rasterize the distance band without a global million-segment buffer.

    Distance to a boundary is distance to the union of its line segments.
    Buffering overlapping line chunks and rasterizing their union therefore
    keeps the same radius, holes and all-touched semantics. Only GEOS's usual
    circular-arc discretization applies; no input contour is simplified.
    The batches bound intermediate vector memory independently of vertex count.
    """
    if get_num_coordinates(geometry) <= 50_000:
        return _mask(
            _boundary_uncertainty_band(geometry, distance_m, float(transform.a)),
            transform=transform,
            width=width,
            height=height,
        )
    output = np.zeros((height, width), dtype=np.uint8)
    batch: list[tuple[Any, int]] = []
    polygons = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    for polygon in polygons:
        for ring in (polygon.exterior, *polygon.interiors):
            coordinates = np.asarray(ring.coords)
            for start in range(0, len(coordinates) - 1, 1024):
                segment = LineString(coordinates[start : start + 1025])
                batch.append((segment.buffer(distance_m), 1))
                if len(batch) == 128:
                    rasterize(batch, out=output, transform=transform, all_touched=True)
                    batch.clear()
    if batch:
        rasterize(batch, out=output, transform=transform, all_touched=True)
    return output.astype(bool)


def _logit(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    clipped = np.clip(value, 0.001, 0.999)
    return np.asarray(np.log(clipped / (1.0 - clipped)))


def _sigmoid(value: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    return np.asarray(1.0 / (1.0 + np.exp(-value)))


def _apply_positive(
    values: np.ndarray[Any, Any],
    selected: np.ndarray[Any, Any],
    *,
    probabilities: np.ndarray[Any, Any],
    profile_probability: float,
    weight: float,
    direct_support_threshold: float | None = None,
    preserve_direct_support: bool = False,
    previous_probabilities: np.ndarray[Any, Any] | None = None,
) -> None:
    raw_probability = probabilities[selected]
    support = 0.5 + (profile_probability - 0.5) * raw_probability
    delta = np.log(support / (1.0 - support)) * weight
    if previous_probabilities is not None:
        previous_support = 0.5 + (profile_probability - 0.5) * previous_probabilities[selected]
        delta -= np.log(previous_support / (1.0 - previous_support)) * weight
    updated = _sigmoid(_logit(values[selected]) + delta)
    if preserve_direct_support and direct_support_threshold is not None:
        direct = raw_probability >= direct_support_threshold
        updated[direct] = np.maximum(updated[direct], support[direct])
    values[selected] = updated


def _canonical_multipolygon(
    geometry: Polygon | MultiPolygon, *, round_coordinates: bool = True
) -> dict[str, Any]:
    wgs84 = _polygonal(geometry_transform(_TO_WGS84.transform, geometry))
    if wgs84 is None or wgs84.is_empty:
        raise ValueError("fire state polygonization produced no WGS84 geometry")
    return _canonical_wgs84_multipolygon(wgs84, round_coordinates=round_coordinates)


def _canonical_wgs84_multipolygon(
    geometry: Polygon | MultiPolygon, *, round_coordinates: bool
) -> dict[str, Any]:
    normalized = normalize(geometry)
    polygons = [normalized] if isinstance(normalized, Polygon) else list(normalized.geoms)
    ordered = sorted(
        (orient(item, sign=1.0) for item in polygons),
        key=lambda item: (
            round(item.bounds[0], 3),
            round(item.bounds[1], 3),
            round(item.area, 3),
        ),
    )
    payload = cast(dict[str, Any], mapping(MultiPolygon(ordered)))

    def rounded(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, float) and round_coordinates:
            return round(value, 7)
        if isinstance(value, tuple | list):
            return [rounded(item) for item in value]
        return value

    return cast(dict[str, Any], rounded(payload))


def _cumulative_affected_geometry(
    affected: dict[str, Any] | None,
    active: dict[str, Any] | None,
    prior_affected: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Preserve the published WGS84 edges, without a second projection or rounding.

    A straight edge in WGS84 is not a straight edge in Lambert-93. Reprojecting
    the previous contour for a vector union and rounding its intersections can
    remove slivers even when the probability grid is perfectly cumulative.
    """
    parts = [shape(value) for value in (affected, active, prior_affected) if value is not None]
    if not parts:
        return None
    cumulative = _polygonal(unary_union(parts))
    if cumulative is None or cumulative.is_empty:
        raise ValueError("cumulative affected geometry is not polygonal")
    # Retain intersection coordinates exactly; no buffer or relaxed safety gate.
    return _canonical_wgs84_multipolygon(cumulative, round_coordinates=False)


def _polygonize_probability(
    probability: np.ndarray[Any, Any],
    *,
    transform: Affine,
    selected: np.ndarray[Any, Any],
) -> dict[str, Any] | None:
    if not bool(np.any(selected)):
        return None
    polygons: list[Polygon | MultiPolygon] = []
    encoded = selected.astype("uint8")
    for geometry, value in shapes(encoded, mask=selected, transform=transform):
        if int(value) != 1:
            continue
        candidate = _polygonal(shape(geometry))
        if candidate is not None and not candidate.is_empty:
            polygons.append(candidate)
    if not polygons:
        return None
    merged = _polygonal(unary_union(polygons))
    return None if merged is None else _canonical_multipolygon(merged)


def _ordered_observations(
    observations: tuple[SpatialObservationV2, ...],
) -> tuple[SpatialObservationV2, ...]:
    selected: dict[str, SpatialObservationV2] = {}
    for item in sorted(observations, key=lambda row: (row.observed_at, row.observation_id)):
        current = selected.get(item.observation_id)
        if current is not None and current != item:
            raise ValueError("spatial observation identifier maps to conflicting payloads")
        selected[item.observation_id] = item
    return tuple(sorted(selected.values(), key=lambda row: (row.observed_at, row.observation_id)))


def _empty_state(
    *,
    incident_id: str,
    episode_id: str | None,
    local_date: date,
    source_input_sha256: str,
    source_observation_ids: tuple[str, ...],
    source_family_ids: tuple[str, ...],
    contradiction_codes: tuple[str, ...],
    prior_state_sha256: str | None,
    profile: FusionProfileV1,
    algorithm_version: str = BASELINE_ALGORITHM_VERSION,
) -> FusedFireState:
    profile_identity = profile.identity()
    perimeter = PerimeterEstimateV3(
        incident_id=incident_id,
        local_date=local_date,
        status="insufficient",
        observed_fraction=0,
        fused_fraction=0,
        interpolated_fraction=0,
        observable_fraction=0,
        uncertainty_fraction=0,
        evidence_strength=0,
        calibration_state=profile.calibration_state,
        fusion_profile=profile_identity,
        source_family_ids=source_family_ids,
        contradiction_codes=contradiction_codes,
        evidence_refs=source_observation_ids,
    )
    state = DailyFireStateV2(
        state_id=f"DFS-{local_date.isoformat()}-{source_input_sha256[:24]}",
        incident_id=incident_id,
        episode_id=episode_id,
        local_date=local_date,
        status="insufficient",
        perimeter=perimeter,
        prior_state_sha256=prior_state_sha256,
        source_observation_ids=source_observation_ids,
        source_family_ids=source_family_ids,
        contradiction_codes=contradiction_codes,
        algorithm_id=FUSION_ALGORITHM_ID,
        algorithm_version=algorithm_version,
        fusion_profile=profile_identity,
        source_input_sha256=source_input_sha256,
    )
    return FusedFireState(state, None, None, None, None, None, None, None, None)


def regrid_array(
    values: np.ndarray[Any, Any] | None,
    source_transform: Affine,
    destination_transform: Affine,
    dimensions: tuple[int, int],
    resampling: Resampling,
    fill: float,
) -> np.ndarray[Any, Any]:
    if values is None:
        raise ValueError("prior_probability_grid_unavailable")
    if source_transform == destination_transform and values.shape == dimensions:
        return values.copy()
    output = np.full(dimensions, fill, dtype=np.float32)
    reproject(
        values,
        output,
        src_transform=source_transform,
        dst_transform=destination_transform,
        src_crs="EPSG:2154",
        dst_crs="EPSG:2154",
        dst_nodata=fill,
        resampling=resampling,
        init_dest_nodata=True,
        num_threads=1,
    )
    return output


def _ledger_metadata(group: list[SpatialObservationV2], previous: Any) -> dict[str, Any]:
    metadata = dict(previous or {})
    products = dict(metadata.get("products", {}))
    observations = dict(metadata.get("observations", {}))
    for item in group:
        revision = products.get(item.upstream_product_id)
        if revision is not None and revision != item.source_revision_sha256:
            raise ValueError("lineage_reprocessing_requires_revision")
        products[item.upstream_product_id] = item.source_revision_sha256
        identity = sha256_hex(item.model_dump(mode="json", by_alias=True))
        if item.observation_id in observations and observations[item.observation_id] != identity:
            raise ValueError("observation_revision_requires_replay")
        observations[item.observation_id] = identity
    return {
        "products": products,
        "observations": observations,
        "lineage_id": group[0].lineage_id,
        "kind": group[0].observation_kind,
        "target": group[0].target_state,
        "family": group[0].source_family_id,
    }


def fuse_probability_baseline(**kwargs: Any) -> FusedFireState:
    """Explicit historical 3.2.1 replay only; production must supply a spatial context."""
    return _fuse_probability_state(**kwargs, algorithm_version=BASELINE_ALGORITHM_VERSION)


def fuse_daily_fire_state(
    *,
    incident_id: str,
    episode_id: str | None,
    local_date: date,
    observations: tuple[SpatialObservationV2, ...],
    spatial_context: Part4SpatialContextV1 | None = None,
    prior_result: FusedFireState | None = None,
    profile: FusionProfileV1 | None = None,
) -> FusedFireState:
    from fireviewer_fire_state.part4_spatial_framing import reconstruct_framed_state

    if spatial_context is None:
        raise ValueError("awaiting_spatial_initialization")
    return reconstruct_framed_state(
        incident_id=incident_id,
        episode_id=episode_id,
        local_date=local_date,
        observations=observations,
        context=spatial_context,
        prior=prior_result,
        profile=profile
        or load_fusion_profile(
            DEFAULT_FUSION_PROFILE_ID, algorithm_version=FUSION_ALGORITHM_VERSION
        ),
    )


def _fuse_probability_state(
    *,
    incident_id: str,
    episode_id: str | None,
    local_date: date,
    observations: tuple[SpatialObservationV2, ...],
    prior_affected: dict[str, Any] | None = None,
    prior_active: dict[str, Any] | None = None,
    prior_observed_at: datetime | None = None,
    prior_state_sha256: str | None = None,
    profile: FusionProfileV1 | None = None,
    spatial_context: Part4SpatialContextV1 | None = None,
    prior_result: FusedFireState | None = None,
    algorithm_version: str = BASELINE_ALGORITHM_VERSION,
) -> FusedFireState:
    """Fuse one incident-day state without reading any published reference geometry."""

    selected = _ordered_observations(observations)
    effective_profile = profile or load_fusion_profile(
        "part4-baseline-v3-provenance" if spatial_context is None else DEFAULT_FUSION_PROFILE_ID,
        algorithm_version=algorithm_version,
    )
    if effective_profile.algorithm_id != FUSION_ALGORITHM_ID:
        raise ValueError("fusion profile algorithm identifier does not match Part.4")
    if algorithm_version not in effective_profile.algorithm_versions:
        raise ValueError("fusion profile does not support the active Part.4 algorithm")
    profile_payload = effective_profile.model_dump(mode="json", by_alias=True)
    source_payload = {
        "algorithm_id": FUSION_ALGORITHM_ID,
        "algorithm_version": algorithm_version,
        "incident_id": incident_id,
        "episode_id": episode_id,
        "local_date": local_date.isoformat(),
        "observations": [item.model_dump(mode="json", by_alias=True) for item in selected],
        "prior_affected": prior_affected,
        "prior_active": prior_active,
        "prior_observed_at": prior_observed_at,
        "prior_state_sha256": prior_state_sha256,
        "fusion_profile": profile_payload,
        "published_reference_accessed": False,
    }
    if spatial_context is not None:
        source_payload["spatial_context"] = spatial_context.model_dump(mode="json", by_alias=True)
    source_input_sha256 = sha256_hex(source_payload)
    observation_ids = tuple(item.observation_id for item in selected)
    family_ids = tuple(sorted({item.source_family_id for item in selected}))
    contradictions: list[str] = []
    projected: dict[str, Polygon | MultiPolygon] = {}
    projected_coverages: dict[str, Polygon | MultiPolygon] = {}
    for item in selected:
        geometry = _projected_observation_geometry(item)
        coverage = _projected_geojson(item.coverage_geojson)
        if coverage is not None:
            projected_coverages[item.observation_id] = coverage
        if geometry is None and item.raster_uri is not None:
            contradictions.append("raster_observation_not_materialized")
        elif geometry is None and coverage is None:
            contradictions.append("spatial_observation_geometry_invalid")
        elif geometry is not None:
            projected[item.observation_id] = geometry
    prior_affected_projected = _projected_geojson(prior_affected)
    prior_active_projected = _projected_geojson(prior_active)
    bounds_geometries = (
        [_projected_geojson(spatial_context.calculation_domain)]
        if spatial_context is not None
        else [*projected.values(), *projected_coverages.values()]
    )
    if prior_affected_projected is not None:
        bounds_geometries.append(prior_affected_projected)
    if prior_active_projected is not None:
        bounds_geometries.append(prior_active_projected)
    if not bounds_geometries:
        return _empty_state(
            incident_id=incident_id,
            episode_id=episode_id,
            local_date=local_date,
            source_input_sha256=source_input_sha256,
            source_observation_ids=observation_ids,
            source_family_ids=family_ids,
            contradiction_codes=tuple(sorted(set(contradictions))),
            prior_state_sha256=prior_state_sha256,
            profile=effective_profile,
        )
    # Bounding boxes do not require a topology union of detailed source shapes.
    boxes = [item.bounds for item in bounds_geometries if item is not None]
    merged_bounds = (
        min(item[0] for item in boxes),
        min(item[1] for item in boxes),
        max(item[2] for item in boxes),
        max(item[3] for item in boxes),
    )
    transform, width, height, resolution = _grid_for_bounds(
        merged_bounds,
        effective_profile,
        minimum_resolution=float(prior_result.transform.a)
        if prior_result is not None and prior_result.transform is not None
        else 0,
    )
    affected = np.full(
        (height, width), effective_profile.priors.background_probability, dtype=np.float32
    )
    active = np.full(
        (height, width), effective_profile.priors.background_probability, dtype=np.float32
    )
    observable = np.zeros((height, width), dtype=np.float32)
    current_support = np.zeros((height, width), dtype=bool)
    prior_support = np.zeros((height, width), dtype=bool)
    observed_support = np.zeros((height, width), dtype=np.float32)
    family_support: dict[str, np.ndarray[Any, Any]] = {}
    contribution_support: dict[str, np.ndarray[Any, Any]] = {}
    positive_active_by_family: dict[str, np.ndarray[Any, Any]] = {}
    valid_negative_by_family: dict[str, np.ndarray[Any, Any]] = {}
    new_direct_affected = np.zeros((height, width), dtype=bool)
    new_direct_active = np.zeros((height, width), dtype=bool)

    if prior_affected_projected is not None:
        prior_mask = _mask(
            prior_affected_projected,
            transform=transform,
            width=width,
            height=height,
        )
        affected[prior_mask] = effective_profile.priors.affected_probability
        prior_support |= prior_mask
        if spatial_context is not None and prior_result is None:
            uncertain_seed = boundary_uncertainty_mask(
                prior_affected_projected,
                spatial_context.seed.horizontal_accuracy_m,
                transform=transform,
                width=width,
                height=height,
            )
            # Administrative accuracy is uncertainty, not a newly burned halo.
            affected[uncertain_seed & ~prior_mask] = (
                effective_profile.uncertainty.halo_base_probability
            )
            affected[uncertain_seed & prior_mask] = effective_profile.thresholds.affected
            prior_support |= uncertain_seed
    if prior_active_projected is not None:
        prior_active_mask = _mask(
            prior_active_projected,
            transform=transform,
            width=width,
            height=height,
        )
        target_end = (
            spatial_context.state_valid_at
            if spatial_context
            else datetime.combine(local_date, time.max, tzinfo=_PARIS).astimezone(UTC)
        )
        source_time = (
            prior_observed_at.astimezone(UTC)
            if prior_observed_at is not None
            else target_end.replace(hour=0, minute=0, second=0, microsecond=0)
        )
        elapsed_hours = max(0.0, (target_end - source_time).total_seconds() / 3_600.0)
        decay = 0.5 ** (elapsed_hours / effective_profile.temporal.active_half_life_hours)
        active[prior_active_mask] = np.maximum(
            active[prior_active_mask],
            effective_profile.priors.active_probability * decay,
        )
        propagation_m = min(
            effective_profile.temporal.max_propagation_m,
            effective_profile.temporal.propagation_rate_m_per_hour * elapsed_hours,
        )
        if propagation_m > 0:
            propagated = _polygonal(prior_active_projected.buffer(propagation_m))
            if propagated is not None:
                propagated_mask = _mask(
                    propagated,
                    transform=transform,
                    width=width,
                    height=height,
                )
                active[propagated_mask & ~prior_active_mask] = np.maximum(
                    active[propagated_mask & ~prior_active_mask],
                    effective_profile.priors.propagated_active_probability,
                )
                prior_support |= propagated_mask

    target_end = (
        spatial_context.state_valid_at
        if spatial_context
        else datetime.combine(local_date, time.max, tzinfo=_PARIS).astimezone(UTC)
    )
    lineage_grids: dict[str, np.ndarray[Any, Any]] = {}
    lineage_metadata: dict[str, Any] = {}
    if prior_result is not None:
        if prior_result.transform is None or prior_result.state.state_valid_at is None:
            raise ValueError("prior_probability_grid_unavailable")
        affected = regrid_array(
            prior_result.affected_probability,
            prior_result.transform,
            transform,
            (height, width),
            Resampling.max,
            effective_profile.priors.background_probability,
        )
        restored_active = regrid_array(
            prior_result.active_probability,
            prior_result.transform,
            transform,
            (height, width),
            Resampling.average,
            effective_profile.priors.background_probability,
        )
        elapsed = (target_end - prior_result.state.state_valid_at).total_seconds() / 3600
        decay = 0.5 ** (max(0, elapsed) / effective_profile.temporal.active_half_life_hours)
        # Keep only the bounded propagation halo above; do not refill the old
        # active polygon at a uniform probability before restoring its state.
        if prior_active_projected is not None:
            active[prior_active_mask] = effective_profile.priors.background_probability
        active = np.maximum(active, restored_active * decay)
        prior_support |= affected >= effective_profile.thresholds.uncertainty_low
        lineage_grids = {
            key: regrid_array(
                values, prior_result.transform, transform, (height, width), Resampling.max, 0
            )
            for key, values in prior_result.lineage_grids.items()
        }
        lineage_metadata = json.loads(json.dumps(prior_result.lineage_metadata))

    lineage_groups: dict[tuple[str, str, str], list[SpatialObservationV2]] = {}
    for item in selected:
        lineage_groups.setdefault(
            (item.lineage_id, item.observation_kind, item.target_state), []
        ).append(item)

    # Negative evidence is applied only after every positive lineage. This
    # makes suppression independent from lexical lineage identifiers and keeps
    # the fusion deterministic for an identical observation set.
    for group_key in sorted(
        lineage_groups,
        key=lambda key: (
            key[1] == "valid_negative",
            min((item.observed_at, item.observation_id) for item in lineage_groups[key]),
            key,
        ),
    ):
        group = lineage_groups[group_key]
        late_complement = bool(
            (
                prior_result is not None
                and prior_result.state.state_valid_at is not None
                and max(item.observed_at for item in group) < prior_result.state.state_valid_at
            )
            or (
                spatial_context is not None
                and max(item.observed_at for item in group).astimezone(_PARIS).date() < local_date
            )
        )
        ledger_key = sha256_hex(list(group_key))
        old_probability = lineage_grids.get(ledger_key, np.zeros((height, width), dtype=np.float32))
        group_families = {item.source_family_id for item in group}
        if len(group_families) != 1:
            raise ValueError("one observation lineage cannot span multiple source families")
        family_id = next(iter(group_families))
        observation_kind = group[0].observation_kind
        target_state = group[0].target_state
        selected_mask = np.zeros((height, width), dtype=bool)
        # Float64 preserves the exact scalar arithmetic used by Part.4 3.0.1
        # when a lineage contains a single observation. The final probability
        # grids remain float32 and retain their canonical baseline hashes.
        selected_probability = np.zeros((height, width), dtype=np.float64)
        coverage_mask = np.zeros((height, width), dtype=bool)
        coverage_observability = np.zeros((height, width), dtype=np.float32)
        negative_probability = np.zeros((height, width), dtype=np.float64)
        for item in group:
            geometry = projected.get(item.observation_id)
            coverage = projected_coverages.get(item.observation_id)
            item_coverage_mask = (
                _mask(coverage, transform=transform, width=width, height=height)
                if coverage is not None
                else None
            )
            if item_coverage_mask is not None:
                observability = (
                    item.observability_probability
                    if item.observability_probability is not None
                    else item.probability
                )
                coverage_mask |= item_coverage_mask
                coverage_observability[item_coverage_mask] = np.maximum(
                    coverage_observability[item_coverage_mask], observability
                )
                negative_probability[item_coverage_mask] = np.maximum(
                    negative_probability[item_coverage_mask], item.probability
                )
            if geometry is None:
                continue
            item_mask = _mask(geometry, transform=transform, width=width, height=height)
            selected_mask |= item_mask
            selected_probability[item_mask] = np.maximum(
                selected_probability[item_mask], item.probability
            )
            uncertainty_m = max(
                item.horizontal_accuracy_m or 0,
                item.resolution_m or 0,
                resolution,
            )
            if (
                uncertainty_m > resolution
                and observation_kind != "valid_negative"
                and item.observation_id
                not in lineage_metadata.get(ledger_key, {}).get("observations", {})
            ):
                buffered = _polygonal(geometry.buffer(uncertainty_m))
                if buffered is not None:
                    halo = _mask(buffered, transform=transform, width=width, height=height)
                    halo &= ~item_mask
                    halo_probability = (
                        effective_profile.uncertainty.halo_base_probability
                        + item.probability * effective_profile.uncertainty.halo_probability_scale
                    )
                    if target_state in {"affected", "both"}:
                        affected[halo] = np.maximum(affected[halo], halo_probability)
                    if target_state in {"active", "both"}:
                        active[halo] = np.maximum(active[halo], halo_probability)

        contribution_mask = contribution_support.setdefault(
            family_id,
            np.zeros((height, width), dtype=bool),
        )
        if bool(np.any(coverage_mask)) and not late_complement:
            observable[coverage_mask] = np.maximum(
                observable[coverage_mask], coverage_observability[coverage_mask]
            )
            contribution_mask |= coverage_mask
        if observation_kind == "valid_negative":
            if not bool(np.any(coverage_mask)):
                raise ValueError("valid negative lineage has no observable coverage")
            sensor = effective_profile.sensors[observation_kind]
            reduction = np.maximum(
                effective_profile.valid_negative.minimum_reduction_factor,
                1.0 - sensor.profile_probability * negative_probability[coverage_mask],
            )
            old_reduction = np.maximum(
                effective_profile.valid_negative.minimum_reduction_factor,
                1.0 - sensor.profile_probability * old_probability[coverage_mask],
            )
            active[coverage_mask] = np.maximum(
                effective_profile.valid_negative.probability_floor,
                active[coverage_mask] * np.minimum(1, reduction / old_reduction),
            )
            lineage_grids[ledger_key] = np.maximum(old_probability, negative_probability).astype(
                np.float32
            )
            if spatial_context is not None:
                lineage_metadata[ledger_key] = _ledger_metadata(
                    group, lineage_metadata.get(ledger_key)
                )
            negative_mask = valid_negative_by_family.setdefault(
                family_id,
                np.zeros((height, width), dtype=bool),
            )
            negative_mask |= coverage_mask
            continue
        if not bool(np.any(selected_mask)):
            continue
        if not late_complement:
            current_support |= selected_mask
            observed_support[selected_mask] = np.maximum(
                observed_support[selected_mask],
                selected_probability[selected_mask].astype(np.float32),
            )
        else:
            prior_support |= selected_mask
        contribution_mask |= selected_mask
        if observation_kind not in _MULTI_SOURCE_INELIGIBLE_OBSERVATIONS and not late_complement:
            family_mask = family_support.setdefault(
                family_id,
                np.zeros((height, width), dtype=bool),
            )
            family_mask |= selected_mask
        if not bool(np.any(coverage_mask)) and not late_complement:
            observable[selected_mask] = np.maximum(
                observable[selected_mask], selected_probability[selected_mask]
            )
        sensor = effective_profile.sensors[observation_kind]
        new_probability = np.maximum(old_probability, selected_probability)
        lineage_grids[ledger_key] = new_probability.astype(np.float32)
        if spatial_context is not None:
            lineage_metadata[ledger_key] = _ledger_metadata(group, lineage_metadata.get(ledger_key))
        selected_mask &= selected_probability.astype(np.float32) > old_probability
        if observation_kind in _DIRECT_GEOMETRY_OBSERVATIONS:
            if target_state in {"affected", "both"}:
                new_direct_affected |= selected_mask
            if target_state in {"active", "both"} and observation_kind != "modelled_perimeter":
                new_direct_active |= selected_mask
                new_direct_affected |= selected_mask
        # The accumulator stores a per-cell maximum for the lineage. Apply
        # only the log-odds difference, including when a pass is completed later.
        if target_state in {"affected", "both"}:
            _apply_positive(
                affected,
                selected_mask,
                probabilities=selected_probability,
                profile_probability=sensor.profile_probability,
                weight=sensor.weight,
                direct_support_threshold=effective_profile.thresholds.affected,
                preserve_direct_support=(observation_kind in _DIRECT_GEOMETRY_OBSERVATIONS),
                previous_probabilities=old_probability,
            )
        if target_state in {"active", "both"}:
            active_before = active.copy() if late_complement else None
            _apply_positive(
                active,
                selected_mask,
                probabilities=selected_probability,
                profile_probability=sensor.profile_probability,
                weight=sensor.weight,
                direct_support_threshold=effective_profile.thresholds.active,
                preserve_direct_support=(observation_kind in _DIRECT_GEOMETRY_OBSERVATIONS),
                previous_probabilities=old_probability,
            )
            if active_before is not None:
                age_hours = (
                    target_end - max(item.observed_at for item in group)
                ).total_seconds() / 3600
                active = np.maximum(
                    active_before,
                    active
                    * (
                        0.5
                        ** (max(0, age_hours) / effective_profile.temporal.active_half_life_hours)
                    ),
                )
            positive_mask = positive_active_by_family.setdefault(
                family_id,
                np.zeros((height, width), dtype=bool),
            )
            positive_mask |= selected_mask
            if observation_kind != "thermal_footprint":
                transfer = effective_profile.active_to_affected
                _apply_positive(
                    affected,
                    selected_mask,
                    probabilities=selected_probability,
                    profile_probability=transfer.profile_probability,
                    weight=transfer.weight,
                    previous_probabilities=old_probability,
                )

    independent_contradiction = any(
        positive_family != negative_family and bool(np.any(positive_mask & negative_mask))
        for positive_family, positive_mask in positive_active_by_family.items()
        for negative_family, negative_mask in valid_negative_by_family.items()
    )
    if independent_contradiction:
        contradictions.append("independent_active_and_valid_negative_overlap")
    if spatial_context is not None:
        direct_active = new_direct_active.copy()
        direct_affected = new_direct_affected.copy()
        if prior_affected_projected is not None:
            direct_affected |= _mask(
                prior_affected_projected, transform=transform, width=width, height=height
            )
        if prior_active_projected is not None:
            direct_active |= _mask(
                prior_active_projected, transform=transform, width=width, height=height
            )
        # Kernels and priors remain uncertainty, never exact burned/front polygons.
        active[~direct_active] = np.minimum(
            active[~direct_active],
            np.nextafter(np.float32(effective_profile.thresholds.active), np.float32(0)),
        )
        affected[~direct_affected] = np.minimum(
            affected[~direct_affected],
            np.nextafter(np.float32(effective_profile.thresholds.affected), np.float32(0)),
        )
    active_selected = active >= effective_profile.thresholds.active
    affected[active_selected] = np.maximum(affected[active_selected], active[active_selected])
    affected_selected = affected >= effective_profile.thresholds.affected
    uncertainty_selected = (
        np.maximum(affected, active) >= effective_profile.thresholds.uncertainty_low
    ) & (np.maximum(affected, active) < effective_profile.thresholds.uncertainty_high)
    affected_geojson = _polygonize_probability(
        affected,
        transform=transform,
        selected=affected_selected,
    )
    active_geojson = _polygonize_probability(active, transform=transform, selected=active_selected)
    uncertainty_geojson = _polygonize_probability(
        np.maximum(affected, active),
        transform=transform,
        selected=uncertainty_selected,
    )
    affected_geojson = _cumulative_affected_geometry(
        affected_geojson, active_geojson, prior_affected
    )
    if affected_geojson is None:
        quality: DailyFireQuality = "insufficient"
        active_geojson = None
    elif not bool(np.any(current_support)):
        quality = "interpolated"
    elif prior_affected_projected is not None or len(family_support) > 1:
        quality = "fused"
    else:
        quality = "observed"
    grid_cells = width * height
    observed_fraction = float(np.count_nonzero(observable > 0) / grid_cells)
    result_mask = affected_selected | active_selected | uncertainty_selected
    result_count = int(np.count_nonzero(result_mask))
    family_count = np.zeros((height, width), dtype=np.uint16)
    for supported in family_support.values():
        family_count += supported.astype(np.uint16)
    multi_source_support = ((family_count >= 2) & result_mask).astype(np.float32)
    prior_interpolated_support = (prior_support & ~current_support & result_mask).astype(np.float32)
    combined_probability = np.maximum(affected, active)
    uncertainty_support = np.where(
        uncertainty_selected,
        combined_probability,
        0.0,
    ).astype(np.float32)
    fused_fraction = (
        float(np.count_nonzero((family_count >= 2) & result_mask) / result_count)
        if result_count
        else 0.0
    )
    interpolated_fraction = (
        float(np.count_nonzero(prior_support & ~current_support & result_mask) / result_count)
        if result_count
        else 0.0
    )
    uncertainty_fraction = (
        float(np.count_nonzero(uncertainty_selected & result_mask) / result_count)
        if result_count
        else 0.0
    )
    evidence_strength = (
        float(np.mean(np.maximum(affected, active)[result_mask])) if result_count else 0.0
    )
    times = [item.observed_at for item in selected]
    if prior_result is not None and prior_result.state.latest_observation_at is not None:
        times.append(prior_result.state.latest_observation_at)
    latest = max(times, default=None)
    latest_age = (
        max(0, int((target_end - latest.astimezone(UTC)).total_seconds()))
        if latest is not None
        else None
    )
    contradiction_codes = tuple(sorted(set(contradictions)))
    contributions: list[SensorContributionV1] = []
    for family_id in family_ids:
        family_observations = tuple(item for item in selected if item.source_family_id == family_id)
        support_mask = contribution_support.get(family_id)
        contributions.append(
            SensorContributionV1(
                source_family_id=family_id,
                observation_ids=tuple(item.observation_id for item in family_observations),
                observation_kinds=tuple(
                    sorted({item.observation_kind for item in family_observations})
                ),
                lineage_count=len({item.lineage_id for item in family_observations}),
                supported_fraction=(
                    round(float(np.count_nonzero(support_mask) / grid_cells), 6)
                    if support_mask is not None
                    else 0.0
                ),
                maximum_raw_probability=max(item.probability for item in family_observations),
                maximum_observability_probability=max(
                    (
                        item.observability_probability
                        if item.observability_probability is not None
                        else item.probability
                    )
                    for item in family_observations
                ),
                calibration_state=effective_profile.calibration_state,
            )
        )
    evidence_refs = tuple(
        sorted(
            {reference for item in selected for reference in item.evidence_refs}
            | set(observation_ids)
        )
    )
    perimeter = PerimeterEstimateV3(
        incident_id=incident_id,
        local_date=local_date,
        status=quality,
        affected=affected_geojson,
        active=active_geojson,
        uncertainty_band=uncertainty_geojson,
        resolution_m=resolution,
        observed_fraction=round(observed_fraction, 6),
        fused_fraction=round(fused_fraction, 6),
        interpolated_fraction=round(interpolated_fraction, 6),
        observable_fraction=round(observed_fraction, 6),
        uncertainty_fraction=round(uncertainty_fraction, 6),
        evidence_strength=round(evidence_strength, 6),
        calibration_state=effective_profile.calibration_state,
        fusion_profile=effective_profile.identity(),
        source_family_ids=family_ids,
        contradiction_codes=contradiction_codes,
        evidence_refs=evidence_refs,
        eligible_for_automatic_publication=False,
        needs_human_review=True,
    )
    state = DailyFireStateV2(
        state_id=f"DFS-{local_date.isoformat()}-{source_input_sha256[:24]}",
        incident_id=incident_id,
        episode_id=episode_id,
        local_date=local_date,
        status=quality,
        perimeter=perimeter,
        prior_state_sha256=prior_state_sha256,
        latest_observation_at=latest,
        latest_observation_age_seconds=latest_age,
        source_observation_ids=observation_ids,
        source_family_ids=family_ids,
        contributions=tuple(contributions),
        contradiction_codes=contradiction_codes,
        algorithm_id=FUSION_ALGORITHM_ID,
        algorithm_version=algorithm_version,
        fusion_profile=effective_profile.identity(),
        source_input_sha256=source_input_sha256,
        spatial_context=spatial_context,
        state_valid_at=target_end if spatial_context is not None else None,
    )
    return FusedFireState(
        state=state,
        affected_probability=affected,
        active_probability=active,
        observable_probability=observable,
        observed_support=observed_support,
        multi_source_support=multi_source_support,
        prior_interpolated_support=prior_interpolated_support,
        uncertainty_support=uncertainty_support,
        transform=transform,
        lineage_grids=lineage_grids,
        lineage_metadata=lineage_metadata,
    )


def _write_cog(result: FusedFireState, destination: Path) -> None:
    if (
        result.affected_probability is None
        or result.active_probability is None
        or result.observable_probability is None
        or result.transform is None
    ):
        raise ValueError("an insufficient fire state has no probability raster")
    profile = result.state.fusion_profile
    if profile is None:
        raise ValueError("Part.4 3.1 probability grids require a fusion profile identity")
    temporary = destination.with_suffix(".source.tif")
    arrays = (
        result.affected_probability,
        result.active_probability,
        result.observable_probability,
    )
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        width=arrays[0].shape[1],
        height=arrays[0].shape[0],
        count=3,
        dtype="float32",
        crs="EPSG:2154",
        transform=result.transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        predictor=3,
        nodata=-1.0,
    ) as dataset:
        for index, (name, values) in enumerate(
            zip(("affected", "active", "observable"), arrays, strict=True),
            start=1,
        ):
            dataset.write(values.astype("float32"), index)
            dataset.set_band_description(index, name)
        dataset.update_tags(
            source_input_sha256=result.state.source_input_sha256,
            algorithm_id=result.state.algorithm_id,
            algorithm_version=result.state.algorithm_version,
            calibration_state=profile.calibration_state,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            profile_sha256=profile.profile_sha256,
            published_reference_accessed="false",
            **_framing_tags(result.state),
        )
    raster_copy(
        temporary,
        destination,
        driver="COG",
        compress="DEFLATE",
        blocksize=256,
        overview_resampling="average",
    )
    temporary.unlink(missing_ok=True)


def _write_spatial_provenance_cog(result: FusedFireState, destination: Path) -> None:
    arrays = (
        result.observed_support,
        result.multi_source_support,
        result.prior_interpolated_support,
        result.uncertainty_support,
    )
    if any(item is None for item in arrays) or result.transform is None:
        raise ValueError("Part.4 3.2 requires four aligned spatial provenance arrays")
    profile = result.state.fusion_profile
    if profile is None:
        raise ValueError("Part.4 3.2 spatial provenance requires a fusion profile identity")
    concrete = cast(tuple[np.ndarray[Any, Any], ...], arrays)
    temporary = destination.with_suffix(".source.tif")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        width=concrete[0].shape[1],
        height=concrete[0].shape[0],
        count=4,
        dtype="float32",
        crs="EPSG:2154",
        transform=result.transform,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        compress="DEFLATE",
        predictor=3,
        nodata=-1.0,
    ) as dataset:
        for index, (name, values) in enumerate(
            zip(
                (
                    "observed_support",
                    "multi_source_support",
                    "prior_interpolated_support",
                    "uncertainty_support",
                ),
                concrete,
                strict=True,
            ),
            start=1,
        ):
            dataset.write(values.astype("float32"), index)
            dataset.set_band_description(index, name)
        dataset.update_tags(
            source_input_sha256=result.state.source_input_sha256,
            algorithm_id=result.state.algorithm_id,
            algorithm_version=result.state.algorithm_version,
            artifact_kind="spatial_provenance",
            provenance_encoding="support_probability_and_masks_v1",
            calibration_state=profile.calibration_state,
            profile_id=profile.profile_id,
            profile_version=profile.profile_version,
            profile_sha256=profile.profile_sha256,
            published_reference_accessed="false",
            **_framing_tags(result.state),
        )
    raster_copy(
        temporary,
        destination,
        driver="COG",
        compress="DEFLATE",
        blocksize=256,
        overview_resampling="nearest",
    )
    temporary.unlink(missing_ok=True)


def _framing_tags(state: DailyFireStateV2) -> dict[str, str]:
    if state.framing is None:
        return {}
    return {
        "context_sha256": state.framing.context_sha256,
        "seed_id": state.framing.seed_id,
        "parent_state_sha256": state.prior_state_sha256 or "none",
        "state_valid_at": state.state_valid_at.isoformat() if state.state_valid_at else "none",
    }


def persist_probability_grid(
    result: FusedFireState,
    *,
    object_store: ObjectStore,
) -> FusedFireState:
    if result.transform is None or result.affected_probability is None:
        return result
    if any(
        item is None
        for item in (
            result.observed_support,
            result.multi_source_support,
            result.prior_interpolated_support,
            result.uncertainty_support,
        )
    ):
        raise ValueError("Part.4 3.2 cannot persist without spatial provenance arrays")
    state = result.state
    profile = state.fusion_profile
    if profile is None:
        raise ValueError("Part.4 3.1 probability grids require a fusion profile identity")
    key = (
        f"incident-fire-states/{state.incident_id}/{state.local_date.isoformat()}/"
        f"{state.source_input_sha256}"
    )
    cog_uri = object_store.uri_for(f"{key}/fire-state.tif")
    provenance_cog_uri = object_store.uri_for(f"{key}/spatial-provenance.tif")
    manifest_uri = object_store.uri_for(f"{key}/manifest.json")
    lineage_uri = object_store.uri_for(f"{key}/lineage.npz")
    framed = state.algorithm_version.startswith("3.3.")
    if framed and (state.framing is None or state.spatial_context is None):
        raise ValueError("framed_state_receipt_required")
    manifest: dict[str, Any] | None = None
    try:
        manifest = json.loads(object_store.read_bytes(manifest_uri))
    except (ObjectStorageError, json.JSONDecodeError, UnicodeDecodeError):
        manifest = None
    if manifest is not None:
        if (
            manifest.get("source_input_sha256") != state.source_input_sha256
            or manifest.get("profile_id") != profile.profile_id
            or manifest.get("profile_version") != profile.profile_version
            or manifest.get("profile_sha256") != profile.profile_sha256
            or manifest.get("calibration_state") != profile.calibration_state
            or manifest.get("schema")
            != (
                "fireviewer.fire-probability-grid-manifest.v3"
                if framed
                else "fireviewer.fire-probability-grid-manifest.v2"
            )
            or manifest.get("band_mapping") != {"affected": 1, "active": 2, "observable": 3}
            or manifest.get("provenance_band_mapping")
            != {
                "observed_support": 1,
                "multi_source_support": 2,
                "prior_interpolated_support": 3,
                "uncertainty_support": 4,
            }
            or manifest.get("provenance_encoding") != "support_probability_and_masks_v1"
            or manifest.get("published_reference_accessed") is not False
        ):
            raise ObjectStorageError("Stored fire-state manifest has an unexpected input hash.")
        metadata = object_store.head(cog_uri)
        provenance_metadata = object_store.head(provenance_cog_uri)
        if metadata.size_bytes != manifest.get(
            "cog_byte_count"
        ) or provenance_metadata.size_bytes != manifest.get("provenance_cog_byte_count"):
            raise ObjectStorageError(
                "Stored Part.4 raster failed its immutable hash/size manifest check."
            )
        if (
            manifest.get("cog_sha256")
            != hashlib.sha256(object_store.read_bytes(cog_uri)).hexdigest()
        ):
            raise ObjectStorageError("Stored fire-state COG failed its immutable hash check.")
        if (
            manifest.get("provenance_cog_sha256")
            != hashlib.sha256(object_store.read_bytes(provenance_cog_uri)).hexdigest()
        ):
            raise ObjectStorageError(
                "Stored spatial-provenance COG failed its immutable hash check."
            )
        if framed:
            assert state.framing is not None
            ledger = object_store.read_bytes(lineage_uri)
            if (
                len(ledger) != manifest.get("lineage_byte_count")
                or hashlib.sha256(ledger).hexdigest() != manifest.get("lineage_sha256")
                or manifest.get("context_sha256") != state.framing.context_sha256
            ):
                raise ObjectStorageError("Stored lineage/context identity mismatch.")
    else:
        with tempfile.TemporaryDirectory(prefix="fireviewer-fire-state-") as raw_directory:
            directory = Path(raw_directory)
            cog_path = directory / "fire-state.tif"
            provenance_cog_path = directory / "spatial-provenance.tif"
            _write_cog(result, cog_path)
            _write_spatial_provenance_cog(result, provenance_cog_path)
            cog_bytes = cog_path.read_bytes()
            provenance_cog_bytes = provenance_cog_path.read_bytes()
            lineage_bytes = b""
            if framed:
                if (
                    sum(values.nbytes for values in result.lineage_grids.values())
                    > 256 * 1024 * 1024
                ):
                    raise ObjectStorageError("Per-incident lineage scratch budget exceeded.")
                metadata_bytes = json.dumps(result.lineage_metadata, sort_keys=True).encode("utf-8")
                np.savez_compressed(
                    directory / "lineage.npz",
                    **cast(dict[str, Any], result.lineage_grids),
                    metadata=np.frombuffer(metadata_bytes, dtype=np.uint8),
                )
                lineage_bytes = (directory / "lineage.npz").read_bytes()
            manifest_payload: dict[str, Any] = {
                "schema": (
                    "fireviewer.fire-probability-grid-manifest.v3"
                    if framed
                    else "fireviewer.fire-probability-grid-manifest.v2"
                ),
                "incident_id": state.incident_id,
                "local_date": state.local_date.isoformat(),
                "source_input_sha256": state.source_input_sha256,
                "profile_id": profile.profile_id,
                "profile_version": profile.profile_version,
                "profile_sha256": profile.profile_sha256,
                "calibration_state": profile.calibration_state,
                "cog_sha256": hashlib.sha256(cog_bytes).hexdigest(),
                "cog_byte_count": len(cog_bytes),
                "provenance_cog_sha256": hashlib.sha256(provenance_cog_bytes).hexdigest(),
                "provenance_cog_byte_count": len(provenance_cog_bytes),
                "band_mapping": {"affected": 1, "active": 2, "observable": 3},
                "provenance_band_mapping": {
                    "observed_support": 1,
                    "multi_source_support": 2,
                    "prior_interpolated_support": 3,
                    "uncertainty_support": 4,
                },
                "provenance_encoding": "support_probability_and_masks_v1",
                "published_reference_accessed": False,
            }
            if framed:
                assert state.framing is not None
                manifest_payload.update(
                    {
                        "context_sha256": state.framing.context_sha256,
                        "seed_id": state.framing.seed_id,
                        "parent_state_sha256": state.prior_state_sha256,
                        "state_valid_at": state.state_valid_at.isoformat()
                        if state.state_valid_at
                        else None,
                        "lineage_sha256": hashlib.sha256(lineage_bytes).hexdigest(),
                        "lineage_byte_count": len(lineage_bytes),
                    }
                )
            (directory / "manifest.json").write_text(
                json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            object_store.finalize_tree(directory, key)
            manifest = manifest_payload
        metadata = object_store.head(cog_uri)
        provenance_metadata = object_store.head(provenance_cog_uri)
    assert manifest is not None
    transform = result.transform
    grid = FireProbabilityGridReference(
        cog_uri=cog_uri,
        sha256=str(manifest["cog_sha256"]),
        byte_count=metadata.size_bytes,
        width=result.affected_probability.shape[1],
        height=result.affected_probability.shape[0],
        resolution_m=float(transform.a),
        transform=(
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ),
        band_mapping={"affected": 1, "active": 2, "observable": 3},
    )
    provenance_grid = FireSpatialProvenanceGridReference(
        cog_uri=provenance_cog_uri,
        sha256=str(manifest["provenance_cog_sha256"]),
        byte_count=provenance_metadata.size_bytes,
        width=result.affected_probability.shape[1],
        height=result.affected_probability.shape[0],
        resolution_m=float(transform.a),
        transform=(
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ),
        band_mapping={
            "observed_support": 1,
            "multi_source_support": 2,
            "prior_interpolated_support": 3,
            "uncertainty_support": 4,
        },
    )
    lineage_artifact = (
        FramingArtifactReference(
            uri=lineage_uri,
            sha256=str(manifest["lineage_sha256"]),
            byte_count=int(manifest["lineage_byte_count"]),
        )
        if framed
        else None
    )
    return replace(
        result,
        state=state.model_copy(
            update={
                "grid": grid,
                "provenance_grid": provenance_grid,
                "lineage_artifact": lineage_artifact,
            }
        ),
    )


def restore_probability_state(
    state: DailyFireStateV2, *, object_store: ObjectStore
) -> FusedFireState:
    """Restore full immutable state, not uniform probabilities inferred from polygons."""
    import io
    import zipfile

    from rasterio.io import MemoryFile  # type: ignore[import-untyped]

    if (
        state.grid is None
        or state.provenance_grid is None
        or state.lineage_artifact is None
        or state.spatial_context is None
        or state.framing is None
        or state.fusion_profile is None
    ):
        raise ValueError("prior_requires_seeded_reconstruction")
    manifest_uri = state.grid.cog_uri.rsplit("/", 1)[0] + "/manifest.json"
    manifest = json.loads(object_store.read_bytes(manifest_uri))
    if (
        manifest.get("schema") != "fireviewer.fire-probability-grid-manifest.v3"
        or manifest.get("source_input_sha256") != state.source_input_sha256
        or manifest.get("context_sha256") != state.framing.context_sha256
        or manifest.get("profile_sha256") != state.fusion_profile.profile_sha256
        or manifest.get("parent_state_sha256") != state.prior_state_sha256
    ):
        raise ObjectStorageError("Prior manifest identity mismatch.")
    arrays: list[np.ndarray[Any, Any]] = []
    for reference, count in ((state.grid, 3), (state.provenance_grid, 4)):
        if reference.byte_count > 256 * 1024 * 1024:
            raise ObjectStorageError("Prior COG exceeds the per-incident byte budget.")
        content = object_store.read_bytes(reference.cog_uri)
        if (
            len(content) != reference.byte_count
            or hashlib.sha256(content).hexdigest() != reference.sha256
        ):
            raise ObjectStorageError("Prior COG identity mismatch.")
        with MemoryFile(content) as memory, memory.open() as dataset:
            if (
                dataset.count != count
                or dataset.width != reference.width
                or dataset.height != reference.height
                or dataset.crs.to_epsg() != 2154
                or tuple(dataset.transform)[:6] != reference.transform
                or dataset.tags().get("source_input_sha256") != state.source_input_sha256
                or any(
                    dataset.tags().get(key) != value for key, value in _framing_tags(state).items()
                )
            ):
                raise ObjectStorageError("Prior COG grid mismatch.")
            loaded = dataset.read()
            if not np.all(np.isfinite(loaded)) or np.any((loaded < 0) | (loaded > 1)):
                raise ObjectStorageError("Prior COG probabilities are invalid.")
            arrays.extend(loaded)
    artifact = state.lineage_artifact
    if artifact.byte_count > 256 * 1024 * 1024:
        raise ObjectStorageError("Prior lineage artifact is too large.")
    content = object_store.read_bytes(artifact.uri)
    if (
        len(content) != artifact.byte_count
        or hashlib.sha256(content).hexdigest() != artifact.sha256
    ):
        raise ObjectStorageError("Prior lineage identity mismatch.")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if sum(item.file_size for item in archive.infolist()) > 256 * 1024 * 1024:
            raise ObjectStorageError("Prior lineage expansion exceeds the scratch budget.")
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        metadata = json.loads(data["metadata"].tobytes())
        grids = {key: data[key].copy() for key in data.files if key != "metadata"}
    if set(grids) != set(metadata) or any(
        values.shape != arrays[0].shape
        or not np.all(np.isfinite(values))
        or np.any((values < 0) | (values > 1))
        for values in grids.values()
    ):
        raise ObjectStorageError("Prior lineage grids are not aligned.")
    return FusedFireState(
        state,
        arrays[0],
        arrays[1],
        arrays[2],
        arrays[3],
        arrays[4],
        arrays[5],
        arrays[6],
        Affine(*state.grid.transform),
        grids,
        metadata,
    )


__all__ = [
    "FUSION_ALGORITHM_ID",
    "FUSION_ALGORITHM_VERSION",
    "FusedFireState",
    "fuse_daily_fire_state",
    "persist_probability_grid",
]
