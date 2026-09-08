"""Advisory comparison of a private activity-zone draft with same-day official geometry."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Literal

import numpy as np
from pyproj import Transformer
from shapely import STRtree, distance, get_num_coordinates, points
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import transform, unary_union

_WGS84_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)


@dataclass(frozen=True, slots=True)
class ActivityZoneComparison:
    local_date: date
    assessment: Literal["coherent", "a_revoir"]
    intersection_over_union: float
    predicted_area_ha: float
    official_area_ha: float
    predicted_covered_percent: float
    official_covered_percent: float
    centroid_distance_m: float
    surface_bias_percent: float
    boundary_f1: float
    boundary_tolerance_m: float
    hausdorff95_m: float
    advisory_only: Literal[True] = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _polygonal_geometry(payload: dict[str, Any]) -> MultiPolygon:
    payload_type = payload.get("type")
    if payload_type == "FeatureCollection":
        geometries = [
            shape(feature["geometry"])
            for feature in payload.get("features", [])
            if isinstance(feature, dict) and isinstance(feature.get("geometry"), dict)
        ]
        geometry = unary_union(
            [item for item in geometries if isinstance(item, Polygon | MultiPolygon)]
        )
    elif payload_type == "Feature":
        geometry_payload = payload.get("geometry")
        if not isinstance(geometry_payload, dict):
            raise ValueError("GeoJSON feature has no geometry")
        geometry = shape(geometry_payload)
    else:
        geometry = shape(payload)
    if isinstance(geometry, Polygon):
        geometry = MultiPolygon([geometry])
    if not isinstance(geometry, MultiPolygon) or geometry.is_empty or not geometry.is_valid:
        raise ValueError("activity-zone comparison requires a valid Polygon or MultiPolygon")
    return geometry


def _boundary_chunks(boundary: Any) -> list[LineString]:
    lines = [boundary] if isinstance(boundary, LineString) else list(boundary.geoms)
    chunks: list[LineString] = []
    for line in lines:
        coordinates = np.asarray(line.coords)
        chunks.extend(
            LineString(coordinates[start : start + 1025])
            for start in range(0, len(coordinates) - 1, 1024)
        )
    return chunks


def _boundary_support_length(source: Any, target: Any, tolerance_m: float) -> float:
    if get_num_coordinates(target) <= 50_000:
        return float(source.intersection(target.buffer(tolerance_m)).length)
    # Subtract each band from the remaining source instead of summing/unioning
    # clipped pieces. At projected-coordinate magnitudes, intersection endpoints
    # can round onto almost coincident lines; union then double-counts overlap.
    # S \ (B1 union B2) == (S \ B1) \ B2 keeps each source portion counted once.
    chunks = _boundary_chunks(target)
    source_length = float(source.length)
    remaining = source
    for start in range(0, len(chunks), 128):
        if remaining.is_empty:
            break
        band = unary_union([chunk.buffer(tolerance_m) for chunk in chunks[start : start + 128]])
        remaining = remaining.difference(band)
    return max(0.0, source_length - float(remaining.length))


def _directed_boundary_distances(source: Any, target: Any, tolerance_m: float) -> list[float]:
    length = float(source.length)
    spacing = max(20.0, min(tolerance_m / 2, 250.0))
    count = max(2, min(20_000, math.ceil(length / spacing)))
    if max(get_num_coordinates(source), get_num_coordinates(target)) <= 50_000:
        return [
            float(source.interpolate(length * index / (count - 1)).distance(target))
            for index in range(count)
        ]
    # Same arc-length sample locations, without scanning every segment for every
    # point. STRtree changes nearest-distance lookup cost, not the metric.
    chunks = _boundary_chunks(source)
    coordinates = [np.asarray(chunk.coords) for chunk in chunks]
    starts = np.concatenate([value[:-1] for value in coordinates])
    vectors = np.concatenate([value[1:] - value[:-1] for value in coordinates])
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    valid = lengths > 0
    starts, vectors, lengths = starts[valid], vectors[valid], lengths[valid]
    cumulative = np.cumsum(lengths)
    offsets = np.linspace(0, cumulative[-1], count)
    indices = np.minimum(np.searchsorted(cumulative, offsets, side="right"), len(lengths) - 1)
    before = np.concatenate(([0.0], cumulative[:-1]))[indices]
    fraction = (offsets - before) / lengths[indices]
    samples = points(starts[indices] + fraction[:, None] * vectors[indices])
    tree = STRtree(_boundary_chunks(target))
    nearest = tree.nearest(samples)
    return [float(value) for value in distance(samples, tree.geometries.take(nearest))]


def compare_activity_zones(
    predicted_geojson: dict[str, Any],
    official_geojson: dict[str, Any],
    *,
    predicted_local_date: date,
    official_local_date: date,
    boundary_tolerance_m: float = 100.0,
) -> ActivityZoneComparison:
    """Measure plausibility only; this result can never approve or publish a layer."""

    if predicted_local_date != official_local_date:
        raise ValueError("predicted and official activity zones must describe the same local date")
    if not math.isfinite(boundary_tolerance_m) or boundary_tolerance_m <= 0:
        raise ValueError("boundary tolerance must be a positive finite distance")
    predicted = transform(_WGS84_TO_L93.transform, _polygonal_geometry(predicted_geojson))
    official = transform(_WGS84_TO_L93.transform, _polygonal_geometry(official_geojson))
    intersection_area = predicted.intersection(official).area
    predicted_area = predicted.area
    official_area = official.area
    union_area = predicted_area + official_area - intersection_area
    iou = intersection_area / union_area if union_area else 0.0
    predicted_covered = intersection_area / predicted_area if predicted_area else 0.0
    official_covered = intersection_area / official_area if official_area else 0.0
    centroid_distance = predicted.centroid.distance(official.centroid)
    area_ratio = predicted_area / official_area if official_area else float("inf")
    surface_bias = (
        ((predicted_area - official_area) / official_area) * 100 if official_area else float("inf")
    )
    predicted_boundary = predicted.boundary
    official_boundary = official.boundary
    boundary_precision = (
        _boundary_support_length(predicted_boundary, official_boundary, boundary_tolerance_m)
        / predicted_boundary.length
        if predicted_boundary.length
        else 0.0
    )
    boundary_recall = (
        _boundary_support_length(official_boundary, predicted_boundary, boundary_tolerance_m)
        / official_boundary.length
        if official_boundary.length
        else 0.0
    )
    if any(
        not math.isfinite(fraction) or not -1e-9 <= fraction <= 1.0 + 1e-9
        for fraction in (boundary_precision, boundary_recall)
    ):
        raise ValueError("boundary support fraction is outside [0, 1]")
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )

    distances = [
        *_directed_boundary_distances(predicted_boundary, official_boundary, boundary_tolerance_m),
        *_directed_boundary_distances(official_boundary, predicted_boundary, boundary_tolerance_m),
    ]
    distances.sort()
    hausdorff95 = distances[min(len(distances) - 1, math.ceil(len(distances) * 0.95) - 1)]
    coherent = (
        centroid_distance <= 2_000
        and 0.05 <= area_ratio <= 20
        and (iou >= 0.15 or predicted_covered >= 0.2 or official_covered >= 0.2)
    )
    return ActivityZoneComparison(
        local_date=predicted_local_date,
        assessment="coherent" if coherent else "a_revoir",
        intersection_over_union=round(iou, 6),
        predicted_area_ha=round(predicted_area / 10_000, 3),
        official_area_ha=round(official_area / 10_000, 3),
        predicted_covered_percent=round(predicted_covered * 100, 2),
        official_covered_percent=round(official_covered * 100, 2),
        centroid_distance_m=round(centroid_distance, 1),
        surface_bias_percent=round(surface_bias, 2),
        boundary_f1=round(boundary_f1, 6),
        boundary_tolerance_m=round(boundary_tolerance_m, 1),
        hausdorff95_m=round(hausdorff95, 1),
    )
