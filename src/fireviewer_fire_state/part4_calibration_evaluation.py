"""Reference-opening half of the isolated Part.4 calibration workflow."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pyproj import Transformer
from shapely import make_valid
from shapely.errors import GeometryTypeError
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import transform, unary_union

from fireviewer_contracts.backend.fire_state_schemas import DailyFireStateV2
from fireviewer_contracts.backend.hashing import sha256_hex
from fireviewer_contracts.backend.part4_calibration_schemas import Part4ReferenceSnapshotV1
from fireviewer_fire_state.activity_zone_quality import compare_activity_zones

_TO_L93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
_TO_WGS84 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)


def _load_state(path: Path) -> DailyFireStateV2:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DailyFireStateV2.model_validate(payload)


def _geometry_valid(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    try:
        geometry = shape(payload)
    except (TypeError, ValueError, GeometryTypeError):
        return False
    return (
        isinstance(geometry, Polygon | MultiPolygon)
        and not geometry.is_empty
        and geometry.is_valid
    )


def _polygonal_result(value: Any) -> Polygon | MultiPolygon | None:
    if (
        isinstance(value, Polygon | MultiPolygon)
        and not value.is_empty
        and value.is_valid
    ):
        return value
    candidate = make_valid(value)
    if isinstance(candidate, Polygon | MultiPolygon) and not candidate.is_empty:
        return candidate
    if isinstance(candidate, GeometryCollection):
        polygons = [
            item for item in candidate.geoms if isinstance(item, Polygon | MultiPolygon)
        ]
        merged = make_valid(unary_union(polygons))
        if isinstance(merged, Polygon | MultiPolygon) and not merged.is_empty:
            return merged
    return None


def _grid_features(path: Path | None) -> tuple[float, float]:
    if path is None:
        return 0.0, 0.0
    if path.name.endswith(".support.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != (
            "fireviewer.part4-frozen-grid-support.v1"
        ):
            raise ValueError("frozen grid support receipt is invalid")
        return (
            float(payload.get("observable_fraction", 0.0)),
            float(payload.get("uncertainty_fraction", 0.0)),
        )
    with np.load(path, allow_pickle=False) as payload:
        observable = np.asarray(payload["observable"], dtype=np.float64)
        uncertainty = np.asarray(payload["uncertainty_support"], dtype=np.float64)
    return (
        float(np.mean(np.clip(observable, 0.0, 1.0))),
        float(np.mean(uncertainty > 0)),
    )


def _source_resolution(path: Path | None) -> float | None:
    """Legacy grids do not prove sensor resolution; neither does the reference."""
    if path is None or not path.name.endswith(".support.json"):
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("best_observation_resolution_m")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("observation resolution receipt is invalid")
    if not math.isfinite(value) or not 0 < value <= 100_000:
        raise ValueError("observation resolution receipt is invalid")
    return float(value)


def _metric_float(values: Mapping[str, object], key: str) -> float:
    value = values.get(key)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _boundary_coverage(
    uncertainty: dict[str, Any] | None,
    reference: dict[str, Any],
) -> float:
    if uncertainty is None:
        return 0.0
    uncertainty_geometry = transform(_TO_L93.transform, shape(uncertainty))
    reference_geometry = transform(_TO_L93.transform, shape(reference))
    if reference_geometry.boundary.length <= 0:
        return 0.0
    value = reference_geometry.boundary.intersection(uncertainty_geometry).length
    return float(max(0.0, min(1.0, value / reference_geometry.boundary.length)))


def _unjustified_regression(
    current: dict[str, Any] | None,
    previous: dict[str, Any] | None,
    *,
    correction_recorded: bool,
) -> bool:
    if current is None or previous is None or correction_recorded or current == previous:
        return False
    current_geometry = shape(current)
    previous_geometry = shape(previous)
    if not isinstance(current_geometry, Polygon | MultiPolygon) or not isinstance(
        previous_geometry, Polygon | MultiPolygon
    ):
        return True
    return not previous_geometry.within(current_geometry.buffer(1e-7))


def simplify_reference_for_evaluation(
    reference: Part4ReferenceSnapshotV1,
) -> tuple[Part4ReferenceSnapshotV1, float]:
    """Reduce vertices below the reference's stated spatial resolving power."""

    if reference.geometry_geojson is None:
        return reference, 0.0
    tolerance_m = max(
        20.0,
        min(reference.resolution_m, reference.spatial_accuracy_m) / 2.0,
    )
    source = _polygonal_result(shape(reference.geometry_geojson))
    if source is None:
        raise ValueError("calibration reference is not polygonal")
    # Remove vertices below the product's resolving power before the costly
    # projection. This conservative degree conversion is only a pre-pass; the
    # authoritative simplification remains metric in EPSG:2154 below.
    pre_simplified = _polygonal_result(
        source.simplify(tolerance_m / 111_320.0, preserve_topology=False)
    )
    if pre_simplified is None:
        raise ValueError("calibration reference cannot be pre-simplified")
    projected = _polygonal_result(transform(_TO_L93.transform, pre_simplified))
    if projected is None:
        raise ValueError("calibration reference is not polygonal after projection")
    simplified = _polygonal_result(projected.simplify(tolerance_m, preserve_topology=True))
    if simplified is None:
        raise ValueError("simplified calibration reference is not polygonal")
    wgs84 = _polygonal_result(transform(_TO_WGS84.transform, simplified))
    if wgs84 is None:
        raise ValueError("simplified calibration reference cannot be reprojected")
    return (
        reference.model_copy(update={"geometry_geojson": mapping(wgs84)}),
        tolerance_m,
    )


def evaluate_frozen_affected_prediction(
    *,
    frozen_state_path: Path,
    frozen_grid_path: Path | None,
    reference: Part4ReferenceSnapshotV1,
    previous_affected: dict[str, Any] | None = None,
    comparison_cache: dict[str, Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    """Open a reference only after the prediction has been serialized."""

    if reference.component != "affected" or reference.grade not in {"A", "B"}:
        raise ValueError("affected calibration accepts only grade A or B affected references")
    if reference.geometry_geojson is None:
        raise ValueError("raster references must be materialized before geometry evaluation")
    state = _load_state(frozen_state_path)
    perimeter = state.perimeter
    if state.incident_id != reference.incident_id or state.episode_id != reference.episode_id:
        raise ValueError("prediction and reference identifiers differ")
    if perimeter.local_date != reference.valid_at.date():
        raise ValueError("prediction and reference must describe the same local date")
    forbidden = set(reference.forbidden_input_refs)
    exposed = {
        *state.source_observation_ids,
        *state.source_family_ids,
        *perimeter.evidence_refs,
    }
    leakage = bool(forbidden.intersection(exposed))
    affected = perimeter.affected
    geometry_valid = _geometry_valid(affected)
    observed_resolution = _source_resolution(frozen_grid_path)
    boundary_tolerance = max(100.0, perimeter.resolution_m or 0, observed_resolution or 0)
    comparison: Mapping[str, object] = {}
    comparison_cached = False
    if affected is not None and geometry_valid and not leakage:
        comparison_key = sha256_hex({
            "affected": affected, "reference": reference.geometry_geojson,
            "date": perimeter.local_date.isoformat(), "boundary_tolerance_m": boundary_tolerance,
        })
        if comparison_cache is not None and comparison_key in comparison_cache:
            comparison = comparison_cache[comparison_key]
            comparison_cached = True
        else:
            comparison = compare_activity_zones(
                affected,
                reference.geometry_geojson,
                predicted_local_date=perimeter.local_date,
                official_local_date=reference.valid_at.date(),
                boundary_tolerance_m=boundary_tolerance,
            ).to_dict()
            if comparison_cache is not None:
                comparison_cache[comparison_key] = comparison
    observable_fraction, uncertainty_fraction = _grid_features(frozen_grid_path)
    simulated = any(
        value.casefold().startswith(("simulation:", "simulated:", "mock:"))
        for value in (*state.source_observation_ids, *state.source_family_ids)
    )
    age_hours = (
        float(state.latest_observation_age_seconds or 0) / 3_600.0
        if state.latest_observation_at is not None
        else 0.0
    )
    correction_recorded = "affected_geometry_corrected" in state.contradiction_codes
    return {
        "schema": "fireviewer.part4-calibration-evaluation-row.v1",
        "incident_id": state.incident_id,
        "episode_id": state.episode_id,
        "local_date": state.local_date.isoformat(),
        "reference_id": reference.reference_id,
        "reference_grade": reference.grade,
        "prediction_frozen_before_reference_open": True,
        "reference_leakage_detected": leakage,
        "simulated_contribution_detected": simulated,
        "geometry_valid": geometry_valid,
        "iou": _metric_float(comparison, "intersection_over_union"),
        "boundary_f1": _metric_float(comparison, "boundary_f1"),
        "surface_bias_ratio": _metric_float(comparison, "surface_bias_percent") / 100.0,
        "predicted_area_ha": _metric_float(comparison, "predicted_area_ha"),
        "reference_area_ha": _metric_float(comparison, "official_area_ha"),
        "centroid_distance_m": _metric_float(comparison, "centroid_distance_m"),
        "geometry_comparison_cached": comparison_cached,
        "hausdorff95_m": _metric_float(comparison, "hausdorff95_m"),
        "uncertainty_boundary_coverage": _boundary_coverage(
            perimeter.uncertainty_band,
            reference.geometry_geojson,
        ),
        "evaluation_method_revision": "source-resolution-v3",
        "best_observation_resolution_m": observed_resolution,
        "observation_resolution_verified": observed_resolution is not None,
        "boundary_tolerance_m": boundary_tolerance,
        "grid_resolution_m": perimeter.resolution_m,
        "observed_fraction": perimeter.observed_fraction,
        "fused_fraction": perimeter.fused_fraction,
        "interpolated_fraction": perimeter.interpolated_fraction,
        "evidence_strength": perimeter.evidence_strength,
        "observable_fraction": observable_fraction,
        "uncertainty_fraction": uncertainty_fraction,
        "resolution_m": perimeter.resolution_m or reference.resolution_m,
        "observation_age_hours": age_hours,
        "independent_family_count": len(state.source_family_ids),
        "contradiction_count": len(state.contradiction_codes),
        "contradiction_codes": list(state.contradiction_codes),
        "reconstruction_status": state.status,
        "unjustified_affected_regression_detected": _unjustified_regression(
            affected,
            previous_affected,
            correction_recorded=correction_recorded,
        ),
    }


__all__ = ["evaluate_frozen_affected_prediction", "simplify_reference_for_evaluation"]
