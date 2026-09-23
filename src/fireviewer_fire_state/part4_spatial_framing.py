"""Geographic admission and continuity around the existing CPU fusion formula."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, date, datetime, time
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
from shapely import get_num_coordinates, normalize
from shapely.geometry import LineString, MultiPolygon, Polygon, shape
from shapely.ops import unary_union

from fireviewer_contracts.backend.fire_state_schemas import (
    DailyFireStateV2,
    FusionProfileV1,
    GeometryCorrectionProposalV1,
    SpatialObservationV2,
)
from fireviewer_contracts.backend.hashing import sha256_hex
from fireviewer_contracts.backend.part4_framing_schemas import (
    IncidentSpatialSeedV1,
    Part4FramingReceiptV1,
    Part4SpatialContextV1,
    SpatialAdmissionDecisionV1,
    SpatialAnchorV1,
    SpatialObservationAvailabilityV1,
    SurfaceAreaEvidenceV1,
)
from fireviewer_fire_state.fire_state_fusion import (
    FUSION_ALGORITHM_VERSION,
    FusedFireState,
    _canonical_multipolygon,
    _canonical_wgs84_multipolygon,
    _fuse_probability_state,
    _projected_geojson,
    _projected_observation_geometry,
)

_SPATIAL = {
    "burned_probability",
    "active_probability",
    "official_perimeter",
    "camera_ground_intersection",
}
_Proposal = tuple[dict[str, Any], str, str, Literal["affected", "active"]]


def day_end(local_date: date) -> datetime:
    return datetime.combine(local_date, time.max, tzinfo=ZoneInfo("Europe/Paris")).astimezone(UTC)


def area_ha(geometry: dict[str, Any] | None) -> float:
    projected = _projected_geojson(geometry)
    return float(projected.area / 10_000) if projected is not None else 0.0


def _dilate_detailed_polygon(geometry: Any, radius_m: float) -> Any:
    """Offset outer rings and erode holes separately, with bounded line chunks.

    For a valid polygon the dilation is its dilated shell minus eroded holes.
    This avoids noding all rings together during a global GEOS buffer. Polygon
    components are reunited only after their individually bounded offsets.
    """
    parts = list(geometry.geoms) if isinstance(geometry, MultiPolygon) else [geometry]
    dilated = []
    for polygon in parts:
        shell = Polygon(polygon.exterior)
        coordinates = np.asarray(polygon.exterior.coords)
        if len(coordinates) > 8192:
            bands = []
            for start in range(0, len(coordinates) - 1, 1024):
                segment = LineString(coordinates[start : start + 1025])
                bands.append(segment.buffer(radius_m))
            # Offset chunks are small, and the shell fills the inner boundary.
            expanded = unary_union([shell, *bands])
        else:
            expanded = shell.buffer(radius_m)
        holes = [Polygon(ring).buffer(-radius_m) for ring in polygon.interiors]
        remaining_holes = [hole for hole in holes if not hole.is_empty]
        if remaining_holes:
            expanded = expanded.difference(unary_union(remaining_holes))
        dilated.append(expanded)
    return unary_union(dilated)


def _search_dilation(geometry: Any, radius_m: float, cell_size_m: float) -> Any:
    """Bound a search dilation without giant intersections of detailed segments.

    Euclidean dilation composes by adding radii. A first grid-sized dilation
    removes sub-grid concavities before the large dilation; the sum is still
    exactly the authorised radius. No simplification touches the source contour,
    and neither intermediate geometry is a burned-area observation.
    """
    first_step = min(radius_m, cell_size_m)
    expanded = (
        _dilate_detailed_polygon(geometry, first_step)
        if get_num_coordinates(geometry) > 50_000
        else geometry.buffer(first_step)
    )
    return expanded.buffer(radius_m - first_step) if radius_m > first_step else expanded


def build_spatial_context(
    *,
    seed: IncidentSpatialSeedV1,
    local_date: date,
    observations: tuple[SpatialObservationV2, ...],
    profile: FusionProfileV1,
    prior: FusedFireState | None = None,
    state_valid_at: datetime | None = None,
    evaluation_cutoff_at: datetime | None = None,
    area_evidence: tuple[SurfaceAreaEvidenceV1, ...] = (),
    forbidden_reference_ids: tuple[str, ...] = (),
    observation_availability: dict[str, SpatialObservationAvailabilityV1] | None = None,
) -> Part4SpatialContextV1:
    valid_at = state_valid_at or day_end(local_date)
    cutoff = evaluation_cutoff_at or valid_at
    parent_at = prior.state.state_valid_at if prior else None
    if prior and (parent_at is None or prior.state.spatial_context is None):
        raise ValueError("prior_requires_seeded_reconstruction")
    base = prior.state.perimeter.affected if prior else seed.affected
    projected_base = _projected_geojson(base)
    if projected_base is None:
        raise ValueError("initial_affected_geometry_required")
    elapsed = max(0.0, (valid_at - (parent_at or seed.valid_at)).total_seconds() / 3600)
    growth = min(
        profile.temporal.max_propagation_m, elapsed * profile.temporal.propagation_rate_m_per_hour
    )
    cell_size = profile.grid.resolutions_m[0]
    domain_parts = [
        _search_dilation(projected_base, growth + seed.horizontal_accuracy_m, cell_size)
    ]
    anchors: list[SpatialAnchorV1] = []
    usable: list[tuple[SpatialObservationV2, Any]] = []
    forbidden = set(forbidden_reference_ids)
    for item in observations:
        if (
            item.observed_at > cutoff
            or item.observed_at > valid_at
            or (item.observed_end_at is not None and item.observed_end_at > valid_at)
            or item.observed_at < (parent_at or seed.valid_at)
            or (item.available_at is not None and item.available_at > cutoff)
            or (
                item.observation_id in (observation_availability or {})
                and (observation_availability or {})[item.observation_id].known_at > cutoff
            )
            or forbidden.intersection(
                (
                    *item.evidence_refs,
                    item.upstream_product_id,
                    item.source_revision_sha256,
                    item.observation_id,
                )
            )
        ):
            continue
        geometry = _projected_observation_geometry(item)
        if geometry is None or item.probability <= 0:
            continue
        if item.observation_kind in _SPATIAL | {"thermal_footprint"}:
            anchors.append(
                SpatialAnchorV1(
                    observation_id=item.observation_id,
                    observed_at=item.observed_at,
                    geometry=item.geometry_geojson or {},
                    horizontal_accuracy_m=item.horizontal_accuracy_m or item.resolution_m or 0,
                    source_family_id=item.source_family_id,
                    lineage_id=item.lineage_id,
                    kind=item.observation_kind,
                )
            )
            usable.append((item, geometry))
        if item.observation_kind == "thermal_footprint":
            # A thermal detection may anchor a separate component. Its footprint
            # remains evidence, not the boundary of that component.
            domain_parts.append(
                _search_dilation(geometry, growth + (item.horizontal_accuracy_m or 0), cell_size)
            )
    for index, (left, geometry) in enumerate(usable):
        if left.observation_kind not in _SPATIAL:
            continue
        for right, other in usable[index + 1 :]:
            if (
                right.observation_kind not in _SPATIAL
                or left.source_family_id == right.source_family_id
                or left.lineage_id == right.lineage_id
            ):
                continue
            overlap = geometry.intersection(other)
            if not overlap.is_empty:
                domain_parts.append(
                    _search_dilation(
                        overlap,
                        growth
                        + max(left.horizontal_accuracy_m or 0, right.horizontal_accuracy_m or 0),
                        cell_size,
                    )
                )
    # Computational domains may contain sub-centimetre slivers after buffering.
    # Rounding coordinates can collapse a hole or join rings and invalidate them.
    # Preserve their precision; the administrative input is never repaired.
    domain = _canonical_multipolygon(unary_union(domain_parts), round_coordinates=False)
    return Part4SpatialContextV1(
        seed=seed,
        local_date=local_date,
        state_valid_at=valid_at,
        evaluation_cutoff_at=cutoff,
        parent_state_sha256=prior.state.source_input_sha256 if prior else None,
        parent_valid_at=parent_at,
        admissible_domain=domain,
        calculation_domain=domain,
        anchors=tuple(sorted(anchors, key=lambda item: item.observation_id)),
        area_evidence=area_evidence,
        forbidden_reference_ids=forbidden_reference_ids,
        observation_availability=observation_availability or {},
    )


def _product_key(item: SpatialObservationV2) -> str:
    return sha256_hex([
        item.lineage_id, item.observation_kind, item.target_state, item.upstream_product_id,
    ])


def _observation_fingerprint(item: SpatialObservationV2) -> str:
    # A wrapper ID, another citation or a later availability annotation does not
    # constitute new spatial evidence from the same immutable product.
    return sha256_hex(item.model_dump(
        mode="json", by_alias=True,
        exclude={"observation_id", "available_at", "evidence_refs"},
    ))


def _component_id(component: Polygon) -> str:
    return hashlib.sha256(normalize(component).wkb).hexdigest()


def _known_at(item: SpatialObservationV2, context: Part4SpatialContextV1) -> datetime | None:
    declared = context.observation_availability.get(item.observation_id)
    if declared is None:
        return item.available_at
    if declared.known_at < (item.observed_end_at or item.observed_at) or (
        item.available_at is not None and declared.known_at < item.available_at
    ):
        raise ValueError("observation_availability_precedes_source")
    return declared.known_at


def _admit(
    observations: tuple[SpatialObservationV2, ...],
    context: Part4SpatialContextV1,
    prior: FusedFireState | None,
    *,
    temporal_basis: Literal["daily", "instant"] = "daily",
) -> tuple[
    tuple[SpatialObservationV2, ...], dict[str, str], list[_Proposal], float,
    dict[str, SpatialAdmissionDecisionV1],
]:
    domain = _projected_geojson(context.admissible_domain)
    assert domain is not None
    admitted: list[SpatialObservationV2] = []
    rejected: dict[str, str] = {}
    proposals: list[_Proposal] = []
    outside_area = 0.0
    decisions = (
        dict(prior.state.framing.observation_admissions) if prior and prior.state.framing else {}
    )
    reviewed = {record.observation_fingerprint: record for record in decisions.values()}
    products: dict[str, list[SpatialAdmissionDecisionV1]] = {}
    for record in decisions.values():
        products.setdefault(record.product_key, []).append(record)
    seen = {
        key: value
        for data in (prior.lineage_metadata.values() if prior else ())
        for key, value in data.get("observations", {}).items()
    }
    forbidden = set(context.forbidden_reference_ids)
    for item in sorted(observations, key=lambda item: item.observation_id):
        known_at = _known_at(item, context)
        refs = (
            *item.evidence_refs,
            item.upstream_product_id,
            item.source_revision_sha256,
            item.observation_id,
        )
        if forbidden.intersection(refs):
            raise ValueError("evaluation_reference_in_spatial_inputs")
        if item.raster_uri is not None and item.geometry_geojson is None:
            raise ValueError("raster_observation_not_materialized")
        if (
            item.observed_at > context.state_valid_at
            or (item.observed_end_at is not None and item.observed_end_at > context.state_valid_at)
            or (item.available_at is not None and item.available_at > context.evaluation_cutoff_at)
            or (known_at is not None and known_at > context.evaluation_cutoff_at)
        ):
            rejected[item.observation_id] = "future_observation"
            continue
        fingerprint = _observation_fingerprint(item)
        existing = decisions.get(item.observation_id)
        if existing is not None and existing.observation_fingerprint != fingerprint:
            raise ValueError("observation_revision_requires_replay")
        existing = existing or reviewed.get(fingerprint)
        if existing is not None:
            rejected[item.observation_id] = (
                "already_integrated" if existing.reason == "admitted"
                else "previously_contested_observation"
            )
            continue
        product_key = _product_key(item)
        prior_product = products.get(product_key, [])
        if any(
            record.source_revision_sha256 != item.source_revision_sha256
            or record.processor_revision != item.processor_revision
            for record in prior_product
        ):
            raise ValueError("lineage_reprocessing_requires_revision")
        held_ids = {key for record in prior_product for key in record.held_component_ids}
        if item.observation_id in seen:
            if seen[item.observation_id] != sha256_hex(item.model_dump(mode="json", by_alias=True)):
                raise ValueError("observation_revision_requires_replay")
            rejected[item.observation_id] = "already_integrated"
            continue
        if item.observed_at < (context.parent_valid_at or context.seed.valid_at):
            key = sha256_hex([item.lineage_id, item.observation_kind, item.target_state])
            if (
                prior is None
                or (
                    key not in prior.lineage_metadata
                    and not (
                        known_at is not None
                        and context.parent_valid_at is not None
                        and context.parent_valid_at < known_at <= context.evaluation_cutoff_at
                    )
                )
                or item.observed_at < context.seed.valid_at
                or item.observation_kind == "valid_negative"
            ):
                rejected[item.observation_id] = "historical_observation_requires_revision"
                continue
        if temporal_basis == "daily" and item.observation_kind == "valid_negative" and (
            item.observed_at.astimezone(ZoneInfo("Europe/Paris")).date() != context.local_date
        ):
            rejected[item.observation_id] = "historical_observation_requires_revision"
            continue
        geometry = _projected_observation_geometry(item)
        admitted_id: str | None = item.observation_id
        reason = "admitted"
        contested_ids: list[str] = []
        if geometry is not None and (held_ids or not domain.buffer(0.1).covers(geometry)):
            # Reject complete connected components, not the portion outside a
            # clipping box. A bounding-box cut must not manufacture a boundary.
            source = shape(item.geometry_geojson) if item.geometry_geojson else None
            if isinstance(source, Polygon | MultiPolygon):
                components = [source] if isinstance(source, Polygon) else list(source.geoms)
                inside_components = []
                outside_components = []
                previously_held = False
                for component in components:
                    from shapely.geometry import mapping

                    component_json = dict(mapping(component))
                    projected = _projected_geojson(component_json)
                    assert projected is not None
                    component_id = _component_id(component)
                    if component_id in held_ids:
                        contested_ids.append(component_id)
                        previously_held = True
                    elif domain.buffer(0.1).covers(projected):
                        inside_components.append(component)
                    else:
                        outside_area += float(projected.difference(domain).area / 10_000)
                        outside_components.append(component)
                        contested_ids.append(component_id)
                # A fragmented raster is still one observation. Preserve every
                # complete component without manufacturing thousands of IDs.
                if inside_components:
                    inside_json = _canonical_wgs84_multipolygon(
                        MultiPolygon(inside_components), round_coordinates=False
                    )
                    admitted_id = (
                        f"{item.observation_id[:110]}-admitted-" + sha256_hex(inside_json)[:12]
                    )
                    admitted.append(
                        item.model_copy(
                            update={
                                "observation_id": admitted_id,
                                "geometry_geojson": inside_json,
                            }
                        )
                    )
                else:
                    admitted_id = None
                if outside_components and item.observation_kind != "thermal_footprint":
                    outside_json = _canonical_wgs84_multipolygon(
                        MultiPolygon(outside_components), round_coordinates=False
                    )
                    assert outside_json is not None
                    proposals.append(
                        (
                            outside_json,
                            "outside_admissible_domain",
                            item.observation_id,
                            item.target_state if item.target_state == "active" else "affected",
                        )
                    )
                reason = (
                    "previously_contested_component" if previously_held and not outside_components
                    else "outside_admissible_domain"
                )
                if not contested_ids:
                    reason = "admitted"
            else:
                admitted_id = None
                reason = "outside_admissible_domain"
            if reason != "admitted":
                rejected[item.observation_id] = reason
        else:
            admitted.append(item)
        record = SpatialAdmissionDecisionV1(
            product_key=product_key,
            observation_fingerprint=fingerprint,
            source_revision_sha256=item.source_revision_sha256,
            processor_revision=item.processor_revision,
            decided_at=context.state_valid_at,
            admitted_observation_id=admitted_id,
            reason=reason,
            held_component_ids=tuple(sorted(contested_ids)),
        )
        decisions[item.observation_id] = record
        reviewed[fingerprint] = record
        products.setdefault(product_key, []).append(record)
    return tuple(admitted), rejected, proposals, outside_area, decisions


def _area_checks(
    context: Part4SpatialContextV1,
    result: FusedFireState,
    previous_area: float,
    episode_id: str | None,
) -> tuple[tuple[dict[str, Any], ...], bool]:
    checks: list[dict[str, Any]] = []
    blocked = False
    for evidence in context.area_evidence:
        if set(context.forbidden_reference_ids).intersection(
            (evidence.evidence_id, evidence.source_id, evidence.source_revision)
        ):
            raise ValueError("evaluation_reference_in_surface_inputs")
        applicable = (
            evidence.incident_id == context.seed.incident_id
            and evidence.available_at <= context.evaluation_cutoff_at
            and evidence.valid_from <= context.state_valid_at <= evidence.valid_until
            and (evidence.scope == "incident" or evidence.episode_id == episode_id)
        )
        # An episode total cannot cap the incident's accumulated burned area.
        if evidence.component == "affected" and evidence.scope == "episode":
            applicable = False
        value = area_ha(getattr(result.state.perimeter, evidence.component))
        if evidence.accumulation == "incremental":
            applicable = (
                applicable
                and evidence.component == "affected"
                and (evidence.valid_from == (context.parent_valid_at or context.seed.valid_at))
            )
            value = max(0, value - previous_area)
        lower, upper = evidence.lower_ha, evidence.upper_ha
        if evidence.qualifier == "exact":
            lower = upper = evidence.value_ha
        conflict = applicable and (
            (lower is not None and value < lower) or (upper is not None and value > upper)
        )
        checks.append(
            {
                "evidence_id": evidence.evidence_id,
                "component": evidence.component,
                "applicable": applicable,
                "estimated_ha": value,
                "lower_ha": lower,
                "upper_ha": upper,
                "nominal_ha": evidence.value_ha,
                "conflict": conflict,
            }
        )
        blocked |= conflict
    return tuple(checks), blocked


def reconstruct_framed_state(
    *,
    incident_id: str,
    episode_id: str | None,
    local_date: date,
    observations: tuple[SpatialObservationV2, ...],
    context: Part4SpatialContextV1,
    prior: FusedFireState | None,
    profile: FusionProfileV1,
    temporal_basis: Literal["daily", "instant"] = "daily",
) -> FusedFireState:
    if context.policy_revision != "part4-spatial-framing-1.1.0":
        raise ValueError("spatial_framing_policy_requires_replay")
    if context.seed.incident_id != incident_id or context.local_date != local_date:
        raise ValueError("spatial_context_identity_mismatch")
    if context.parent_state_sha256 != (prior.state.source_input_sha256 if prior else None):
        raise ValueError("spatial_parent_identity_mismatch")
    if prior and (
        prior.state.spatial_context is None
        or prior.state.spatial_context.policy_revision != context.policy_revision
        or prior.state.spatial_context.seed != context.seed
        or prior.state.framing is None
        or not prior.state.framing.parent_eligible
        or prior.state.fusion_profile != profile.identity()
        or prior.state.state_valid_at != context.parent_valid_at
    ):
        raise ValueError("prior_requires_seeded_reconstruction")
    admitted, rejected, candidates, outside_area, decisions = _admit(
        observations, context, prior, temporal_basis=temporal_basis
    )
    previous = prior.state.perimeter.affected if prior else context.seed.affected
    previous_area = area_ha(previous)
    common: dict[str, Any] = dict(
        incident_id=incident_id,
        episode_id=episode_id,
        local_date=local_date,
        prior_affected=previous,
        prior_active=prior.state.perimeter.active if prior else None,
        prior_observed_at=context.parent_valid_at,
        prior_state_sha256=context.parent_state_sha256,
        profile=profile,
        prior_result=prior,
        spatial_context=context,
        algorithm_version=FUSION_ALGORITHM_VERSION,
        temporal_basis=temporal_basis,
    )
    result = _fuse_probability_state(observations=admitted, **common)
    result = _retain_supported_growth(result, previous, admitted)
    checks, blocked = _area_checks(context, result, previous_area, episode_id)
    if blocked:
        admitted_ids = {item.observation_id for item in admitted}
        for item in observations:
            decision = decisions.get(item.observation_id)
            if decision is not None and decision.admitted_observation_id in admitted_ids:
                geometry = shape(item.geometry_geojson) if item.geometry_geojson else None
                components = (
                    list(geometry.geoms) if isinstance(geometry, MultiPolygon)
                    else [geometry] if isinstance(geometry, Polygon) else []
                )
                decisions[item.observation_id] = decision.model_copy(update={
                    "admitted_observation_id": None,
                    "reason": "surface_area_conflict",
                    "held_component_ids": tuple(sorted(_component_id(part) for part in components)),
                })
        for component in ("affected", "active"):
            candidate_geometry = getattr(result.state.perimeter, component)
            if candidate_geometry is not None and any(
                check["conflict"] and check["component"] == component for check in checks
            ):
                candidates.append(
                    (candidate_geometry, "surface_area_conflict", "area-constraints", component)
                )
        rejected.update({item.observation_id: "surface_area_conflict" for item in admitted})
        result = _fuse_probability_state(observations=(), **common)
        result = _retain_supported_growth(result, previous, ())
        admitted = ()
    input_sha = sha256_hex(
        {
            "algorithm": FUSION_ALGORITHM_VERSION,
            "context": context.model_dump(mode="json", by_alias=True),
            "profile": profile.identity().model_dump(mode="json"),
            "observations": [
                item.model_dump(mode="json", by_alias=True)
                for item in sorted(observations, key=lambda row: row.observation_id)
            ],
            **({"temporal_basis": "instant-v1"} if temporal_basis == "instant" else {}),
        }
    )
    source_perimeter_sha = sha256_hex(result.state.perimeter.model_dump(mode="json", by_alias=True))
    proposals = tuple(
        GeometryCorrectionProposalV1(
            correction_id=f"GCP-{input_sha[:24]}-{index}",
            incident_id=incident_id,
            local_date=local_date,
            source_perimeter_sha256=source_perimeter_sha,
            competing_geometry_geojson=geometry,
            component=component,
            reason_codes=(reason,),
            evidence_refs=(reference,),
        ).model_dump(mode="json", by_alias=True)
        for index, (geometry, reason, reference, component) in enumerate(candidates)
    )
    for identifier, decision in list(decisions.items()):
        proposal_ids = tuple(
            proposal["correction_id"] for proposal in proposals
            if identifier in proposal["evidence_refs"] or (
                decision.decided_at == context.state_valid_at
                and decision.reason == "surface_area_conflict"
                and "area-constraints" in proposal["evidence_refs"]
            )
        )
        if proposal_ids:
            decisions[identifier] = decision.model_copy(update={
                "proposal_ids": tuple(sorted({*decision.proposal_ids, *proposal_ids})),
            })
    positive = [
        _projected_observation_geometry(item)
        for item in admitted
        if item.probability > 0 and item.observation_kind != "valid_negative"
    ]
    positive = [item for item in positive if item is not None]
    supported = float(unary_union(positive).area / 10_000) if positive else 0.0
    area = area_ha(result.state.perimeter.affected)
    receipt = Part4FramingReceiptV1(
        context_sha256=sha256_hex(context.model_dump(mode="json", by_alias=True)),
        seed_id=context.seed.seed_id,
        seed_revision=context.seed.revision,
        parent_state_sha256=context.parent_state_sha256,
        previous_area_ha=previous_area,
        affected_area_ha=area,
        added_area_ha=max(0, area - previous_area),
        spatially_supported_area_ha=supported,
        outside_domain_area_ha=outside_area,
        admitted_observation_ids=tuple(item.observation_id for item in admitted),
        rejected_observations=rejected,
        observation_admissions=decisions,
        late_observation_ids=tuple(sorted(
            item.observation_id for item in admitted
            if item.observed_at < (context.parent_valid_at or context.seed.valid_at)
            or item.observed_at.astimezone(ZoneInfo("Europe/Paris")).date() != local_date
        )),
        pending_proposal_ids=tuple(sorted({
            *(
                prior.state.framing.pending_proposal_ids
                if prior and prior.state.framing else ()
            ),
            *(proposal["correction_id"] for proposal in proposals),
        })),
        area_checks=checks,
        reason_codes=tuple(sorted(set(rejected.values()))),
        competing_proposals=proposals,
        active_knowledge="estimated" if result.state.perimeter.active else "unknown",
    )
    perimeter = result.state.perimeter.model_copy(update={"framing": receipt})
    state = result.state.model_copy(
        update={
            "state_id": (
                f"IFS-{context.state_valid_at.astimezone(UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{input_sha[:24]}"
                if temporal_basis == "instant" else f"DFS-{local_date.isoformat()}-{input_sha[:24]}"
            ),
            "source_input_sha256": input_sha,
            "framing": receipt,
            "perimeter": perimeter,
        }
    )
    state = DailyFireStateV2.model_validate(state.model_dump())
    return replace(result, state=state)


def _retain_supported_growth(
    result: FusedFireState,
    previous: dict[str, Any] | None,
    admitted: tuple[SpatialObservationV2, ...],
) -> FusedFireState:
    """No grid-edge inflation on an unobserved day; exact prior edges survive."""
    spatial = [
        shape(item.geometry_geojson)
        for item in admitted
        if item.geometry_geojson is not None
        and item.probability > 0
        and item.observation_kind
        in {"burned_probability", "active_probability", "official_perimeter", "modelled_perimeter"}
    ]
    previous_shape = shape(previous) if previous else Polygon()
    current = result.state.perimeter.affected
    growth = shape(current).intersection(unary_union(spatial)) if current and spatial else Polygon()
    union = previous_shape if growth.is_empty else unary_union([previous_shape, growth])
    affected = _canonical_wgs84_multipolygon(union, round_coordinates=False)
    perimeter = result.state.perimeter.model_copy(update={"affected": affected})
    return replace(result, state=result.state.model_copy(update={"perimeter": perimeter}))
