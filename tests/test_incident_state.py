from datetime import UTC, date, datetime, timedelta

import numpy as np
import pytest
from shapely.geometry import box, mapping, shape

from fireviewer_contracts.backend.fire_state_schemas import SpatialObservationV2
from fireviewer_contracts.backend.incident_state_schemas import TemporalEvidence
from fireviewer_contracts.backend.part4_framing_schemas import IncidentSpatialSeedV1
from fireviewer_fire_state.fire_state_fusion import fuse_daily_fire_state, fuse_state_at
from fireviewer_fire_state.incident_state import daily_interval, update_incident_state
from fireviewer_fire_state.part4_fusion_profiles import load_fusion_profile
from fireviewer_fire_state.part4_spatial_framing import build_spatial_context

T = datetime(2026, 7, 1, 8, tzinfo=UTC)


def seed():
    return IncidentSpatialSeedV1(seed_id="seed-test", revision=1, incident_id="FR-test", episode_id="E01",
        longitude=5.11, latitude=44.71, affected=mapping(box(5.1, 44.7, 5.12, 44.72)),
        valid_at=T, created_at=T, created_by="test")


def evidence(at=T + timedelta(hours=2), known=None, kind="official_perimeter"):
    known = known or at
    observation = SpatialObservationV2(observation_id="official", observation_kind=kind,
        target_state="active" if kind == "valid_negative" else "affected", observed_at=at,
        probability=0 if kind == "valid_negative" else 0.98,
        geometry_geojson=mapping(box(5.101, 44.701, 5.119, 44.719)),
        coverage_geojson=mapping(box(5.1, 44.7, 5.12, 44.72)) if kind == "valid_negative" else None,
        source_family_id="official", lineage_id="official-1", upstream_product_id="official-1",
        source_revision_sha256="a" * 64, processor_revision="test", available_at=known)
    return TemporalEvidence(evidence_id="proof-1", revision=1, incident_id="FR-test", episode_id="E01",
        source_id="test", content_sha256="a" * 64, license="fixture", media_kind="geometry",
        admissibility="admitted", observed_at=at, retrieved_at=known, recorded_at=known,
        observations=(observation,))


def test_intraday_fusion_and_empty_updates_preserve_affected_surface():
    proof = evidence()
    first = update_incident_state(previous_state=None, evidence=(proof,), valid_at=proof.observed_at,
        knowledge_cutoff=proof.known_at, spatial_context=seed())
    later = update_incident_state(previous_state=first.checkpoint, evidence=(),
        valid_at=T + timedelta(hours=8), knowledge_cutoff=T + timedelta(hours=8), spatial_context=seed())
    assert first.checkpoint.state.state_id != later.checkpoint.state.state_id
    assert first.checkpoint.state.local_date == later.checkpoint.state.local_date
    assert shape(later.checkpoint.state.perimeter.affected).covers(shape(first.checkpoint.state.perimeter.affected))
    assert later.checkpoint.state.latest_observation_at == first.checkpoint.state.latest_observation_at


def test_late_evidence_is_excluded_from_causal_reconstruction():
    proof = evidence(known=T + timedelta(days=2))
    causal = update_incident_state(previous_state=None, evidence=(proof,), valid_at=T + timedelta(hours=3),
        knowledge_cutoff=T + timedelta(hours=3), reconstruction_mode="causal", spatial_context=seed())
    retrospective = update_incident_state(previous_state=None, evidence=(proof,), valid_at=T + timedelta(hours=3),
        knowledge_cutoff=proof.known_at, spatial_context=seed())
    assert causal.evidence_refs == ()
    assert retrospective.evidence_refs == (proof.reference,)
    assert "official" in retrospective.checkpoint.state.source_observation_ids


def test_daily_adapter_has_exact_probability_and_provenance_parity():
    proof = evidence()
    profile = load_fusion_profile("part4-framed-v1", algorithm_version="3.3.0")
    context = build_spatial_context(seed=seed(), local_date=T.date(), observations=proof.observations, profile=profile)
    arguments = dict(incident_id="FR-test", episode_id="E01", observations=proof.observations, spatial_context=context, profile=profile)
    daily = fuse_daily_fire_state(local_date=T.date(), **arguments)
    adapted = fuse_state_at(valid_at=context.state_valid_at, daily_compatibility=True, **arguments)
    assert daily.state.model_dump() == adapted.state.model_dump()
    for name in ("affected_probability", "active_probability", "observable_probability", "observed_support", "multi_source_support", "prior_interpolated_support", "uncertainty_support"):
        np.testing.assert_array_equal(getattr(daily, name), getattr(adapted, name))


@pytest.mark.parametrize("day,hours", [(date(2026, 3, 29), 23), (date(2026, 10, 25), 25), (date(2026, 7, 1), 24)])
def test_daily_projection_uses_local_calendar_boundaries(day, hours):
    start, end = daily_interval(day)
    assert end - start == timedelta(hours=hours)
    assert start.tzinfo == end.tzinfo == UTC


def test_duplicate_proof_versions_cannot_be_counted_twice():
    proof = evidence()
    with pytest.raises(ValueError, match="one immutable version"):
        update_incident_state(previous_state=None, evidence=(proof, proof), valid_at=proof.observed_at,
            knowledge_cutoff=proof.known_at, spatial_context=seed())


@pytest.mark.parametrize("changes", [{"incident_id": "FR-other"}, {"episode_id": "E99"}])
def test_foreign_evidence_cannot_contribute_to_this_incident_map(changes):
    proof = evidence().model_copy(update=changes)
    with pytest.raises(ValueError, match="another incident or episode"):
        update_incident_state(previous_state=None, evidence=(proof,), valid_at=proof.observed_at,
            knowledge_cutoff=proof.known_at, spatial_context=seed())


def test_active_decay_does_not_jump_at_local_midnight():
    proof = evidence(at=T + timedelta(hours=10))
    proof = proof.model_copy(update={"observations": (
        proof.observations[0].model_copy(update={"target_state": "both"}),
    )})
    peaks = []
    for at in (datetime(2026, 7, 1, 21, 59, 59, tzinfo=UTC), datetime(2026, 7, 1, 22, 0, 1, tzinfo=UTC)):
        result = update_incident_state(previous_state=None, evidence=(proof,), valid_at=at,
            knowledge_cutoff=at, spatial_context=seed())
        peaks.append(float(np.max(result.checkpoint.active_probability)))
    assert peaks[0] > 0.1
    assert peaks[1] <= peaks[0]
    assert peaks[0] - peaks[1] < 0.001
