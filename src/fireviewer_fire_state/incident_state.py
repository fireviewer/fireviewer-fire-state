"""Continuous incident fusion, with an internal adapter for existing raster checkpoints."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from fireviewer_contracts.backend.fire_state_schemas import FusionProfileV1
from fireviewer_contracts.backend.incident_state_schemas import (
    EvidenceRevisionRef,
    ReconstructionMode,
    TemporalEvidence,
    check_cutoff,
)
from fireviewer_contracts.backend.part4_framing_schemas import IncidentSpatialSeedV1
from fireviewer_fire_state.fire_state_fusion import FusedFireState, fuse_state_at
from fireviewer_fire_state.part4_fusion_profiles import load_fusion_profile
from fireviewer_fire_state.part4_spatial_framing import build_spatial_context


@dataclass(frozen=True)
class IncidentFusionResult:
    valid_at: datetime
    evidence_refs: tuple[EvidenceRevisionRef, ...]
    checkpoint: FusedFireState


def daily_interval(
    local_date: date, timezone: str = "Europe/Paris"
) -> tuple[datetime, datetime]:
    """Local calendar boundaries, [start, end), including 23/25-hour days."""
    zone = ZoneInfo(timezone)
    start = datetime.combine(local_date, time.min, tzinfo=zone)
    end = datetime.combine(local_date + timedelta(days=1), time.min, tzinfo=zone)
    return start.astimezone(UTC), end.astimezone(UTC)


def update_incident_state(
    *,
    previous_state: FusedFireState | None,
    evidence: tuple[TemporalEvidence, ...],
    valid_at: datetime,
    knowledge_cutoff: datetime,
    spatial_context: IncidentSpatialSeedV1,
    reconstruction_mode: ReconstructionMode = "retrospective",
    profile: FusionProfileV1 | None = None,
) -> IncidentFusionResult:
    """No database, scheduling, review or publication. Callers select immutable versions.

    A corrected/withdrawn earlier proof requires replay from a preceding unaffected state.
    The backend owns this dependency decision; this function never silently edits a parent.
    """
    for instant in (valid_at, knowledge_cutoff):
        if instant.tzinfo is None or instant.utcoffset() is None:
            raise ValueError("temporal fusion requires timezone-aware instants")
    valid_at, knowledge_cutoff = (
        valid_at.astimezone(UTC),
        knowledge_cutoff.astimezone(UTC),
    )
    check_cutoff(valid_at, knowledge_cutoff, reconstruction_mode)
    if spatial_context.valid_at > valid_at:
        raise ValueError("awaiting_spatial_initialization")
    if len({item.evidence_id for item in evidence}) != len(evidence):
        raise ValueError("select one immutable version per proof before fusion")
    observations = []
    references = []
    for item in evidence:
        if item.incident_id != spatial_context.incident_id or (
            item.episode_id is not None
            and item.episode_id != spatial_context.episode_id
        ):
            raise ValueError("evidence belongs to another incident or episode")
        if item.known_at > knowledge_cutoff:
            continue
        if (
            item.observed_at is None
            or (item.observed_until or item.observed_at) > valid_at
        ):
            continue
        references.append(item.reference)
        if item.admissibility != "admitted":
            continue
        for observation in item.observations:
            # System availability cannot be backdated by a provider/model field.
            available = max(item.known_at, observation.available_at or item.known_at)
            observations.append(
                observation.model_copy(update={"available_at": available})
            )
    identities = [item.observation_id for item in observations]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate observation identity across proof versions")
    effective_profile = profile or load_fusion_profile(
        "part4-framed-v1", algorithm_version="3.3.0"
    )
    context = build_spatial_context(
        seed=spatial_context,
        local_date=valid_at.astimezone(ZoneInfo("Europe/Paris")).date(),
        observations=tuple(observations),
        profile=effective_profile,
        prior=previous_state,
        state_valid_at=valid_at,
        evaluation_cutoff_at=knowledge_cutoff,
    )
    result = fuse_state_at(
        incident_id=spatial_context.incident_id,
        episode_id=spatial_context.episode_id,
        valid_at=valid_at,
        observations=tuple(observations),
        spatial_context=context,
        prior_result=previous_state,
        profile=effective_profile,
    )
    return IncidentFusionResult(
        valid_at=valid_at,
        evidence_refs=tuple(sorted(references, key=lambda ref: ref.evidence_id)),
        checkpoint=result,
    )
