"""Reference-isolated Hugging Face replay and evaluation for Part.4 calibration."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from fireviewer_contracts.backend.fire_state_schemas import FusionProfileV1, SpatialObservationV2
from fireviewer_contracts.backend.hashing import sha256_hex
from fireviewer_contracts.backend.part4_calibration_schemas import (
    HuggingFaceArtifactRefV1,
    Part4CalibrationCaseV1,
    Part4CorpusCatalogRowV1,
    Part4ObservationBundleV1,
    Part4PreHoldoutGateReceiptV1,
    Part4ReferenceSnapshotV1,
)
from fireviewer_fire_state.fire_state_fusion import (
    BASELINE_ALGORITHM_VERSION as FUSION_ALGORITHM_VERSION,
)
from fireviewer_fire_state.part4_calibration import (
    ACTIVE_HALF_LIFE_HOURS,
    ACTIVE_THRESHOLDS,
    AFFECTED_THRESHOLDS,
    UNCERTAINTY_HIGH_THRESHOLDS,
    UNCERTAINTY_LOW_THRESHOLDS,
    WEIGHT_MULTIPLIERS,
    affected_objective,
    build_restricted_affected_profiles,
    build_screening_profiles,
    fit_confidence_calibrator,
    rank_candidate_profiles,
    select_restricted_affected_profile,
)
from fireviewer_fire_state.part4_calibration_campaign import (
    CalibrationReplayPayload,
    HuggingFaceCampaignStore,
    replay_case,
    write_scratch_text,
)
from fireviewer_fire_state.part4_calibration_evaluation import (
    evaluate_frozen_affected_prediction,
    simplify_reference_for_evaluation,
)


def _write_json(path: Path, payload: object) -> None:
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", by_alias=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_scratch_text(path, json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized: list[str] = []
    for row in rows:
        if hasattr(row, "model_dump"):
            row = row.model_dump(mode="json", by_alias=True)
        serialized.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    write_scratch_text(path, "\n".join(serialized) + "\n")


def load_corpus_catalog(path: Path) -> tuple[Part4CorpusCatalogRowV1, ...]:
    rows = tuple(
        Part4CorpusCatalogRowV1.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not rows:
        raise ValueError("Part.4 corpus catalog is empty")
    case_ids = [row.case.case_id for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Part.4 corpus catalog contains duplicate cases")
    split_ids = {row.case.split_id for row in rows}
    if len(split_ids) != 1:
        raise ValueError("Part.4 corpus catalog mixes split identities")
    if any(row.case.split == "holdout" for row in rows):
        raise ValueError("calibration catalog must not expose holdout cases")
    return rows


def _download_observation_bundle(
    store: HuggingFaceCampaignStore,
    case: Part4CalibrationCaseV1,
) -> Part4ObservationBundleV1:
    bundle = Part4ObservationBundleV1.model_validate(
        json.loads(store.download(case.observations).read_text(encoding="utf-8"))
    )
    if (
        bundle.case_id != case.case_id
        or bundle.incident_id != case.incident_id
        or bundle.episode_id != case.episode_id
        or bundle.evaluation_cutoff_at != case.evaluation_cutoff_at
    ):
        raise ValueError("observation bundle and calibration case differ")
    forbidden = set(case.forbidden_input_refs)
    for observation in bundle.observations:
        exposed = {
            observation.observation_id,
            observation.upstream_product_id,
            *observation.evidence_refs,
        }
        if forbidden.intersection(exposed):
            raise ValueError("reference-derived input leaked into an observation bundle")
    return bundle


def _support_receipt(
    result: Any, bundle: Part4ObservationBundleV1 | None = None,
    *, observations: Sequence[SpatialObservationV2] | None = None,
) -> dict[str, Any]:
    observable = result.observable_probability
    uncertainty = result.uncertainty_support
    selected_ids = set(result.state.source_observation_ids)
    framing = getattr(result.state, "framing", None)
    if framing is not None:
        selected_ids.update(
            identifier for identifier, decision in framing.observation_admissions.items()
            if decision.admitted_observation_id in selected_ids
        )
    inputs = observations if observations is not None else (bundle.observations if bundle else ())
    resolutions = [
        item.resolution_m for item in inputs
        if item.observation_id in selected_ids and item.resolution_m is not None
        and item.observation_kind != "valid_negative"
    ]
    return {
        "schema": "fireviewer.part4-frozen-grid-support.v1",
        "best_observation_resolution_m": min(resolutions, default=None),
        "observable_fraction": (
            float(np.mean(np.clip(observable, 0.0, 1.0))) if observable is not None else 0.0
        ),
        "uncertainty_fraction": (
            float(np.mean(uncertainty > 0)) if uncertainty is not None else 0.0
        ),
    }


def _sorted_catalog(
    rows: Sequence[Part4CorpusCatalogRowV1],
) -> tuple[Part4CorpusCatalogRowV1, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.case.incident_id,
                row.case.evaluation_cutoff_at,
                row.case.case_id,
            ),
        )
    )


def _incident_groups(
    rows: Sequence[Part4CorpusCatalogRowV1],
) -> tuple[tuple[Part4CorpusCatalogRowV1, ...], ...]:
    grouped: dict[str, list[Part4CorpusCatalogRowV1]] = defaultdict(list)
    for row in _sorted_catalog(rows):
        if row.case.split == "holdout":
            raise ValueError("calibration catalog must not expose holdout cases")
        grouped[row.case.incident_id].append(row)
    for group in grouped.values():
        if len({row.case.split for row in group}) != 1:
            raise ValueError("one incident cannot span calibration and validation splits")
    return tuple(tuple(group) for group in grouped.values())


def _replay_profile_rows(
    profile: FusionProfileV1,
    group: Sequence[Part4CorpusCatalogRowV1],
    bundles: Mapping[str, Part4ObservationBundleV1],
) -> list[dict[str, Any]]:
    """Pure CPU work with a separate chronological prior for each profile/episode."""
    prior_by_episode: dict[str | None, Any] = {}
    frozen_rows: list[dict[str, Any]] = []
    for row in group:
        case = row.case
        prior = prior_by_episode.get(case.episode_id)
        result = replay_case(
            CalibrationReplayPayload(
                case=case,
                observations=bundles[case.case_id].observations,
                prior_affected=prior.perimeter.affected if prior is not None else None,
                prior_active=prior.perimeter.active if prior is not None else None,
                prior_observed_at=prior.latest_observation_at if prior is not None else None,
                prior_state_sha256=prior.source_input_sha256 if prior is not None else None,
            ),
            profile=profile,
        )
        if result.state.published_reference_accessed:
            raise RuntimeError("Part.4 accessed a published reference during replay")
        frozen_rows.append({
            "case_id": case.case_id,
            "split": case.split,
            "state": result.state.model_dump(mode="json", by_alias=True),
            "support": _support_receipt(result, bundles[case.case_id]),
        })
        prior_by_episode[case.episode_id] = result.state
        del result
    return frozen_rows


def _replay_profiles(
    profiles: Sequence[FusionProfileV1],
    group: Sequence[Part4CorpusCatalogRowV1],
    bundles: Mapping[str, Part4ObservationBundleV1],
    *, workers: int,
) -> Iterator[tuple[FusionProfileV1, list[dict[str, Any]]]]:
    if workers == 1:
        for profile in profiles:
            yield profile, _replay_profile_rows(profile, group, bundles)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="part4-cpu") as pool:
        # Submit at most one bounded batch, never queue an entire screening campaign.
        for offset in range(0, len(profiles), workers):
            batch = profiles[offset:offset + workers]
            futures = [
                pool.submit(_replay_profile_rows, profile, group, bundles) for profile in batch
            ]
            for profile, future in zip(batch, futures, strict=True):
                yield profile, future.result()


def replay_screening_to_hf(
    *,
    store: HuggingFaceCampaignStore,
    repo_id: str,
    catalog_repo: str,
    catalog_revision: str,
    catalog_rows: Sequence[Part4CorpusCatalogRowV1],
    base_profile: FusionProfileV1,
    campaign_id: str,
    code_revision: str,
    scratch: Path,
    baseline_only: bool = False,
    restricted_affected: bool = False,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Freeze every screening prediction before any reference artifact is opened."""

    if len(code_revision) < 7:
        raise ValueError("campaign code revision must be immutable")
    ordered = _sorted_catalog(catalog_rows)
    if not ordered:
        raise ValueError("calibration catalog is empty")
    groups = _incident_groups(ordered)
    if baseline_only and restricted_affected:
        raise ValueError("baseline and restricted calibration are mutually exclusive")
    if not 1 <= replay_workers <= 4 or (replay_workers != 1 and not restricted_affected):
        raise ValueError("only restricted calibration permits 2-4 bounded CPU workers")
    if (baseline_only or restricted_affected) and not 1 <= len(groups) <= 15:
        raise ValueError("a functional baseline pilot is limited to 15 complete incidents")
    profiles = (
        (base_profile,) if baseline_only else build_restricted_affected_profiles(base_profile)
        if restricted_affected else build_screening_profiles(base_profile)
    )
    stage = scratch / "screening-replays"
    artifacts: list[tuple[Path, str]] = []
    manifest_profiles: list[dict[str, Any]] = []
    started = time.perf_counter()
    for profile in profiles:
        profile_path = stage / "profiles" / f"{profile.profile_id}.json"
        _write_json(profile_path, profile)
        profile_remote = f"receipts/{campaign_id}/replays/profiles/{profile_path.name}"
        artifacts.append((profile_path, profile_remote))
        manifest_profiles.append(
            {
                "profile_id": profile.profile_id,
                "profile_sha256": profile.profile_sha256,
                "profile_path": profile_remote,
                "profile_byte_count": profile_path.stat().st_size,
            }
        )
    artifact_revision = store.upload_artifacts(
        repo_id=repo_id,
        artifacts=artifacts,
        commit_message=f"Freeze Part.4 screening replays {campaign_id}",
    )
    incident_manifests: list[dict[str, Any]] = []
    for group in groups:
        incident_id = group[0].case.incident_id
        print(json.dumps({"stage": "replay", "incident_id": incident_id}), flush=True)
        incident_key = sha256_hex(incident_id)[:24]
        with store.incident_scope(incident_id) as incident_store:
            # Read each case once, not once for every candidate profile.
            bundles = {
                row.case.case_id: _download_observation_bundle(incident_store, row.case)
                for row in group
            }
            incident_artifacts: list[tuple[Path, str]] = []
            entries: list[dict[str, Any]] = []
            for profile, frozen_rows in _replay_profiles(
                profiles, group, bundles, workers=replay_workers
            ):
                path = incident_store.scratch / "frozen" / f"{profile.profile_id}.jsonl"
                _write_jsonl(path, frozen_rows)
                remote = f"receipts/{campaign_id}/replays/incidents/{incident_key}/{path.name}"
                incident_artifacts.append((path, remote))
                entries.append(
                    {
                        "profile_id": profile.profile_id,
                        "frozen_path": remote,
                        "frozen_byte_count": path.stat().st_size,
                    }
                )
            revision = incident_store.upload_artifacts(
                repo_id=repo_id,
                artifacts=incident_artifacts,
                commit_message=f"Freeze Part.4 incident {incident_id} for {campaign_id}",
            )
            incident_manifests.append(
                {
                    "incident_id": incident_id,
                    "case_ids": [row.case.case_id for row in group],
                    "artifact_revision": revision,
                    "profiles": entries,
                }
            )
        # incident_scope removes this incident only after remote verification succeeds.
    manifest = {
        "schema": "fireviewer.part4-screening-replay-manifest.v2",
        "campaign_id": campaign_id,
        "purpose": (
            "baseline_functional_pilot" if baseline_only else "restricted_affected_calibration"
            if restricted_affected else "parameter_screening"
        ),
        "baseline_profile_id": base_profile.profile_id,
        "replay_workers": replay_workers,
        "created_at": datetime.now(UTC).isoformat(),
        "code_revision": code_revision,
        "algorithm_version": FUSION_ALGORITHM_VERSION,
        "catalog_repo": catalog_repo,
        "catalog_revision": catalog_revision,
        "split_id": ordered[0].case.split_id,
        "incident_count": len({row.case.incident_id for row in ordered}),
        "snapshot_count": len(ordered),
        "profile_count": len(profiles),
        "artifact_revision": artifact_revision,
        "references_opened": False,
        "profiles": manifest_profiles,
        "incidents": incident_manifests,
        "local_retention": "verified_incident_upload_then_cleanup",
        "cpu_seconds": round(time.perf_counter() - started, 6),
    }
    manifest_path = stage / "replay-manifest.json"
    _write_json(manifest_path, manifest)
    manifest_revision = store.upload_artifacts(
        repo_id=repo_id,
        artifacts=[
            (
                manifest_path,
                f"receipts/{campaign_id}/replays/replay-manifest.json",
            )
        ],
        commit_message=f"Publish Part.4 replay manifest {campaign_id}",
    )
    return {**manifest, "manifest_revision": manifest_revision}


def _artifact_ref(
    *,
    repo_id: str,
    revision: str,
    path: str,
    byte_count: int,
) -> HuggingFaceArtifactRefV1:
    return HuggingFaceArtifactRefV1(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path=path,
        byte_count=byte_count,
    )


def _load_frozen_profiles(
    *,
    store: HuggingFaceCampaignStore,
    repo_id: str,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, FusionProfileV1], dict[str, dict[str, dict[str, Any]]]]:
    revision = str(manifest["artifact_revision"])
    profiles: dict[str, FusionProfileV1] = {}
    frozen: dict[str, dict[str, dict[str, Any]]] = {}
    entries = manifest.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise ValueError("screening replay manifest has no profiles")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("screening replay profile entry is invalid")
        profile_ref = _artifact_ref(
            repo_id=repo_id,
            revision=str(entry.get("profile_revision", revision)),
            path=str(entry["profile_path"]),
            byte_count=int(entry["profile_byte_count"]),
        )
        replay_ref = _artifact_ref(
            repo_id=repo_id,
            revision=revision,
            path=str(entry["frozen_path"]),
            byte_count=int(entry["frozen_byte_count"]),
        )
        profile = FusionProfileV1.model_validate(
            json.loads(store.download(profile_ref).read_text(encoding="utf-8"))
        )
        if profile.profile_sha256 != entry.get("profile_sha256"):
            raise ValueError("screening profile identity differs from replay manifest")
        case_rows: dict[str, dict[str, Any]] = {}
        for line in store.download(replay_ref).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            value = json.loads(line)
            case_id = str(value.get("case_id", "")) if isinstance(value, dict) else ""
            if not case_id or case_id in case_rows:
                raise ValueError("frozen screening replay contains duplicate cases")
            case_rows[case_id] = value
        profiles[profile.profile_id] = profile
        frozen[profile.profile_id] = case_rows
    return profiles, frozen


def _evaluation_incidents(
    store: HuggingFaceCampaignStore,
    repo_id: str,
    manifest: Mapping[str, Any],
    rows: Sequence[Part4CorpusCatalogRowV1],
) -> Iterator[
    tuple[
        HuggingFaceCampaignStore,
        dict[str, FusionProfileV1],
        dict[str, dict[str, dict[str, Any]]],
        tuple[Part4CorpusCatalogRowV1, ...],
    ]
]:
    groups = _incident_groups(rows)
    if manifest.get("schema") == "fireviewer.part4-screening-replay-manifest.v1":
        # Historical v1 remains readable; new campaigns always emit incident shards.
        profiles, frozen = _load_frozen_profiles(store=store, repo_id=repo_id, manifest=manifest)
        for group in groups:
            yield store, profiles, frozen, group
        return
    if manifest.get("schema") != "fireviewer.part4-screening-replay-manifest.v2":
        raise ValueError("unsupported screening replay manifest")
    entries = manifest.get("incidents")
    base_profiles = manifest.get("profiles")
    if not isinstance(entries, list) or not isinstance(base_profiles, list):
        raise ValueError("incident replay manifest is incomplete")
    by_incident = {entry["incident_id"]: entry for entry in entries}
    if len(by_incident) != len(groups) or len(entries) != len(groups):
        raise ValueError("incident replay manifest does not match the catalog")
    for group in groups:
        incident_id = group[0].case.incident_id
        entry = by_incident.get(incident_id)
        if entry is None or entry["case_ids"] != [row.case.case_id for row in group]:
            raise ValueError("incident replay cases do not match the catalog")
        shards = {item["profile_id"]: item for item in entry["profiles"]}
        if len(shards) != len(base_profiles):
            raise ValueError("incident replay profiles are incomplete")
        scoped_manifest = {
            "artifact_revision": entry["artifact_revision"],
            "profiles": [
                {
                    **item,
                    **shards[item["profile_id"]],
                    "profile_revision": manifest["artifact_revision"],
                }
                for item in base_profiles
            ],
        }
        with store.incident_scope(incident_id) as scoped:
            profiles, frozen = _load_frozen_profiles(
                store=scoped, repo_id=repo_id, manifest=scoped_manifest
            )
            yield scoped, profiles, frozen, group


def _median(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    values = [float(row.get(key, 0.0)) for row in rows]
    finite = [value for value in values if math.isfinite(value)]
    return float(median(finite)) if finite else 0.0


def _load_immutable_reference(path: Path) -> Part4ReferenceSnapshotV1:
    """Validate metadata without repeating the ingestion-time topology repair."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("immutable calibration reference is not a JSON object")
    geometry = payload.get("geometry_geojson")
    if not isinstance(geometry, dict) or geometry.get("type") not in {
        "Polygon",
        "MultiPolygon",
    }:
        raise ValueError("immutable calibration reference geometry is not polygonal")
    metadata_payload = dict(payload)
    metadata_payload["geometry_geojson"] = {
        "type": "Polygon",
        "coordinates": [[[0.0, 0.0], [0.0, 0.001], [0.001, 0.0], [0.0, 0.0]]],
    }
    validated = Part4ReferenceSnapshotV1.model_validate(metadata_payload)
    # The real geometry was already validated before the immutable HF reference
    # revision was created. It is checked again after bounded simplification.
    return validated.model_copy(update={"geometry_geojson": geometry})


def evaluate_screening_from_hf(
    *,
    store: HuggingFaceCampaignStore,
    repo_id: str,
    catalog_repo: str,
    catalog_revision: str,
    catalog_rows: Sequence[Part4CorpusCatalogRowV1],
    replay_manifest: Mapping[str, Any],
    scratch: Path,
) -> dict[str, Any]:
    """Evaluate only frozen replays, then emit a pre-holdout gate receipt."""

    if replay_manifest.get("references_opened") is not False:
        raise ValueError("screening replays were not proven reference-isolated")
    if replay_manifest.get("catalog_revision") != catalog_revision:
        raise ValueError("replay manifest and calibration catalog revisions differ")
    campaign_id = str(replay_manifest["campaign_id"])
    evaluation_root = scratch / "screening-evaluation"
    rows_by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    profiles: dict[str, FusionProfileV1] = {}
    cache_hits = 0
    unique_evaluations = 0
    geometry_comparisons = 0
    comparison_cache_hits = 0
    for incident_store, profiles, frozen, group in _evaluation_incidents(
        store, repo_id, replay_manifest, catalog_rows
    ):
        print(
            json.dumps({"stage": "evaluation", "incident_id": group[0].case.incident_id}),
            flush=True,
        )
        state_path = incident_store.scratch / "frozen-state.json"
        support_path = incident_store.scratch / "frozen-state.support.json"
        previous_affected: dict[tuple[str, str], dict[str, Any] | None] = {}
        evaluation_cache: dict[str, dict[str, Any]] = {}
        comparison_cache: dict[str, Mapping[str, object]] = {}
        for catalog_row in group:
            reference = _load_immutable_reference(incident_store.download(catalog_row.reference))
            if reference.reference_id != catalog_row.reference_id:
                raise ValueError("catalog and opened reference identifiers differ")
            reference, simplification_m = simplify_reference_for_evaluation(reference)
            for profile_id in sorted(profiles):
                frozen_row = frozen[profile_id].get(catalog_row.case.case_id)
                if frozen_row is None:
                    raise ValueError("screening profile is missing a catalog case")
                _write_json(state_path, frozen_row["state"])
                _write_json(support_path, frozen_row["support"])
                prior_key = (
                    profile_id,
                    catalog_row.case.episode_id or catalog_row.case.incident_id,
                )
                state = frozen_row["state"]
                perimeter = state["perimeter"]
                cache_key = sha256_hex(
                    {
                        "case_id": catalog_row.case.case_id,
                        "state": {
                            "incident_id": state["incident_id"],
                            "episode_id": state.get("episode_id"),
                            "local_date": state["local_date"],
                            "status": state["status"],
                            "latest_observation_at": state.get("latest_observation_at"),
                            "latest_observation_age_seconds": state.get(
                                "latest_observation_age_seconds"
                            ),
                            "source_observation_ids": state["source_observation_ids"],
                            "source_family_ids": state["source_family_ids"],
                            "contradiction_codes": state["contradiction_codes"],
                        },
                        "perimeter": {
                            "affected": perimeter.get("affected"),
                            "uncertainty_band": perimeter.get("uncertainty_band"),
                            "resolution_m": perimeter.get("resolution_m"),
                            "observed_fraction": perimeter["observed_fraction"],
                            "fused_fraction": perimeter["fused_fraction"],
                            "interpolated_fraction": perimeter["interpolated_fraction"],
                            "evidence_strength": perimeter["evidence_strength"],
                        },
                        "support": frozen_row["support"],
                        "previous_affected": previous_affected.get(prior_key),
                        "reference_id": reference.reference_id,
                        "reference_simplification_m": simplification_m,
                    }
                )
                cached = evaluation_cache.get(cache_key)
                if cached is None:
                    cached = evaluate_frozen_affected_prediction(
                        frozen_state_path=state_path,
                        frozen_grid_path=support_path,
                        reference=reference,
                        previous_affected=previous_affected.get(prior_key),
                        comparison_cache=comparison_cache,
                    )
                    comparison_cache_hits += int(cached["geometry_comparison_cached"])
                    cached["reference_simplification_m"] = simplification_m
                    evaluation_cache[cache_key] = cached
                else:
                    cache_hits += 1
                evaluated = dict(cached)
                evaluated.update(
                    {
                        "case_id": catalog_row.case.case_id,
                        "split": catalog_row.case.split,
                        "profile_id": profile_id,
                        "profile_sha256": profiles[profile_id].profile_sha256,
                    }
                )
                rows_by_profile[profile_id].append(evaluated)
                previous_affected[prior_key] = state["perimeter"].get("affected")
        unique_evaluations += len(evaluation_cache)
        geometry_comparisons += len(comparison_cache)
        incident_id = group[0].case.incident_id
        metrics_path = incident_store.scratch / "incident-metrics.json"
        _write_json(
            metrics_path,
            {
                "incident_id": incident_id,
                "rows": {
                    key: [row for row in values if row["incident_id"] == incident_id]
                    for key, values in rows_by_profile.items()
                },
            },
        )
        incident_store.upload_artifacts(
            repo_id=repo_id,
            artifacts=[
                (
                    metrics_path,
                    f"receipts/{campaign_id}/evaluation/incidents/{sha256_hex(incident_id)[:24]}.json",
                )
            ],
            commit_message=f"Freeze Part.4 evaluation for {incident_id}",
        )
    calibration_results = {
        profile_id: [row for row in rows if row["split"] == "calibration"]
        for profile_id, rows in rows_by_profile.items()
    }
    if any(not rows for rows in calibration_results.values()):
        raise ValueError("every screening profile requires calibration rows")
    calibration_ranking = rank_candidate_profiles(calibration_results, keep=8)
    validation_results = {
        profile_id: [row for row in rows_by_profile[profile_id] if row["split"] == "validation"]
        for profile_id, _score in calibration_ranking
    }
    if all(validation_results.values()):
        validation_ranking = rank_candidate_profiles(validation_results, keep=8)
        selected_profile_id = validation_ranking[0][0]
    else:
        validation_ranking = ()
        selected_profile_id = calibration_ranking[0][0]
    selected_profile = profiles[selected_profile_id]
    restricted_affected = replay_manifest.get("purpose") == "restricted_affected_calibration"
    restricted_selection = None
    if restricted_affected:
        restricted_selection = select_restricted_affected_profile(
            rows_by_profile, baseline_id=str(replay_manifest["baseline_profile_id"])
        )
        selected_profile_id = str(restricted_selection["selected_profile_id"])
        selected_profile = profiles[selected_profile_id]
    selected_rows = rows_by_profile[selected_profile_id]
    calibrator: object | None = None
    calibrator_error: str | None = None
    baseline_only = replay_manifest.get("purpose") == "baseline_functional_pilot"
    if baseline_only or restricted_affected:
        calibrator_error = "limited_pilot_does_not_fit_confidence"
    else:
        try:
            calibrator = fit_confidence_calibrator(
                selected_rows,
                calibrator_id=f"part4-france-pilot-{campaign_id}",
            )
        except ValueError as exc:
            calibrator_error = str(exc)
    incidents = {row.case.incident_id for row in catalog_rows}
    calibration_incidents = {
        row.case.incident_id for row in catalog_rows if row.case.split == "calibration"
    }
    validation_incidents = {
        row.case.incident_id for row in catalog_rows if row.case.split == "validation"
    }
    leakage_count = sum(row["reference_leakage_detected"] is True for row in selected_rows)
    simulated_count = sum(row["simulated_contribution_detected"] is True for row in selected_rows)
    invalid_count = sum(row["geometry_valid"] is not True for row in selected_rows)
    metrics: dict[str, float | int] = {
        "selected_affected_objective": affected_objective(selected_rows),
        "median_iou_global": _median(selected_rows, "iou"),
        "median_boundary_f1": _median(selected_rows, "boundary_f1"),
        "median_uncertainty_boundary_coverage": _median(
            selected_rows,
            "uncertainty_boundary_coverage",
        ),
        "invalid_geometry_count": invalid_count,
        "reference_leakage_count": leakage_count,
        "simulated_contribution_count": simulated_count,
        "unique_geometry_evaluation_count": unique_evaluations,
        "geometry_evaluation_cache_hit_count": cache_hits,
        "geometry_comparison_count": geometry_comparisons,
        "geometry_comparison_cache_hit_count": comparison_cache_hits,
    }
    gates = {
        "minimum_incidents": len(incidents) >= 50,
        "minimum_snapshots": len(catalog_rows) >= 300,
        "calibration_split_present": bool(calibration_incidents),
        "validation_split_present": bool(validation_incidents),
        "valid_geometries": invalid_count == 0,
        "no_reference_leakage": leakage_count == 0,
        "no_simulated_contribution": simulated_count == 0,
        "confidence_calibrator_fitted": calibrator is not None,
        "parameter_screening_completed": not (baseline_only or restricted_affected),
        "code_revision_clean": not str(replay_manifest.get("code_revision", "")).endswith("-dirty"),
    }
    receipt = Part4PreHoldoutGateReceiptV1(
        receipt_id=f"P4-PREHOLDOUT-{campaign_id}",
        evaluated_at=datetime.now(UTC),
        campaign_id=campaign_id,
        calibration_dataset_repo=catalog_repo,
        calibration_dataset_revision=catalog_revision,
        split_id=catalog_rows[0].case.split_id,
        selected_fusion_profile_id=selected_profile.profile_id,
        selected_fusion_profile_sha256=selected_profile.profile_sha256,
        incident_count=len(incidents),
        snapshot_count=len(catalog_rows),
        calibration_incident_count=len(calibration_incidents),
        validation_incident_count=len(validation_incidents),
        confidence_calibrator_fitted=calibrator is not None,
        metrics=metrics,
        gates=gates,
        holdout_opened=False,
        qualified=False,
    )
    all_rows = [
        row for profile_id in sorted(rows_by_profile) for row in rows_by_profile[profile_id]
    ]
    rows_path = evaluation_root / "screening-evaluation.jsonl"
    ranking_path = evaluation_root / "screening-ranking.json"
    selected_path = evaluation_root / "selected-profile.json"
    receipt_path = evaluation_root / "pre-holdout-gates.json"
    _write_jsonl(rows_path, all_rows)
    _write_json(
        ranking_path,
        {
            "schema": "fireviewer.part4-screening-ranking.v1",
            "campaign_id": campaign_id,
            "purpose": replay_manifest.get("purpose", "parameter_screening"),
            "calibration": calibration_ranking,
            "validation": validation_ranking,
            "selected_profile_id": selected_profile_id,
            "confidence_calibrator_error": calibrator_error,
            "restricted_selection": restricted_selection,
        },
    )
    _write_json(selected_path, selected_profile)
    _write_json(receipt_path, receipt)
    artifacts = [
        (rows_path, f"receipts/{campaign_id}/evaluation/{rows_path.name}"),
        (ranking_path, f"receipts/{campaign_id}/evaluation/{ranking_path.name}"),
        (selected_path, f"receipts/{campaign_id}/evaluation/{selected_path.name}"),
        (receipt_path, f"receipts/{campaign_id}/evaluation/{receipt_path.name}"),
    ]
    if calibrator is not None:
        calibrator_path = evaluation_root / "confidence-calibrator.json"
        _write_json(calibrator_path, calibrator)
        artifacts.append(
            (
                calibrator_path,
                f"receipts/{campaign_id}/evaluation/{calibrator_path.name}",
            )
        )
    evaluation_revision = store.upload_artifacts(
        repo_id=repo_id,
        artifacts=artifacts,
        commit_message=f"Evaluate Part.4 screening campaign {campaign_id}",
    )
    return {
        "campaign_id": campaign_id,
        "evaluation_revision": evaluation_revision,
        "selected_profile_id": selected_profile_id,
        "selected_profile_sha256": selected_profile.profile_sha256,
        "incident_count": len(incidents),
        "snapshot_count": len(catalog_rows),
        "profile_count": len(profiles),
        "metrics": metrics,
        "gates": gates,
        "confidence_calibrator_fitted": calibrator is not None,
        "confidence_calibrator_error": calibrator_error,
        "restricted_selection": restricted_selection,
        "holdout_opened": False,
        "qualified": False,
    }


def screening_parameter_space() -> dict[str, tuple[float, ...]]:
    return {
        "weight_multipliers": WEIGHT_MULTIPLIERS,
        "affected_thresholds": AFFECTED_THRESHOLDS,
        "active_thresholds": ACTIVE_THRESHOLDS,
        "uncertainty_low_thresholds": UNCERTAINTY_LOW_THRESHOLDS,
        "uncertainty_high_thresholds": UNCERTAINTY_HIGH_THRESHOLDS,
        "active_half_life_hours": ACTIVE_HALF_LIFE_HOURS,
    }


__all__ = [
    "evaluate_screening_from_hf",
    "load_corpus_catalog",
    "replay_screening_to_hf",
    "screening_parameter_space",
]
