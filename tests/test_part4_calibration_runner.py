from __future__ import annotations

import json
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from fireviewer_contracts.backend.fire_state_schemas import SpatialObservationV2
from fireviewer_contracts.backend.part4_calibration_schemas import (
    HuggingFaceArtifactRefV1,
    Part4CalibrationCaseV1,
    Part4CorpusCatalogRowV1,
    Part4ObservationBundleV1,
    Part4ReferenceSnapshotV1,
)
from fireviewer_fire_state.fire_state_fusion import (
    BASELINE_ALGORITHM_VERSION as FUSION_ALGORITHM_VERSION,
)
from fireviewer_fire_state.fire_state_fusion import (
    fuse_probability_baseline as fuse_daily_fire_state,
)
from fireviewer_fire_state.part4_calibration_campaign import (
    CalibrationReplayPayload,
    HuggingFaceCampaignStore,
    ScratchBudgetExceeded,
    bounded_scratch,
    check_scratch_budget,
    ensure_calibration_repo,
    freeze_compact_replay_result,
    freeze_replay_result,
    load_replay_payload,
    replay_case,
    resolve_hf_revision,
    write_scratch_text,
)
from fireviewer_fire_state.part4_calibration_runner import (
    evaluate_screening_from_hf,
    load_corpus_catalog,
    replay_screening_to_hf,
)
from fireviewer_fire_state.part4_fusion_profiles import (
    load_fusion_profile,
)


def _polygon(minimum_x: float, maximum_x: float) -> dict[str, Any]:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [minimum_x, 44.70],
                [maximum_x, 44.70],
                [maximum_x, 44.71],
                [minimum_x, 44.71],
                [minimum_x, 44.70],
            ]
        ],
    }


class _FakeStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.files: dict[str, Path] = {}
        self.upload_count = 0
        self.downloaded_reference = False
        self.scratch = root / "scratch"
        self.incident_downloads: list[str] = []

    @contextmanager
    def incident_scope(self, incident_id: str) -> Iterator[_FakeStore]:
        previous = self.scratch
        self.scratch = self.root / "incident-work" / incident_id
        self.scratch.mkdir(parents=True, exist_ok=False)
        try:
            yield self
        except BaseException:
            raise
        else:
            shutil.rmtree(self.scratch)
        finally:
            self.scratch = previous

    def add(self, path: str, payload: object) -> HuggingFaceArtifactRefV1:
        target = self.root / "seed" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(payload, "model_dump"):
            payload = payload.model_dump(mode="json", by_alias=True)
        target.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
        self.files[path] = target
        return HuggingFaceArtifactRefV1(
            repo_id="fireviewer/test-calibration",
            repo_type="dataset",
            revision="1" * 40,
            path=path,
            byte_count=target.stat().st_size,
        )

    def download(self, ref: HuggingFaceArtifactRefV1) -> Path:
        self.incident_downloads.append(ref.path)
        if ref.path.startswith("references/"):
            self.downloaded_reference = True
        target = self.files[ref.path]
        assert target.stat().st_size == ref.byte_count
        return target

    def artifact_ref(
        self, *, repo_id: str, repo_type: str, revision: str, path: str
    ) -> HuggingFaceArtifactRefV1:
        return HuggingFaceArtifactRefV1(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path=path,
            byte_count=self.files[path].stat().st_size,
        )

    def upload_artifacts(
        self,
        *,
        repo_id: str,
        artifacts: list[tuple[Path, str]],
        commit_message: str,
        maximum_operations: int = 200,
    ) -> str:
        assert repo_id == "fireviewer/test-calibration"
        assert commit_message
        assert maximum_operations > 0
        self.upload_count += 1
        for source, remote in artifacts:
            target = self.root / "remote" / remote
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            self.files[remote] = target
        return str(self.upload_count + 1) * 40


def _catalog_case(
    store: _FakeStore,
    *,
    suffix: str,
    split: str,
    observation_geometry: dict[str, Any],
    reference_geometry: dict[str, Any],
    observation_resolution_m: float = 20.0,
) -> Part4CorpusCatalogRowV1:
    observed_at = datetime(2026, 7, 7, 12, tzinfo=UTC)
    incident_id = f"INC-{suffix}"
    case_id = f"CASE-{suffix}"
    observation = SpatialObservationV2(
        observation_id=f"OBS-{suffix}",
        observation_kind="burned_probability",
        target_state="affected",
        observed_at=observed_at,
        geometry_geojson=observation_geometry,
        coverage_geojson=observation_geometry,
        probability=0.95,
        observability_probability=1.0,
        resolution_m=observation_resolution_m,
        horizontal_accuracy_m=20.0,
        source_family_id=f"sentinel-2-{suffix}",
        lineage_id=f"sentinel-2-result-{suffix}",
        upstream_product_id=f"sentinel-2-product-{suffix}",
        source_revision_sha256="a" * 64,
        processor_revision="sentinel2-nbr-test-v1",
        evidence_refs=(f"source:{suffix}",),
    )
    bundle = Part4ObservationBundleV1(
        case_id=case_id,
        incident_id=incident_id,
        evaluation_cutoff_at=observed_at,
        location_source_refs=(f"location:{suffix}",),
        observations=(observation,),
    )
    observation_ref = store.add(f"observations/{case_id}.json", bundle)
    reference = Part4ReferenceSnapshotV1(
        reference_id=f"REF-{suffix}",
        incident_id=incident_id,
        component="affected",
        geometry_geojson=reference_geometry,
        valid_at=observed_at,
        temporal_accuracy_seconds=3_600,
        spatial_accuracy_m=20.0,
        resolution_m=20.0,
        provider="Test authority",
        product="Test perimeter",
        licence="Test-only fixture",
        source_revision=f"reference-{suffix}",
        grade="A",
        forbidden_input_refs=(f"REF-{suffix}",),
    )
    reference_ref = store.add(f"references/{case_id}.json", reference)
    case = Part4CalibrationCaseV1(
        case_id=case_id,
        incident_id=incident_id,
        evaluation_cutoff_at=observed_at,
        latest_input_at=observed_at,
        observations=observation_ref,
        split=split,
        split_id="test-incident-split-v1",
        reference_ids=(reference.reference_id,),
        forbidden_input_refs=reference.forbidden_input_refs,
    )
    return Part4CorpusCatalogRowV1(
        case=case,
        reference_id=reference.reference_id,
        reference=reference_ref,
        location_source_refs=(f"location:{suffix}",),
    )


@pytest.mark.parametrize("manifest_version", ["v1", "v2"])
@pytest.mark.parametrize("mode", ["screening", "baseline", "restricted"])
def test_hf_screening_freezes_before_reference_and_keeps_holdout_sealed(
    tmp_path: Path,
    monkeypatch: Any,
    manifest_version: str,
    mode: str,
) -> None:
    store = _FakeStore(tmp_path)
    rows = (
        _catalog_case(
            store,
            suffix="GOOD",
            split="calibration",
            observation_geometry=_polygon(5.10, 5.11),
            reference_geometry=_polygon(5.10, 5.11),
        ),
        _catalog_case(
            store,
            suffix="BAD",
            split="validation",
            observation_geometry=_polygon(5.20, 5.21),
            reference_geometry=_polygon(5.30, 5.31),
        ),
    )
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text(
        "\n".join(
            json.dumps(row.model_dump(mode="json", by_alias=True), sort_keys=True) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_corpus_catalog(catalog_path) == rows
    profile = load_fusion_profile(
            "part4-baseline-v3-provenance",
            algorithm_version=FUSION_ALGORITHM_VERSION,
    )
    from fireviewer_fire_state.part4_calibration import build_screening_profiles

    profiles = build_screening_profiles(profile)[:2]
    monkeypatch.setattr(
        "fireviewer_fire_state.part4_calibration_runner.build_screening_profiles",
        lambda _profile: profiles,
    )
    replay = replay_screening_to_hf(
        store=store,  # type: ignore[arg-type]
        repo_id="fireviewer/test-calibration",
        catalog_repo="fireviewer/test-calibration",
        catalog_revision="c" * 40,
        catalog_rows=rows,
        base_profile=profile,
        campaign_id="test-real-pilot",
        code_revision="d" * 40,
        scratch=tmp_path / "replay-scratch",
        baseline_only=mode == "baseline",
        restricted_affected=mode == "restricted",
        replay_workers=2 if mode == "restricted" else 1,
    )
    assert replay["profile_count"] == {"baseline": 1, "screening": 2, "restricted": 5}[mode]
    assert replay["references_opened"] is False
    assert replay["schema"].endswith(".v2")
    assert len(replay["incidents"]) == 2
    assert len([path for path in store.incident_downloads if path.startswith("observations/")]) == 2
    assert not list((tmp_path / "incident-work").iterdir())
    assert store.downloaded_reference is False
    if manifest_version == "v1":
        legacy_profiles = []
        for profile_entry in replay["profiles"]:
            parts = [
                shard
                for incident in replay["incidents"]
                for shard in incident["profiles"]
                if shard["profile_id"] == profile_entry["profile_id"]
            ]
            remote_path = f"legacy/{profile_entry['profile_id']}.jsonl"
            combined = tmp_path / f"{profile_entry['profile_id']}-legacy.jsonl"
            combined.write_text(
                "".join(
                    store.files[part["frozen_path"]].read_text(encoding="utf-8") for part in parts
                ),
                encoding="utf-8",
            )
            store.files[remote_path] = combined
            legacy_profiles.append(
                {
                    **profile_entry,
                    "frozen_path": remote_path,
                    "frozen_byte_count": combined.stat().st_size,
                }
            )
        replay = {
            **replay,
            "schema": "fireviewer.part4-screening-replay-manifest.v1",
            "profiles": legacy_profiles,
        }
    evaluated = evaluate_screening_from_hf(
        store=store,  # type: ignore[arg-type]
        repo_id="fireviewer/test-calibration",
        catalog_repo="fireviewer/test-calibration",
        catalog_revision="c" * 40,
        catalog_rows=rows,
        replay_manifest=replay,
        scratch=tmp_path / "evaluation-scratch",
    )
    assert store.downloaded_reference is True
    assert evaluated["incident_count"] == 2
    assert evaluated["snapshot_count"] == 2
    assert evaluated["confidence_calibrator_fitted"] is (mode == "screening")
    assert evaluated["gates"]["parameter_screening_completed"] is (mode == "screening")
    has_calibrator = any(path.endswith("confidence-calibrator.json") for path in store.files)
    assert has_calibrator is (mode == "screening")
    if mode == "restricted":
        assert evaluated["restricted_selection"]["qualification_allowed"] is False
        assert evaluated["metrics"]["geometry_comparison_cache_hit_count"] > 0
    assert evaluated["holdout_opened"] is False
    assert evaluated["qualified"] is False
    assert evaluated["gates"]["minimum_incidents"] is False
    assert evaluated["gates"]["minimum_snapshots"] is False


def test_bounded_parallel_replays_equal_serial_json_and_preserve_inputs(tmp_path: Path) -> None:
    from fireviewer_fire_state.part4_calibration import build_restricted_affected_profiles
    from fireviewer_fire_state.part4_calibration_runner import (
        _download_observation_bundle,
        _replay_profiles,
    )

    store = _FakeStore(tmp_path)
    row = _catalog_case(
        store, suffix="PARALLEL", split="calibration",
        observation_geometry=_polygon(5.10, 5.11), reference_geometry=_polygon(5.10, 5.11),
    )
    bundle = _download_observation_bundle(store, row.case)  # type: ignore[arg-type]
    unchanged = bundle.model_dump_json()
    profiles = build_restricted_affected_profiles(load_fusion_profile(
        "part4-baseline-v3-provenance", algorithm_version=FUSION_ALGORITHM_VERSION
    ))
    bundles = {row.case.case_id: bundle}
    serial = list(_replay_profiles(profiles, (row,), bundles, workers=1))
    parallel = list(_replay_profiles(profiles, (row,), bundles, workers=4))
    assert parallel == serial
    assert bundle.model_dump_json() == unchanged
    assert store.downloaded_reference is False


def test_frozen_sensor_resolution_is_not_inferred_from_grid_or_reference(tmp_path: Path) -> None:
    store = _FakeStore(tmp_path)
    row = _catalog_case(
        store, suffix="COARSE", split="calibration", observation_resolution_m=375.0,
        observation_geometry=_polygon(5.10, 5.11), reference_geometry=_polygon(5.10, 5.11),
    )
    profile = load_fusion_profile(
        "part4-baseline-v3-provenance", algorithm_version=FUSION_ALGORITHM_VERSION
    )
    manifest = replay_screening_to_hf(
        store=store, repo_id="fireviewer/test-calibration",  # type: ignore[arg-type]
        catalog_repo="fireviewer/test-calibration", catalog_revision="c" * 40,
        catalog_rows=(row,), base_profile=profile, campaign_id="coarse-resolution",
        code_revision="d" * 40, scratch=tmp_path / "replays", baseline_only=True,
    )
    evaluate_screening_from_hf(
        store=store, repo_id="fireviewer/test-calibration",  # type: ignore[arg-type]
        catalog_repo="fireviewer/test-calibration", catalog_revision="c" * 40,
        catalog_rows=(row,), replay_manifest=manifest, scratch=tmp_path / "evaluations",
    )
    result = json.loads(store.files[
        "receipts/coarse-resolution/evaluation/screening-evaluation.jsonl"
    ].read_text())
    assert result["grid_resolution_m"] == 20.0
    assert result["best_observation_resolution_m"] == 375.0
    assert result["boundary_tolerance_m"] == 375.0


def test_real_hf_store_adapter_verifies_uploads_and_revisions(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    remote_root = tmp_path / "fake-hf"
    remote_root.mkdir()

    class FakeApi:
        revisions = 0

        def __init__(self, *, token: str | None = None) -> None:
            self.token = token

        def dataset_info(self, _repo_id: str) -> Any:
            return SimpleNamespace(sha="d" * 40)

        def model_info(self, _repo_id: str) -> Any:
            return SimpleNamespace(sha="m" * 40)

        def create_commit(self, **kwargs: Any) -> Any:
            assert kwargs["repo_type"] == "dataset"
            assert kwargs["commit_message"]
            for operation in kwargs["operations"]:
                destination = remote_root / operation.path_in_repo
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(operation.path_or_fileobj), destination)
            type(self).revisions += 1
            return SimpleNamespace(oid=str(type(self).revisions + 1) * 40)

        def list_repo_tree(self, **_kwargs: Any) -> list[Any]:
            return [
                SimpleNamespace(
                    rfilename=path.relative_to(remote_root).as_posix(),
                    size=path.stat().st_size,
                )
                for path in remote_root.rglob("*")
                if path.is_file()
            ]

        def get_paths_info(self, **kwargs: Any) -> list[Any]:
            return [item for item in self.list_repo_tree() if item.rfilename in kwargs["paths"]]

    def fake_download(**kwargs: Any) -> str:
        source = remote_root / str(kwargs["filename"])
        assert source.is_file()
        destination = Path(kwargs["local_dir"]) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return str(destination)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    seed = remote_root / "observations" / "seed.json"
    seed.parent.mkdir(parents=True)
    seed.write_text('{"ok":true}\n', encoding="utf-8")
    store = HuggingFaceCampaignStore(token="test-token", scratch=tmp_path / "scratch")
    ref = store.artifact_ref(
        repo_id="fireviewer/test-calibration",
        repo_type="dataset",
        revision="d" * 40,
        path="observations/seed.json",
    )
    assert store.download(ref).read_text(encoding="utf-8") == '{"ok":true}\n'
    local_a = tmp_path / "a.json"
    local_b = tmp_path / "b.json"
    local_a.write_text('{"a":1}\n', encoding="utf-8")
    local_b.write_text('{"b":2}\n', encoding="utf-8")
    revision = store.upload_artifacts(
        repo_id="fireviewer/test-calibration",
        artifacts=[(local_a, "results/a.json"), (local_b, "results/b.json")],
        commit_message="upload verified results",
        maximum_operations=1,
    )
    assert len(revision) == 40
    incident_revision = store.upload_incident_results(
        repo_id="fireviewer/test-calibration",
        files=[local_a],
        path_prefix="receipts/incident-1",
        commit_message="upload incident result",
    )
    assert len(incident_revision) == 40
    assert (
        resolve_hf_revision(
            "fireviewer/test-calibration",
            repo_type="dataset",
            token="test-token",
        )
        == "d" * 40
    )
    assert (
        resolve_hf_revision(
            "fireviewer/test-model",
            repo_type="model",
            token="test-token",
        )
        == "m" * 40
    )
    with pytest.raises(ValueError, match="dataset or model"):
        resolve_hf_revision("fireviewer/test", repo_type="space", token="test-token")
    with pytest.raises(RuntimeError, match="missing or ambiguous"):
        store.artifact_ref(
            repo_id="fireviewer/test-calibration",
            repo_type="dataset",
            revision="d" * 40,
            path="missing.json",
        )
    with pytest.raises(ValueError, match="cannot be empty"):
        store.upload_artifacts(
            repo_id="fireviewer/test-calibration",
            artifacts=[],
            commit_message="empty",
        )
    with pytest.raises(ValueError, match="positive"):
        store.upload_artifacts(
            repo_id="fireviewer/test-calibration",
            artifacts=[(local_a, "results/a.json")],
            commit_message="invalid chunk",
            maximum_operations=0,
        )
    ensure_calibration_repo("fireviewer/test-calibration")
    with pytest.raises(ValueError, match="holdout"):
        ensure_calibration_repo("fireviewer/part4-france-holdout-v1")


def test_compact_replay_freeze_records_grid_support(tmp_path: Path) -> None:
    profile = load_fusion_profile(
            "part4-baseline-v3-provenance",
            algorithm_version=FUSION_ALGORITHM_VERSION,
    )
    observation = SpatialObservationV2(
        observation_id="OBS-COMPACT",
        observation_kind="burned_probability",
        target_state="affected",
        observed_at=datetime(2026, 7, 7, 12, tzinfo=UTC),
        geometry_geojson=_polygon(5.10, 5.11),
        coverage_geojson=_polygon(5.10, 5.11),
        probability=0.95,
        observability_probability=1.0,
        resolution_m=20.0,
        horizontal_accuracy_m=20.0,
        source_family_id="sentinel-2-compact",
        lineage_id="sentinel-2-compact-result",
        upstream_product_id="sentinel-2-compact-product",
        source_revision_sha256="b" * 64,
        processor_revision="sentinel2-nbr-test-v1",
        evidence_refs=("source:compact",),
    )
    result = fuse_daily_fire_state(
        incident_id="INC-COMPACT",
        episode_id=None,
        local_date=datetime(2026, 7, 7, tzinfo=UTC).date(),
        observations=(observation,),
        profile=profile,
    )
    state_path, support_path = freeze_compact_replay_result(
        result,
        output_dir=tmp_path,
        case_id="CASE-COMPACT",
    )
    assert json.loads(state_path.read_text(encoding="utf-8"))["incident_id"] == ("INC-COMPACT")
    support = json.loads(support_path.read_text(encoding="utf-8"))
    assert support["schema"] == "fireviewer.part4-frozen-grid-support.v1"
    assert 0 < support["observable_fraction"] <= 1
    assert 0 <= support["uncertainty_fraction"] <= 1


def test_replay_payload_guards_future_leakage_and_holdout(tmp_path: Path) -> None:
    observed_at = datetime(2026, 7, 7, 12, tzinfo=UTC)
    observation = SpatialObservationV2(
        observation_id="OBS-PAYLOAD",
        observation_kind="burned_probability",
        target_state="affected",
        observed_at=observed_at,
        geometry_geojson=_polygon(5.10, 5.11),
        coverage_geojson=_polygon(5.10, 5.11),
        probability=0.9,
        observability_probability=1.0,
        resolution_m=20.0,
        horizontal_accuracy_m=20.0,
        source_family_id="sentinel-2-payload",
        lineage_id="sentinel-2-payload-result",
        upstream_product_id="sentinel-2-payload-product",
        source_revision_sha256="e" * 64,
        processor_revision="sentinel2-nbr-test-v1",
        evidence_refs=("source:payload",),
    )
    artifact = HuggingFaceArtifactRefV1(
        repo_id="fireviewer/test-calibration",
        repo_type="dataset",
        revision="f" * 40,
        path="observations/payload.json",
        byte_count=1,
    )
    case = Part4CalibrationCaseV1(
        case_id="CASE-PAYLOAD",
        incident_id="INC-PAYLOAD",
        evaluation_cutoff_at=observed_at,
        latest_input_at=observed_at,
        observations=artifact,
        split="calibration",
        split_id="test-split-v1",
        reference_ids=("REF-PAYLOAD",),
        forbidden_input_refs=("REF-PAYLOAD",),
    )
    path = tmp_path / "payload.json"

    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_replay_payload(path)

    base = {
        "case": case.model_dump(mode="json"),
        "observations": [observation.model_dump(mode="json")],
    }
    path.write_text(json.dumps({**base, "observations": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="requires observations"):
        load_replay_payload(path)

    future = observation.model_copy(update={"observed_at": observed_at.replace(hour=13)})
    path.write_text(
        json.dumps({**base, "observations": [future.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unavailable at the evaluation cutoff"):
        load_replay_payload(path)

    leaked = observation.model_copy(update={"evidence_refs": ("REF-PAYLOAD",)})
    path.write_text(
        json.dumps({**base, "observations": [leaked.model_dump(mode="json")]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="leaked into replay"):
        load_replay_payload(path)

    path.write_text(
        json.dumps({**base, "prior_observed_at": "2026-07-07T10:00:00"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        load_replay_payload(path)

    path.write_text(
        json.dumps({**base, "prior_observed_at": "2026-07-07T10:00:00+00:00"}),
        encoding="utf-8",
    )
    payload = load_replay_payload(path)
    assert payload.prior_observed_at == datetime(2026, 7, 7, 10, tzinfo=UTC)

    profile = load_fusion_profile(
            "part4-baseline-v3-provenance",
            algorithm_version=FUSION_ALGORITHM_VERSION,
    )
    holdout_case = case.model_copy(update={"split": "holdout"})
    with pytest.raises(ValueError, match="cannot open holdout"):
        replay_case(
            CalibrationReplayPayload(
                case=holdout_case,
                observations=(observation,),
            ),
            profile=profile,
        )


def test_full_replay_freeze_accepts_unmaterialized_state(tmp_path: Path) -> None:
    profile = load_fusion_profile(
            "part4-baseline-v3-provenance",
            algorithm_version=FUSION_ALGORITHM_VERSION,
    )
    result = fuse_daily_fire_state(
        incident_id="INC-EMPTY",
        episode_id=None,
        local_date=datetime(2026, 7, 7, tzinfo=UTC).date(),
        observations=(),
        profile=profile,
    )
    state_path, grid_path = freeze_replay_result(
        result,
        output_dir=tmp_path,
        case_id="CASE-EMPTY",
    )
    assert state_path.is_file()
    assert grid_path is None


def test_scratch_budget_refuses_write_and_restores_hf_environment(
    tmp_path: Path, monkeypatch: Any
) -> None:
    import os

    from huggingface_hub import constants

    original_cache = constants.HF_HUB_CACHE
    monkeypatch.setenv("HF_HOME", "original-hf-home")
    monkeypatch.setattr(
        "fireviewer_fire_state.part4_calibration_campaign.scratch_budget_bytes", lambda _path: 4096
    )
    with bounded_scratch(parent=tmp_path) as scratch:
        assert os.environ["HF_HOME"].startswith(str(scratch))
        assert constants.HF_HUB_CACHE.startswith(str(scratch))
        assert constants.HF_XET_CACHE.startswith(str(scratch))
        write_scratch_text(scratch / "small.json", "{}")
        with pytest.raises(ScratchBudgetExceeded):
            write_scratch_text(scratch / "oversized.json", "x" * 4096)
        assert not (scratch / "oversized.json").exists()
        assert check_scratch_budget(scratch) < 4096
    assert not scratch.exists()
    assert os.environ["HF_HOME"] == "original-hf-home"
    assert original_cache == constants.HF_HUB_CACHE


def test_failed_job_preserves_unverified_outputs(tmp_path: Path) -> None:
    with (
        pytest.raises(RuntimeError, match="remote read failed"),
        bounded_scratch(parent=tmp_path) as scratch,
    ):
        write_scratch_text(scratch / "unverified-result.json", '{"result":1}')
        raise RuntimeError("remote read failed")
    assert (scratch / "unverified-result.json").is_file()


def test_incident_scope_does_not_delete_outputs_when_upload_fails(tmp_path: Path) -> None:
    store = object.__new__(HuggingFaceCampaignStore)
    store._scratch = tmp_path
    store._api, store._download, store._token = None, None, None
    with pytest.raises(RuntimeError), store.incident_scope("INC-FAILED") as child:
        write_scratch_text(child.scratch / "result.json", "{}")
        raise RuntimeError("upload failed")
    assert (child.scratch / "result.json").is_file()


def test_download_is_refused_before_network_when_budget_is_insufficient(
    tmp_path: Path, monkeypatch: Any
) -> None:
    monkeypatch.setattr(
        "fireviewer_fire_state.part4_calibration_campaign.scratch_budget_bytes", lambda _path: 4096
    )
    with bounded_scratch(parent=tmp_path) as scratch:
        store = object.__new__(HuggingFaceCampaignStore)
        store._scratch = scratch
        ref = HuggingFaceArtifactRefV1(
            repo_id="fireviewer/test-calibration",
            repo_type="dataset",
            revision="a" * 40,
            path="huge.tif",
            byte_count=8192,
        )
        with pytest.raises(ScratchBudgetExceeded):
            store.download(ref)
