"""Streaming, incident-scoped Part.4 calibration replay helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import numpy as np

from fireviewer_contracts.backend.fire_state_schemas import FusionProfileV1, SpatialObservationV2
from fireviewer_contracts.backend.hashing import sha256_hex
from fireviewer_contracts.backend.part4_calibration_schemas import (
    HuggingFaceArtifactRefV1,
    Part4CalibrationCaseV1,
)
from fireviewer_fire_state.fire_state_fusion import FusedFireState, fuse_probability_baseline

MAX_SCRATCH_BYTES = 20 * 1024**3


class _RedactHubSignedUrls(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # Hub retries can otherwise expose short-lived, signed S3 upload URLs.
        record.msg = re.sub(r"(https?://[^\s?]+)\?[^\s]+", r"\1?[redacted]", record.getMessage())
        record.args = ()
        return True


class ScratchBudgetExceeded(RuntimeError):
    """Raised before admitting data that cannot fit in the job's fixed budget."""


def check_scratch_budget(path: Path, *, additional_bytes: int = 0) -> int:
    """Account for the entire job, including sibling incidents and HF caches."""

    if additional_bytes < 0:
        raise ValueError("scratch reservation cannot be negative")
    resolved = path.resolve()
    root = resolved if resolved.is_dir() else resolved.parent
    for candidate in (root, *root.parents):
        policy = candidate / "scratch-policy.json"
        if policy.is_file():
            budget = int(json.loads(policy.read_text(encoding="utf-8"))["budget_bytes"])
            used = sum(item.stat().st_size for item in candidate.rglob("*") if item.is_file())
            if used + additional_bytes > budget:
                raise ScratchBudgetExceeded(
                    f"Part.4 scratch budget exceeded: {used}+{additional_bytes}>{budget}"
                )
            return used
    return 0


def write_scratch_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    previous_size = path.stat().st_size if path.is_file() else 0
    check_scratch_budget(path, additional_bytes=max(0, len(encoded) - previous_size))
    path.write_bytes(encoded)


def _verify_representative(path: Path) -> None:
    """Decode one downloaded sample, not every remote object or a second full corpus."""

    if path.suffix.lower() in {".tif", ".tiff"}:
        import rasterio  # type: ignore[import-untyped]
        from rasterio.windows import Window  # type: ignore[import-untyped]

        with rasterio.open(path) as dataset:
            if not dataset.crs or dataset.count < 1:
                raise RuntimeError("HF representative raster lacks spatial metadata")
            dataset.read(1, window=Window(0, 0, min(16, dataset.width), min(16, dataset.height)))
    elif path.suffix.lower() in {".json", ".jsonl"}:
        with path.open(encoding="utf-8") as stream:
            if path.suffix.lower() == ".jsonl":
                first = next((line for line in stream if line.strip()), "")
                json.loads(first)
            else:
                json.load(stream)


@dataclass(frozen=True, slots=True)
class CalibrationReplayPayload:
    case: Part4CalibrationCaseV1
    observations: tuple[SpatialObservationV2, ...]
    prior_affected: dict[str, Any] | None = None
    prior_active: dict[str, Any] | None = None
    prior_observed_at: datetime | None = None
    prior_state_sha256: str | None = None


def scratch_budget_bytes(path: Path) -> int:
    """Bound a job scratch directory to 20 GiB or ten percent of free space."""

    free = shutil.disk_usage(path).free
    return max(1, min(MAX_SCRATCH_BYTES, free // 10))


@contextmanager
def bounded_scratch(*, parent: Path | None = None) -> Iterator[Path]:
    """Clean successful jobs; retain bounded failed outputs for recovery."""

    # CLI jobs may resolve authentication before entering this scope. The pinned
    # Hub SDK caches these paths at import time, so environment changes alone do
    # not isolate an already imported SDK. Authentication paths are left intact.
    try:
        sdk_constants = import_module("huggingface_hub.constants")
    except ImportError:
        sdk_constants = None
    root = parent or Path(tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="fireviewer-part4-", dir=root))
    budget = scratch_budget_bytes(root)
    (scratch / "scratch-policy.json").write_text(
        json.dumps(
            {
                "schema": "fireviewer.part4-scratch-policy.v1",
                "budget_bytes": budget,
                "cleanup": "after_remote_read_check",
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    names = {"HF_HOME": "hf-home", "HF_HUB_CACHE": "hf-hub", "HF_XET_CACHE": "hf-xet"}
    previous = {name: os.environ.get(name) for name in names}
    previous_sdk = {name: getattr(sdk_constants, name) for name in names} if sdk_constants else {}
    for name, directory in names.items():
        os.environ[name] = str(scratch / directory)
        if sdk_constants:
            setattr(sdk_constants, name, os.environ[name])
    hub_loggers = [
        logging.getLogger(f"huggingface_hub.{name}")
        for name in ("utils._http", "lfs", "file_download", "_commit_api")
    ]
    redaction = _RedactHubSignedUrls()
    for logger in hub_loggers:
        logger.addFilter(redaction)
    try:
        yield scratch
    except BaseException as exc:
        # Never discard the only copy of outputs after a failed remote commit/read check.
        exc.add_note(f"Bounded Part.4 scratch retained for recovery: {scratch}")
        raise
    else:
        shutil.rmtree(scratch)
    finally:
        for logger in hub_loggers:
            logger.removeFilter(redaction)
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        if sdk_constants:
            for name, value in previous_sdk.items():
                setattr(sdk_constants, name, value)


def load_replay_payload(path: Path) -> CalibrationReplayPayload:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("calibration replay payload must be a JSON object")
    case = Part4CalibrationCaseV1.model_validate(payload.get("case"))
    observations = tuple(
        SpatialObservationV2.model_validate(item) for item in payload.get("observations", ())
    )
    if not observations:
        raise ValueError("calibration replay payload requires observations")
    forbidden = set(case.forbidden_input_refs)
    for observation in observations:
        if observation.observed_at > case.evaluation_cutoff_at:
            raise ValueError("observation was unavailable at the evaluation cutoff")
        exposed = {
            observation.observation_id,
            observation.upstream_product_id,
            *observation.evidence_refs,
        }
        if forbidden.intersection(exposed):
            raise ValueError("reference or reference-derived input leaked into replay observations")
    prior_observed_at = payload.get("prior_observed_at")
    parsed_prior_at = datetime.fromisoformat(prior_observed_at) if prior_observed_at else None
    if parsed_prior_at is not None and (
        parsed_prior_at.tzinfo is None or parsed_prior_at.utcoffset() is None
    ):
        raise ValueError("prior observation timestamp must be timezone-aware")
    return CalibrationReplayPayload(
        case=case,
        observations=observations,
        prior_affected=payload.get("prior_affected"),
        prior_active=payload.get("prior_active"),
        prior_observed_at=parsed_prior_at,
        prior_state_sha256=payload.get("prior_state_sha256"),
    )


def replay_case(
    payload: CalibrationReplayPayload,
    *,
    profile: FusionProfileV1,
) -> FusedFireState:
    if payload.case.split == "holdout":
        raise ValueError("the calibration replay process cannot open holdout cases")
    return fuse_probability_baseline(
        incident_id=payload.case.incident_id,
        episode_id=payload.case.episode_id,
        local_date=payload.case.evaluation_cutoff_at.date(),
        observations=payload.observations,
        prior_affected=payload.prior_affected,
        prior_active=payload.prior_active,
        prior_observed_at=payload.prior_observed_at,
        prior_state_sha256=payload.prior_state_sha256,
        profile=profile,
    )


def freeze_replay_result(
    result: FusedFireState,
    *,
    output_dir: Path,
    case_id: str,
) -> tuple[Path, Path | None]:
    """Write one compact state and optional probability grids for immediate upload."""

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{case_id}.json"
    write_scratch_text(
        state_path,
        json.dumps(result.state.model_dump(mode="json", by_alias=True), sort_keys=True) + "\n",
    )
    arrays = (
        result.affected_probability,
        result.active_probability,
        result.observable_probability,
        result.observed_support,
        result.multi_source_support,
        result.prior_interpolated_support,
        result.uncertainty_support,
    )
    if any(item is None for item in arrays):
        return state_path, None
    affected = cast(np.ndarray[Any, np.dtype[np.float32]], result.affected_probability)
    active = cast(np.ndarray[Any, np.dtype[np.float32]], result.active_probability)
    observable = cast(np.ndarray[Any, np.dtype[np.float32]], result.observable_probability)
    observed = cast(np.ndarray[Any, np.dtype[np.float32]], result.observed_support)
    multi_source = cast(np.ndarray[Any, np.dtype[np.float32]], result.multi_source_support)
    interpolated = cast(np.ndarray[Any, np.dtype[np.float32]], result.prior_interpolated_support)
    uncertainty = cast(np.ndarray[Any, np.dtype[np.float32]], result.uncertainty_support)
    transform = result.transform
    if transform is None:
        raise ValueError("a materialized Part.4 grid requires a transform")
    grid_path = output_dir / f"{case_id}.npz"
    check_scratch_budget(
        output_dir, additional_bytes=sum(a.nbytes for a in arrays if a is not None) * 2 + 16_384
    )
    np.savez_compressed(
        grid_path,
        affected=affected,
        active=active,
        observable=observable,
        observed_support=observed,
        multi_source_support=multi_source,
        prior_interpolated_support=interpolated,
        uncertainty_support=uncertainty,
        transform=np.asarray(tuple(transform)[:6], dtype=np.float64),
    )
    check_scratch_budget(output_dir)
    return state_path, grid_path


def freeze_compact_replay_result(
    result: FusedFireState,
    *,
    output_dir: Path,
    case_id: str,
) -> tuple[Path, Path]:
    """Freeze geometry and aggregate grid support without retaining large arrays."""

    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{case_id}.json"
    write_scratch_text(
        state_path,
        json.dumps(result.state.model_dump(mode="json", by_alias=True), sort_keys=True) + "\n",
    )
    observable = result.observable_probability
    uncertainty = result.uncertainty_support
    support_path = output_dir / f"{case_id}.support.json"
    write_scratch_text(
        support_path,
        json.dumps(
            {
                "schema": "fireviewer.part4-frozen-grid-support.v1",
                "observable_fraction": (
                    float(np.mean(np.clip(observable, 0.0, 1.0))) if observable is not None else 0.0
                ),
                "uncertainty_fraction": (
                    float(np.mean(uncertainty > 0)) if uncertainty is not None else 0.0
                ),
            },
            sort_keys=True,
        )
        + "\n",
    )
    return state_path, support_path


class HuggingFaceCampaignStore:
    """Thin lazy adapter so production backend images do not require HF Hub."""

    def __init__(self, *, token: str | None = None, scratch: Path) -> None:
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("install the backend calibration extra") from exc
        self._api = HfApi(token=token)
        self._download = hf_hub_download
        self._token = token
        self._scratch = scratch

    @property
    def scratch(self) -> Path:
        return self._scratch

    @contextmanager
    def incident_scope(self, incident_id: str) -> Iterator[HuggingFaceCampaignStore]:
        """Share credentials, never downloads, across consecutive incident batches."""

        directory = self._scratch / "incidents" / sha256_hex(incident_id)[:24]
        directory.mkdir(parents=True, exist_ok=False)
        child = object.__new__(HuggingFaceCampaignStore)
        child._api, child._download = self._api, self._download
        child._token, child._scratch = self._token, directory
        try:
            yield child
        except BaseException:
            raise
        else:
            shutil.rmtree(directory)

    def download(self, ref: HuggingFaceArtifactRefV1) -> Path:
        check_scratch_budget(self._scratch, additional_bytes=ref.byte_count + 16_384)
        # Validate remote size before hf_hub_download can materialize an object.
        remote = self.artifact_ref(
            repo_id=ref.repo_id, repo_type=ref.repo_type, revision=ref.revision, path=ref.path
        )
        if remote.byte_count != ref.byte_count:
            raise RuntimeError("remote HF artifact does not match its declared size")
        target = self._download(
            repo_id=ref.repo_id,
            repo_type=ref.repo_type,
            revision=ref.revision,
            filename=ref.path,
            token=self._token,
            cache_dir=self._scratch / "hf-cache",
            local_dir=self._scratch / "downloads" / ref.repo_id.replace("/", "--"),
        )
        path = Path(target)
        if not path.is_file() or path.stat().st_size != ref.byte_count:
            raise RuntimeError("downloaded HF artifact does not match its declared size")
        check_scratch_budget(self._scratch)
        return path

    def upload_incident_results(
        self,
        *,
        repo_id: str,
        files: Sequence[Path],
        path_prefix: str,
        commit_message: str,
    ) -> str:
        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(
                path_in_repo=f"{path_prefix.rstrip('/')}/{path.name}",
                path_or_fileobj=str(path),
            )
            for path in files
        ]
        commit = self._api.create_commit(
            repo_id=repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=commit_message,
        )
        revision = str(commit.oid)
        if len(revision) < 7:
            raise RuntimeError("HF upload did not return an immutable revision")
        listed = {
            item.rfilename: int(getattr(item, "size", 0) or 0)
            for item in self._api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                recursive=True,
                expand=True,
            )
            if hasattr(item, "rfilename")
        }
        for path in files:
            remote = f"{path_prefix.rstrip('/')}/{path.name}"
            if listed.get(remote) != path.stat().st_size:
                raise RuntimeError(f"HF result verification failed for {remote}")
        sample = files[0]
        check_scratch_budget(self._scratch, additional_bytes=sample.stat().st_size + 16_384)
        sample_remote = f"{path_prefix.rstrip('/')}/{sample.name}"
        checked = Path(
            self._download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=sample_remote,
                token=self._token,
                cache_dir=self._scratch / "hf-cache",
                local_dir=self._scratch / "remote-check",
            )
        )
        if not checked.is_file() or checked.stat().st_size != sample.stat().st_size:
            raise RuntimeError("HF representative result could not be read after upload")
        _verify_representative(checked)
        check_scratch_budget(self._scratch)
        return revision

    def artifact_ref(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        path: str,
    ) -> HuggingFaceArtifactRefV1:
        result = self.find_artifact_ref(
            repo_id=repo_id, repo_type=repo_type, revision=revision, path=path
        )
        if result is None:
            raise RuntimeError(f"HF artifact is missing or ambiguous: {path}")
        return result

    def find_artifact_ref(
        self,
        *,
        repo_id: str,
        repo_type: str,
        revision: str,
        path: str,
    ) -> HuggingFaceArtifactRefV1 | None:
        """Inspect a pinned checkpoint without confusing absence with a network error."""
        matches = [
            item
            for item in self._api.get_paths_info(
                repo_id=repo_id,
                paths=[path],
                repo_type=repo_type,
                revision=revision,
            )
            if getattr(item, "rfilename", None) == path
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(f"HF artifact is missing or ambiguous: {path}")
        return HuggingFaceArtifactRefV1(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            path=path,
            byte_count=int(getattr(matches[0], "size", 0) or 0),
        )

    def upload_artifacts(
        self,
        *,
        repo_id: str,
        artifacts: Sequence[tuple[Path, str]],
        commit_message: str,
        maximum_operations: int = 200,
    ) -> str:
        """Upload a bounded tree in small commits, then verify sizes and one read."""

        from huggingface_hub import CommitOperationAdd

        if not artifacts:
            raise ValueError("HF artifact upload cannot be empty")
        if maximum_operations < 1:
            raise ValueError("maximum operations must be positive")
        revision = str(self._api.dataset_info(repo_id).sha)
        for offset in range(0, len(artifacts), maximum_operations):
            batch = artifacts[offset : offset + maximum_operations]
            commit = self._api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                    for local, remote in batch
                ],
                commit_message=f"{commit_message} [{offset + 1}-{offset + len(batch)}]",
                parent_commit=revision,
            )
            revision = str(commit.oid)
        remote_sizes = {
            item.rfilename: int(getattr(item, "size", 0) or 0)
            for item in self._api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                recursive=True,
                expand=True,
            )
            if hasattr(item, "rfilename")
        }
        for local, remote in artifacts:
            if remote_sizes.get(remote) != local.stat().st_size:
                raise RuntimeError(f"HF artifact verification failed for {remote}")
        representative_local, representative_remote = artifacts[0]
        check_scratch_budget(
            self._scratch, additional_bytes=representative_local.stat().st_size + 16_384
        )
        checked = Path(
            self._download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=revision,
                filename=representative_remote,
                token=self._token,
                cache_dir=self._scratch / "hf-cache",
                local_dir=self._scratch / "remote-artifact-check" / revision,
            )
        )
        if checked.stat().st_size != representative_local.stat().st_size:
            raise RuntimeError("HF representative artifact read check failed")
        _verify_representative(checked)
        check_scratch_budget(self._scratch)
        return revision


def ensure_calibration_repo(repo_id: str) -> None:
    if repo_id.endswith("-holdout-v1"):
        raise ValueError("calibration workers must not be configured with the holdout repository")


def resolve_hf_revision(
    repo_id: str,
    *,
    repo_type: str,
    token: str | None = None,
) -> str:
    """Resolve a branch once and persist the immutable Hub commit returned by the API."""

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("install the backend calibration extra") from exc
    api = HfApi(token=token)
    if repo_type == "dataset":
        revision = str(api.dataset_info(repo_id).sha)
    elif repo_type == "model":
        revision = str(api.model_info(repo_id).sha)
    else:
        raise ValueError("Hugging Face repository type must be dataset or model")
    if len(revision) < 7 or revision.casefold() in {"main", "master", "latest"}:
        raise RuntimeError("Hugging Face did not return an immutable revision")
    return revision


__all__ = [
    "CalibrationReplayPayload",
    "HuggingFaceCampaignStore",
    "bounded_scratch",
    "ensure_calibration_repo",
    "freeze_compact_replay_result",
    "freeze_replay_result",
    "load_replay_payload",
    "replay_case",
    "resolve_hf_revision",
    "scratch_budget_bytes",
]
