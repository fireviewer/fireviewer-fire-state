"""Immutable, packaged Part.4 fusion-profile registry."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any

from pydantic import ValidationError

from fireviewer_contracts.backend.fire_state_schemas import FusionProfileV1

DEFAULT_FUSION_PROFILE_ID = "part4-framed-v1"
FUSION_PROFILE_LOCK_RESOURCE = "data/part4_fusion_profiles.lock.json"
_LOCK_SCHEMA = "fireviewer.part4-fusion-profile-lock.v1"
_PROFILE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _profile_content_sha256(payload: dict[str, Any]) -> str:
    hashable = dict(payload)
    hashable.pop("profile_sha256", None)
    return hashlib.sha256(_canonical_bytes(hashable)).hexdigest()


def _read_json_resource(resource_name: str) -> dict[str, Any]:
    if (
        not resource_name.startswith("data/")
        or ".." in resource_name.split("/")
        or not resource_name.endswith(".json")
    ):
        raise RuntimeError("Part.4 fusion profile resource path is invalid.")
    resource = files("fireviewer_fire_state")
    for component in resource_name.split("/"):
        resource = resource.joinpath(component)
    try:
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Part.4 fusion profile resource is unreadable: {resource_name}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Part.4 fusion profile resource is invalid: {resource_name}")
    return payload


@lru_cache(maxsize=1)
def _profile_lock() -> dict[str, tuple[str, str]]:
    payload = _read_json_resource(FUSION_PROFILE_LOCK_RESOURCE)
    if set(payload) != {"schema", "profiles"} or payload.get("schema") != _LOCK_SCHEMA:
        raise RuntimeError("Part.4 fusion profile lock is invalid.")
    entries = payload.get("profiles")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("Part.4 fusion profile lock is empty.")
    locked: dict[str, tuple[str, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
            "profile_id",
            "resource",
            "profile_sha256",
        }:
            raise RuntimeError("Part.4 fusion profile lock entry is invalid.")
        profile_id = entry.get("profile_id")
        resource_name = entry.get("resource")
        expected_sha256 = entry.get("profile_sha256")
        if (
            not isinstance(profile_id, str)
            or _PROFILE_ID.fullmatch(profile_id) is None
            or profile_id in locked
            or not isinstance(resource_name, str)
            or not isinstance(expected_sha256, str)
            or _SHA256.fullmatch(expected_sha256) is None
        ):
            raise RuntimeError("Part.4 fusion profile lock identity is invalid.")
        locked[profile_id] = (resource_name, expected_sha256)
    return locked


@lru_cache(maxsize=32)
def load_fusion_profile(profile_id: str, *, algorithm_version: str) -> FusionProfileV1:
    locked = _profile_lock().get(profile_id)
    if locked is None:
        raise RuntimeError(f"Part.4 fusion profile is not allowlisted: {profile_id}")
    resource_name, expected_sha256 = locked
    payload = _read_json_resource(resource_name)
    content_sha256 = _profile_content_sha256(payload)
    if (
        payload.get("profile_id") != profile_id
        or payload.get("profile_sha256") != expected_sha256
        or content_sha256 != expected_sha256
    ):
        raise RuntimeError(f"Part.4 fusion profile integrity check failed: {profile_id}")
    try:
        profile = FusionProfileV1.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"Part.4 fusion profile contract is invalid: {profile_id}") from exc
    if algorithm_version not in profile.algorithm_versions:
        raise RuntimeError(
            f"Part.4 fusion profile {profile_id} does not support algorithm {algorithm_version}."
        )
    return profile


def clear_fusion_profile_cache() -> None:
    """Clear package-resource caches for bounded integrity tests."""

    load_fusion_profile.cache_clear()
    _profile_lock.cache_clear()


__all__ = [
    "DEFAULT_FUSION_PROFILE_ID",
    "FUSION_PROFILE_LOCK_RESOURCE",
    "clear_fusion_profile_cache",
    "load_fusion_profile",
]
