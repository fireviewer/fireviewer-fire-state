"""CPU-only calibration primitives for frozen Part.4 campaign results.

This module never reads published references while Part.4 is running.  It only
accepts already-frozen prediction metrics produced by the isolated evaluator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

import numpy as np

from fireviewer_contracts.backend.fire_state_schemas import FusionProfileV1
from fireviewer_contracts.backend.hashing import sha256_hex
from fireviewer_contracts.backend.part4_calibration_schemas import (
    Part4ComponentCalibrationProfileV1,
    Part4ComponentReleaseDecisionV1,
    Part4ConfidenceCalibratorV1,
    Part4QualificationReceiptV1,
)

CONFIDENCE_FEATURE_NAMES = (
    "observed_fraction",
    "fused_fraction",
    "interpolated_fraction",
    "evidence_strength",
    "observable_fraction",
    "uncertainty_fraction",
    "resolution_log_m",
    "observation_age_log_hours",
    "independent_family_log_count",
    "contradiction_count",
    "status_observed",
    "status_fused",
)

WEIGHT_MULTIPLIERS = (0.50, 0.75, 1.00, 1.25, 1.50)
AFFECTED_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
ACTIVE_THRESHOLDS = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
UNCERTAINTY_LOW_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
UNCERTAINTY_HIGH_THRESHOLDS = (0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
ACTIVE_HALF_LIFE_HOURS = (2.0, 4.0, 6.0, 9.0, 12.0, 18.0, 24.0)


def _finite(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return default
    numeric = float(value)
    return numeric if math.isfinite(numeric) else default


def _candidate_profile(
    base: FusionProfileV1,
    *,
    candidate_number: int,
    update: tuple[str, str | None, float] | None,
) -> FusionProfileV1:
    payload = base.model_dump(mode="json", by_alias=True)
    payload["profile_id"] = f"part4-calibration-screen-{candidate_number:04d}"
    payload["profile_version"] = "0.0.1"
    payload["calibration_state"] = "uncalibrated"
    if update is not None:
        section, key, value = update
        if key is None:
            payload[section] = value
        else:
            nested = dict(payload[section])
            if section == "sensors":
                sensor = dict(nested[key])
                sensor["weight"] = value
                nested[key] = sensor
            else:
                nested[key] = value
            payload[section] = nested
    payload_without_identity = dict(payload)
    payload_without_identity.pop("profile_sha256", None)
    payload["profile_sha256"] = sha256_hex(payload_without_identity)
    return FusionProfileV1.model_validate(payload)


def build_screening_profiles(base: FusionProfileV1) -> tuple[FusionProfileV1, ...]:
    """Create the bounded one-factor screening set from the immutable baseline."""

    updates: list[tuple[str, str | None, float] | None] = [None]
    for sensor_name, sensor in sorted(base.sensors.items()):
        updates.extend(
            ("sensors", sensor_name, sensor.weight * multiplier)
            for multiplier in WEIGHT_MULTIPLIERS
            if multiplier != 1.0
        )
    updates.extend(
        ("active_to_affected", "weight", base.active_to_affected.weight * multiplier)
        for multiplier in WEIGHT_MULTIPLIERS
        if multiplier != 1.0
    )
    updates.extend(
        ("thresholds", "affected", value)
        for value in AFFECTED_THRESHOLDS
        if value != base.thresholds.affected
    )
    updates.extend(
        ("thresholds", "active", value)
        for value in ACTIVE_THRESHOLDS
        if value != base.thresholds.active
    )
    updates.extend(
        ("thresholds", "uncertainty_low", value)
        for value in UNCERTAINTY_LOW_THRESHOLDS
        if value != base.thresholds.uncertainty_low
    )
    updates.extend(
        ("thresholds", "uncertainty_high", value)
        for value in UNCERTAINTY_HIGH_THRESHOLDS
        if value != base.thresholds.uncertainty_high
    )
    updates.extend(
        ("temporal", "active_half_life_hours", value)
        for value in ACTIVE_HALF_LIFE_HOURS
        if value != base.temporal.active_half_life_hours
    )
    profiles = tuple(
        _candidate_profile(base, candidate_number=index, update=update)
        for index, update in enumerate(updates)
    )
    identities = {profile.profile_sha256 for profile in profiles}
    if len(identities) != len(profiles):
        raise ValueError("screening profile generation produced duplicate candidates")
    return profiles


def build_restricted_affected_profiles(base: FusionProfileV1) -> tuple[FusionProfileV1, ...]:
    """Four predeclared one-factor alternatives; never tune unobserved active labels."""

    updates = (
        ("thresholds", "affected", 0.40),
        ("thresholds", "affected", 0.60),
        ("sensors", "burned_probability", base.sensors["burned_probability"].weight * 0.75),
        ("sensors", "burned_probability", base.sensors["burned_probability"].weight * 1.25),
    )
    alternatives = tuple(
        _candidate_profile(base, candidate_number=index, update=update)
        for index, update in enumerate(updates, start=1)
    )
    return (base, *alternatives)


def select_restricted_affected_profile(
    results: Mapping[str, Sequence[Mapping[str, Any]]], *, baseline_id: str
) -> dict[str, Any]:
    """Choose on calibration only; validation may veto, never nominate another winner."""

    if baseline_id not in results:
        raise ValueError("restricted calibration requires the unchanged baseline")
    calibration = {
        key: [row for row in rows if row.get("split") == "calibration"]
        for key, rows in results.items()
    }
    validation = {
        key: [row for row in rows if row.get("split") == "validation"]
        for key, rows in results.items()
    }
    baseline_by_case = {
        (str(row["incident_id"]), str(row["case_id"])): row for row in results[baseline_id]
    }
    incident_splits: dict[str, set[str]] = defaultdict(set)
    for row in results[baseline_id]:
        incident_splits[str(row["incident_id"])].add(str(row.get("split")))
    if any(len(splits) != 1 for splits in incident_splits.values()):
        raise ValueError("restricted calibration cannot split an incident across groups")
    for rows in results.values():
        if any(row.get("split") not in {"calibration", "validation"} for row in rows):
            raise ValueError("restricted calibration cannot inspect holdout rows")
        keys = [(str(row["incident_id"]), str(row["case_id"])) for row in rows]
        if len(keys) != len(set(keys)) or set(keys) != set(baseline_by_case):
            raise ValueError("restricted profiles must cover identical unique cases")
        if any(
            row.get("split") != baseline_by_case[key].get("split")
            for key, row in zip(keys, rows, strict=True)
        ):
            raise ValueError("restricted profiles must preserve case splits")
    if any(not rows for rows in calibration.values()):
        raise ValueError("restricted profiles require calibration cases")
    base_score = affected_objective(calibration[baseline_id])
    best_id, best_score = rank_candidate_profiles(calibration, keep=1)[0]
    selected_id, reason = baseline_id, "no_calibration_improvement"
    if best_score > base_score + 1e-9:
        if not validation[best_id]:
            reason = "validation_missing"
        else:
            hard_failures = (
                "reference_leakage_detected", "simulated_contribution_detected",
                "unjustified_affected_regression_detected",
            )
            candidate_rows = results[best_id]
            newly_invalid = any(
                row.get("geometry_valid") is not True
                and baseline_by_case[(str(row["incident_id"]), str(row["case_id"]))].get(
                    "geometry_valid"
                ) is True for row in candidate_rows
            )
            if newly_invalid or any(
                row.get(code) is True for row in candidate_rows for code in hard_failures
            ):
                reason = "candidate_hard_failure"
            elif affected_objective(validation[best_id]) + 1e-9 < affected_objective(
                validation[baseline_id]
            ):
                reason = "validation_regression"
            else:
                selected_id, reason = best_id, "exploratory_improvement_only"
    incident_ids = sorted({str(row["incident_id"]) for row in calibration[baseline_id]})
    folds: list[dict[str, Any]] = []
    if len(incident_ids) >= 2:
        for incident in incident_ids:
            train = {
                key: [row for row in rows if row["incident_id"] != incident]
                for key, rows in calibration.items()
            }
            winner, score = rank_candidate_profiles(train, keep=1)[0]
            if score <= affected_objective(train[baseline_id]) + 1e-9:
                winner = baseline_id
            omitted = [row for row in calibration[winner] if row["incident_id"] == incident]
            base_omitted = [
                row for row in calibration[baseline_id] if row["incident_id"] == incident
            ]
            folds.append({
                "omitted_incident_id": incident, "selected_on_other_incidents": winner,
                "omitted_objective_delta": affected_objective(omitted)
                - affected_objective(base_omitted),
            })
    return {
        "schema": "fireviewer.part4-restricted-affected-selection.v1",
        "baseline_profile_id": baseline_id,
        "calibration_winner": best_id,
        "selected_profile_id": selected_id,
        "reason": reason,
        "calibration_objective_delta": best_score - base_score,
        "validation_objective_delta": (
            affected_objective(validation[best_id]) - affected_objective(validation[baseline_id])
            if validation[best_id] else None
        ),
        "leave_one_incident_out": folds,
        "validation_used_to_nominate_candidate": False,
        "confidence_calibrator_fitted": False,
        "qualification_allowed": False,
    }


def refinement_values(center: float, *, lower: float, upper: float) -> tuple[float, ...]:
    """Two deterministic narrowing passes around a validation-selected value."""

    if not lower < center < upper:
        raise ValueError("refinement center must be strictly inside its bounds")
    first_step = min(center - lower, upper - center) / 2.0
    second_step = first_step / 2.0
    return tuple(
        sorted(
            {
                round(center - first_step, 12),
                round(center - second_step, 12),
                round(center, 12),
                round(center + second_step, 12),
                round(center + first_step, 12),
            }
        )
    )


def threshold_masks(
    affected_probability: np.ndarray[Any, Any],
    active_probability: np.ndarray[Any, Any],
    *,
    affected_threshold: float,
    active_threshold: float,
) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], np.ndarray[Any, np.dtype[np.bool_]]]:
    """Sweep thresholds on frozen grids without re-running upstream models."""

    if not 0 < affected_threshold < 1 or not 0 < active_threshold < 1:
        raise ValueError("probability thresholds must be within the open unit interval")
    if affected_probability.shape != active_probability.shape:
        raise ValueError("affected and active grids must be aligned")
    return (
        np.asarray(affected_probability >= affected_threshold, dtype=np.bool_),
        np.asarray(active_probability >= active_threshold, dtype=np.bool_),
    )


def confidence_features(row: Mapping[str, Any]) -> tuple[float, ...]:
    """Convert one frozen result into the stable confidence feature vector."""

    status = str(row.get("reconstruction_status", row.get("status", "insufficient")))
    resolution = max(0.0, _finite(row.get("resolution_m")))
    age_hours = max(0.0, _finite(row.get("observation_age_hours")))
    families = max(0.0, _finite(row.get("independent_family_count")))
    return (
        _finite(row.get("observed_fraction")),
        _finite(row.get("fused_fraction")),
        _finite(row.get("interpolated_fraction")),
        _finite(row.get("evidence_strength")),
        _finite(row.get("observable_fraction", row.get("observed_fraction"))),
        _finite(row.get("uncertainty_fraction")),
        math.log1p(resolution),
        math.log1p(age_hours),
        math.log1p(families),
        max(0.0, _finite(row.get("contradiction_count"))),
        1.0 if status == "observed" else 0.0,
        1.0 if status == "fused" else 0.0,
    )


def calibration_success(row: Mapping[str, Any]) -> bool:
    """Label required by the affected confidence specification."""

    return bool(
        row.get("geometry_valid") is True
        and _finite(row.get("iou"), default=-1) >= 0.45
        and _finite(row.get("boundary_f1"), default=-1) >= 0.70
        and row.get("reference_leakage_detected") is not True
    )


def fit_confidence_calibrator(
    rows: Sequence[Mapping[str, Any]],
    *,
    calibrator_id: str,
    regularization_l2: float = 1.0,
    max_iterations: int = 100,
) -> Part4ConfidenceCalibratorV1:
    """Fit deterministic L2 logistic regression using Newton iterations."""

    if not rows:
        raise ValueError("confidence calibration requires frozen evaluation rows")
    incidents = {str(row.get("incident_id", "")) for row in rows}
    incidents.discard("")
    if not incidents:
        raise ValueError("confidence calibration rows require incident identifiers")
    if regularization_l2 <= 0 or not math.isfinite(regularization_l2):
        raise ValueError("confidence regularization must be finite and positive")
    x = np.asarray([confidence_features(row) for row in rows], dtype=np.float64)
    y = np.asarray([1.0 if calibration_success(row) else 0.0 for row in rows], dtype=np.float64)
    if np.unique(y).size != 2:
        raise ValueError("confidence calibration requires successful and unsuccessful examples")
    means = np.mean(x, axis=0)
    scales = np.std(x, axis=0)
    scales = np.where(scales < 1e-9, 1.0, scales)
    standardized = (x - means) / scales
    design = np.column_stack((np.ones(len(rows), dtype=np.float64), standardized))
    coefficients = np.zeros(design.shape[1], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * regularization_l2
    penalty[0, 0] = 0.0
    for _ in range(max_iterations):
        logits = np.clip(design @ coefficients, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-logits))
        gradient = design.T @ (probability - y) + penalty @ coefficients
        weights = np.clip(probability * (1.0 - probability), 1e-8, None)
        hessian = design.T @ (design * weights[:, None]) + penalty
        try:
            delta = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            delta = np.linalg.pinv(hessian) @ gradient
        coefficients -= delta
        if float(np.max(np.abs(delta))) < 1e-9:
            break
    if not bool(np.all(np.isfinite(coefficients))):
        raise ValueError("confidence calibration did not converge to finite coefficients")
    return Part4ConfidenceCalibratorV1(
        calibrator_id=calibrator_id,
        trained_at=datetime.now(UTC),
        feature_names=CONFIDENCE_FEATURE_NAMES,
        feature_means=tuple(float(value) for value in means),
        feature_scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients[1:]),
        intercept=float(coefficients[0]),
        regularization_l2=regularization_l2,
        fit_incident_count=len(incidents),
        fit_snapshot_count=len(rows),
    )


def predict_confidence(
    calibrator: Part4ConfidenceCalibratorV1,
    row: Mapping[str, Any],
) -> float:
    if calibrator.feature_names != CONFIDENCE_FEATURE_NAMES:
        raise ValueError("confidence calibrator feature contract is incompatible")
    raw = confidence_features(row)
    standardized = [
        (value - mean) / scale
        for value, mean, scale in zip(
            raw,
            calibrator.feature_means,
            calibrator.feature_scales,
            strict=True,
        )
    ]
    logit = calibrator.intercept + sum(
        value * coefficient
        for value, coefficient in zip(
            standardized,
            calibrator.coefficients,
            strict=True,
        )
    )
    probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, logit))))
    return round(probability, 12)


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("Brier score requires paired non-empty values")
    return float(
        sum(
            (float(probability) - (1.0 if label else 0.0)) ** 2
            for probability, label in zip(probabilities, labels, strict=True)
        )
        / len(probabilities)
    )


def expected_calibration_error(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    *,
    bin_count: int = 10,
) -> float:
    """Equal-width expected calibration error."""

    if not probabilities or len(probabilities) != len(labels):
        raise ValueError("ECE requires paired non-empty values")
    if bin_count < 2:
        raise ValueError("ECE requires at least two bins")
    total = len(probabilities)
    error = 0.0
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [
            position
            for position, probability in enumerate(probabilities)
            if lower <= probability < upper or (index == bin_count - 1 and probability == 1.0)
        ]
        if not selected:
            continue
        confidence = sum(probabilities[position] for position in selected) / len(selected)
        accuracy = sum(1.0 if labels[position] else 0.0 for position in selected) / len(selected)
        error += len(selected) / total * abs(accuracy - confidence)
    return float(error)


def incident_bootstrap_precision_lower_bound(
    rows: Sequence[Mapping[str, Any]],
    *,
    confidence_threshold: float = 0.85,
    iterations: int = 2_000,
    seed: int = 20260826,
) -> float:
    """Return the incident-clustered 95% lower precision bound."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        incident_id = str(row.get("incident_id", ""))
        if incident_id:
            grouped[incident_id].append(row)
    incident_ids = sorted(grouped)
    if not incident_ids or iterations <= 0:
        return 0.0
    generator = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        selected_ids = generator.choice(incident_ids, size=len(incident_ids), replace=True)
        eligible: list[Mapping[str, Any]] = []
        for incident_id in selected_ids:
            eligible.extend(
                row
                for row in grouped[str(incident_id)]
                if _finite(row.get("confidence_calibrated"), default=-1) > confidence_threshold
            )
        if eligible:
            samples.append(sum(calibration_success(row) for row in eligible) / len(eligible))
    if not samples:
        return 0.0
    return float(np.quantile(np.asarray(samples, dtype=np.float64), 0.025))


def affected_objective(rows: Sequence[Mapping[str, Any]]) -> float:
    """Aggregate the specified affected objective per incident before global averaging."""

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        incident_id = str(row.get("incident_id", ""))
        if not incident_id:
            raise ValueError("objective rows require incident identifiers")
        if row.get("geometry_valid") is False:
            # Missing comparison metrics are not a zero area bias or a good prediction.
            grouped[incident_id].append(0.0)
            continue
        iou = max(0.0, min(1.0, _finite(row.get("iou"))))
        boundary = max(0.0, min(1.0, _finite(row.get("boundary_f1"))))
        surface_score = 1.0 - min(1.0, abs(_finite(row.get("surface_bias_ratio"))))
        coverage = _finite(row.get("uncertainty_boundary_coverage"))
        uncertainty_score = max(0.0, 1.0 - abs(coverage - 0.90) / 0.90)
        grouped[incident_id].append(
            0.45 * iou + 0.30 * boundary + 0.15 * surface_score + 0.10 * uncertainty_score
        )
    if not grouped:
        raise ValueError("affected objective requires evaluation rows")
    incident_scores = [sum(values) / len(values) for values in grouped.values()]
    return float(sum(incident_scores) / len(incident_scores))


def rank_candidate_profiles(
    results: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    keep: int = 8,
) -> tuple[tuple[str, float], ...]:
    if keep < 1:
        raise ValueError("at least one calibration candidate must be retained")
    ranked = sorted(
        ((profile_id, affected_objective(rows)) for profile_id, rows in results.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return tuple(ranked[:keep])


def qualification_receipt(
    *,
    qualification_id: str,
    fusion_profile_id: str,
    calibrator_id: str,
    campaign_id: str,
    holdout_dataset_repo: str,
    holdout_dataset_revision: str,
    rows: Sequence[Mapping[str, Any]],
    campaign_incident_count: int,
    campaign_snapshot_count: int,
) -> Part4QualificationReceiptV1:
    incidents = {str(row.get("incident_id", "")) for row in rows}
    incidents.discard("")
    high_confidence = [
        row
        for row in rows
        if _finite(row.get("confidence_calibrated"), default=-1) > 0.85
    ]
    high_incidents = {str(row.get("incident_id", "")) for row in high_confidence}
    high_incidents.discard("")

    def median(key: str, *, default: float = 0.0) -> float:
        values = [_finite(row.get(key), default=math.nan) for row in rows]
        finite = [value for value in values if math.isfinite(value)]
        return float(np.median(np.asarray(finite))) if finite else default

    labels = [calibration_success(row) for row in rows]
    probabilities = [
        max(0.0, min(1.0, _finite(row.get("confidence_calibrated")))) for row in rows
    ]
    high_precision = (
        sum(calibration_success(row) for row in high_confidence) / len(high_confidence)
        if high_confidence
        else 0.0
    )
    bootstrap_lower = incident_bootstrap_precision_lower_bound(rows)
    leakage_count = sum(row.get("reference_leakage_detected") is True for row in rows)
    simulated_count = sum(row.get("simulated_contribution_detected") is True for row in rows)
    invalid_count = sum(row.get("geometry_valid") is not True for row in rows)
    regression_count = sum(
        row.get("unjustified_affected_regression_detected") is True for row in rows
    )
    high_resolution_rows = [
        row for row in rows if 0 < _finite(row.get("best_observation_resolution_m")) <= 20
    ]
    median_iou_20m = (
        float(np.median([_finite(row.get("iou")) for row in high_resolution_rows]))
        if high_resolution_rows
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "campaign_incident_count": campaign_incident_count,
        "campaign_snapshot_count": campaign_snapshot_count,
        "holdout_incident_count": len(incidents),
        "holdout_snapshot_count": len(rows),
        "invalid_geometry_count": invalid_count,
        "unjustified_affected_regression_count": regression_count,
        "median_iou_20m": median_iou_20m,
        "median_iou_global": median("iou"),
        "median_boundary_f1": median("boundary_f1"),
        "uncertainty_boundary_coverage_90": median("uncertainty_boundary_coverage"),
        "brier": brier_score(probabilities, labels) if rows else 1.0,
        "ece": expected_calibration_error(probabilities, labels) if rows else 1.0,
        "high_confidence_count": len(high_confidence),
        "high_confidence_incident_count": len(high_incidents),
        "high_confidence_precision": high_precision,
        "high_confidence_precision_bootstrap_lower_95": bootstrap_lower,
        "reference_leakage_count": leakage_count,
        "simulated_contribution_count": simulated_count,
    }
    coverage = float(metrics["uncertainty_boundary_coverage_90"])
    gates = {
        "minimum_incidents": campaign_incident_count >= 50,
        "minimum_snapshots": campaign_snapshot_count >= 300,
        "holdout_minimum_incidents": len(incidents) >= 15,
        "holdout_minimum_snapshots": len(rows) >= 150,
        "valid_geometries": invalid_count == 0,
        "affected_is_monotonic_or_revised": regression_count == 0,
        "median_iou_20m": bool(high_resolution_rows) and median_iou_20m >= 0.60,
        "median_iou_global": float(metrics["median_iou_global"]) >= 0.45,
        "median_boundary_f1": float(metrics["median_boundary_f1"]) >= 0.70,
        "uncertainty_coverage": 0.80 <= coverage <= 0.98,
        "brier": float(metrics["brier"]) <= 0.18,
        "ece": float(metrics["ece"]) <= 0.08,
        "high_confidence_volume": len(high_confidence) >= 50 and len(high_incidents) >= 10,
        "high_confidence_precision": high_precision >= 0.90,
        "bootstrap_precision_lower": bootstrap_lower >= 0.85,
        "no_reference_leakage": leakage_count == 0,
        "no_simulated_contribution": simulated_count == 0,
    }
    return Part4QualificationReceiptV1(
        qualification_id=qualification_id,
        evaluated_at=datetime.now(UTC),
        component="affected",
        fusion_profile_id=fusion_profile_id,
        calibrator_id=calibrator_id,
        campaign_id=campaign_id,
        holdout_dataset_repo=holdout_dataset_repo,
        holdout_dataset_revision=holdout_dataset_revision,
        campaign_incident_count=campaign_incident_count,
        campaign_snapshot_count=campaign_snapshot_count,
        holdout_incident_count=len(incidents),
        holdout_snapshot_count=len(rows),
        metrics=metrics,
        gates=gates,
        qualified=all(gates.values()),
        reference_leakage_detected=leakage_count > 0,
        simulated_contribution_detected=simulated_count > 0,
    )


def component_release_decision(
    *,
    profile: Part4ComponentCalibrationProfileV1,
    perimeter_receipt_id: str,
    incident_id: str,
    episode_id: str | None,
    frozen_result: Mapping[str, Any],
    qualification_revision: str,
) -> Part4ComponentReleaseDecisionV1:
    confidence = predict_confidence(profile.calibrator, frozen_result)
    status = str(
        frozen_result.get("reconstruction_status", frozen_result.get("status", "insufficient"))
    )
    if status not in {"observed", "fused", "interpolated", "insufficient"}:
        raise ValueError("unknown Part.4 reconstruction status")
    contradictions = tuple(
        sorted({str(value) for value in frozen_result.get("contradiction_codes", ())})
    )
    simulated = bool(frozen_result.get("simulated_contribution_detected", False))
    eligible = (
        confidence > 0.85
        and status in {"observed", "fused"}
        and not contradictions
        and not simulated
    )
    reasons = [f"confidence_{'above' if confidence > 0.85 else 'not_above'}_0_85"]
    reasons.append(f"reconstruction_{status}")
    if contradictions:
        reasons.append("hard_contradiction_present")
    if simulated:
        reasons.append("simulated_contribution_present")
    if eligible:
        reasons.append("affected_component_qualified")
    identity = sha256_hex(
        {
            "perimeter_receipt_id": perimeter_receipt_id,
            "component": "affected",
            "qualification_revision": qualification_revision,
        }
    )
    return Part4ComponentReleaseDecisionV1(
        decision_id=f"P4D-{identity[:32]}",
        decided_at=datetime.now(UTC),
        perimeter_receipt_id=perimeter_receipt_id,
        incident_id=incident_id,
        episode_id=episode_id,
        component="affected",
        qualification_revision=qualification_revision,
        confidence_calibrated=confidence,
        reconstruction_status=cast(Any, status),
        reason_codes=tuple(reasons),
        contradiction_codes=contradictions,
        simulated_contribution_detected=simulated,
        eligible_for_automatic_publication=eligible,
    )


def iter_incident_groups(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[Mapping[str, Any], ...]], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        incident_id = str(row.get("incident_id", ""))
        if not incident_id:
            raise ValueError("campaign rows require an incident identifier")
        grouped[incident_id].append(row)
    return tuple((key, tuple(grouped[key])) for key in sorted(grouped))


__all__ = [
    "CONFIDENCE_FEATURE_NAMES",
    "affected_objective",
    "brier_score",
    "build_screening_profiles",
    "calibration_success",
    "component_release_decision",
    "confidence_features",
    "expected_calibration_error",
    "fit_confidence_calibrator",
    "incident_bootstrap_precision_lower_bound",
    "iter_incident_groups",
    "predict_confidence",
    "qualification_receipt",
    "rank_candidate_profiles",
    "refinement_values",
    "threshold_masks",
]
