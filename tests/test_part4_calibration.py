from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from fireviewer_contracts.backend.fire_state_schemas import SpatialObservationV2
from fireviewer_contracts.backend.part4_calibration_schemas import (
    HuggingFaceArtifactRefV1,
    Part4CalibrationCaseV1,
    Part4ComponentReleaseDecisionV1,
    Part4ReferenceSnapshotV1,
)
from fireviewer_fire_state.fire_state_fusion import (
    BASELINE_ALGORITHM_VERSION as FUSION_ALGORITHM_VERSION,
)
from fireviewer_fire_state.fire_state_fusion import (
    fuse_probability_baseline as fuse_daily_fire_state,
)
from fireviewer_fire_state.part4_calibration import (
    affected_objective,
    brier_score,
    build_restricted_affected_profiles,
    build_screening_profiles,
    expected_calibration_error,
    fit_confidence_calibrator,
    predict_confidence,
    qualification_receipt,
    rank_candidate_profiles,
    refinement_values,
    select_restricted_affected_profile,
    threshold_masks,
)
from fireviewer_fire_state.part4_calibration_campaign import (
    bounded_scratch,
    ensure_calibration_repo,
    freeze_replay_result,
    load_replay_payload,
)
from fireviewer_fire_state.part4_calibration_evaluation import (
    evaluate_frozen_affected_prediction,
)
from fireviewer_fire_state.part4_fusion_profiles import load_fusion_profile

NOW = datetime(2026, 7, 15, 12, tzinfo=UTC)
HF_REVISION = "1" * 40


def test_restricted_profiles_preserve_all_unmeasured_parameters() -> None:
    base = load_fusion_profile(
        "part4-baseline-v3-provenance", algorithm_version=FUSION_ALGORITHM_VERSION
    )
    profiles = build_restricted_affected_profiles(base)
    assert profiles[0] is base
    assert len(profiles) == 5
    for profile in profiles:
        assert profile.calibration_state == "uncalibrated"
        assert profile.grid == base.grid
        assert profile.priors == base.priors
        assert profile.temporal == base.temporal
        assert profile.uncertainty == base.uncertainty
        assert profile.active_to_affected == base.active_to_affected
        assert profile.valid_negative == base.valid_negative
        assert profile.thresholds.active == base.thresholds.active
        assert profile.thresholds.uncertainty_low == base.thresholds.uncertainty_low
        assert profile.thresholds.uncertainty_high == base.thresholds.uncertainty_high
        for kind, sensor in profile.sensors.items():
            assert sensor.profile_probability == base.sensors[kind].profile_probability
            if kind != "burned_probability":
                assert sensor == base.sensors[kind]
    assert build_restricted_affected_profiles(base) == profiles


def test_missing_geometry_cannot_gain_objective_from_missing_area_bias() -> None:
    assert affected_objective([{
        "incident_id": "missing", "geometry_valid": False, "surface_bias_ratio": 0.0,
    }]) == 0.0


def test_restricted_selection_uses_validation_only_as_veto() -> None:
    def rows(train: bool, validation: bool) -> list[dict[str, object]]:
        return [
            {**_metric_row(1, success=train), "case_id": "c1", "split": "calibration"},
            {**_metric_row(2, success=train), "case_id": "c2", "split": "calibration"},
            {**_metric_row(3, success=validation), "case_id": "v1", "split": "validation"},
        ]

    baseline = rows(False, True)
    candidate = rows(True, False)
    results = {"baseline": baseline, "calibration-winner": candidate, "val-only": baseline}
    selected = select_restricted_affected_profile(results, baseline_id="baseline")
    assert selected["calibration_winner"] == "calibration-winner"
    assert selected["selected_profile_id"] == "baseline"
    assert selected["reason"] == "validation_regression"
    assert selected["validation_used_to_nominate_candidate"] is False
    assert len(selected["leave_one_incident_out"]) == 2
    results["calibration-winner"] = rows(True, True)
    selected = select_restricted_affected_profile(results, baseline_id="baseline")
    assert selected["selected_profile_id"] == "calibration-winner"
    assert selected["qualification_allowed"] is False
    candidate[0]["split"] = "holdout"
    with pytest.raises(ValueError, match="holdout"):
        select_restricted_affected_profile(
            {"baseline": baseline, "bad": candidate}, baseline_id="baseline"
        )
    with pytest.raises(ValueError, match="identical unique cases"):
        select_restricted_affected_profile(
            {"baseline": baseline, "bad": baseline[:1]}, baseline_id="baseline"
        )


@pytest.mark.parametrize("failure", [
    "reference_leakage_detected", "simulated_contribution_detected",
    "unjustified_affected_regression_detected",
])
def test_restricted_selection_rejects_better_but_unsafe_candidate(failure: str) -> None:
    baseline = [
        {**_metric_row(1, success=False), "case_id": "cal", "split": "calibration"},
        {**_metric_row(2, success=False), "case_id": "val", "split": "validation"},
    ]
    candidate = [
        {**_metric_row(1, success=True), "case_id": "cal", "split": "calibration", failure: True},
        {**_metric_row(2, success=True), "case_id": "val", "split": "validation"},
    ]
    decision = select_restricted_affected_profile(
        {"baseline": baseline, "candidate": candidate}, baseline_id="baseline"
    )
    assert decision["selected_profile_id"] == "baseline"
    assert decision["reason"] == "candidate_hard_failure"


def test_restricted_selection_requires_stable_splits_and_validation() -> None:
    base = [{**_metric_row(1, success=False), "case_id": "cal", "split": "calibration"}]
    better = [{**_metric_row(1, success=True), "case_id": "cal", "split": "calibration"}]
    decision = select_restricted_affected_profile(
        {"baseline": base, "candidate": better}, baseline_id="baseline"
    )
    assert decision["selected_profile_id"] == "baseline"
    assert decision["reason"] == "validation_missing"
    assert decision["leave_one_incident_out"] == []
    changed = [{**better[0], "split": "validation"}]
    with pytest.raises(ValueError, match="preserve case splits"):
        select_restricted_affected_profile(
            {"baseline": base, "candidate": changed}, baseline_id="baseline"
        )
    mixed = [*base, {**base[0], "case_id": "same-fire", "split": "validation"}]
    with pytest.raises(ValueError, match="split an incident"):
        select_restricted_affected_profile({"baseline": mixed}, baseline_id="baseline")


def _artifact(*, repo_id: str = "fireviewer/part4-france-calibration-v1") -> dict[str, object]:
    return HuggingFaceArtifactRefV1(
        repo_id=repo_id,
        repo_type="dataset",
        revision=HF_REVISION,
        path="observations/case-1.json",
        byte_count=123,
    ).model_dump(mode="json", by_alias=True)


def _metric_row(
    index: int,
    *,
    success: bool,
    confidence: float | None = None,
) -> dict[str, object]:
    return {
        "incident_id": f"FR-26-{index % 50:05d}",
        "geometry_valid": True,
        "iou": 0.72 if success else 0.20,
        "boundary_f1": 0.82 if success else 0.40,
        "surface_bias_ratio": 0.05,
        "uncertainty_boundary_coverage": 0.90,
        "best_observation_resolution_m": 20,
        "observed_fraction": 0.8 if success else 0.1,
        "fused_fraction": 0.7 if success else 0.0,
        "interpolated_fraction": 0.1 if success else 0.9,
        "evidence_strength": 0.9 if success else 0.2,
        "observable_fraction": 0.85,
        "uncertainty_fraction": 0.1 if success else 0.8,
        "resolution_m": 20 if success else 250,
        "observation_age_hours": 2 if success else 48,
        "independent_family_count": 3 if success else 1,
        "contradiction_count": 0 if success else 2,
        "reconstruction_status": "fused" if success else "interpolated",
        "reference_leakage_detected": False,
        "simulated_contribution_detected": False,
        "unjustified_affected_regression_detected": False,
        **({"confidence_calibrated": confidence} if confidence is not None else {}),
    }


def test_reference_grades_and_hf_revisions_fail_closed() -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[5.0, 44.0], [5.1, 44.0], [5.1, 44.1], [5.0, 44.0]]],
    }
    reference = Part4ReferenceSnapshotV1(
        reference_id="ref-1",
        incident_id="FR-26-00001",
        component="affected",
        geometry_geojson=geometry,
        valid_at=NOW,
        temporal_accuracy_seconds=3_600,
        spatial_accuracy_m=100,
        resolution_m=20,
        provider="CEMS",
        product="GRA",
        licence="Copernicus terms",
        source_revision="EMSR-test-v1",
        grade="A",
        forbidden_input_refs=("EMSR-test-v1",),
    )
    assert reference.grade == "A"
    with pytest.raises(ValidationError, match="grade A"):
        Part4ReferenceSnapshotV1.model_validate(
            {**reference.model_dump(mode="json", by_alias=True), "spatial_accuracy_m": 101}
        )
    with pytest.raises(ValidationError, match=r"at least 7|immutable revision"):
        HuggingFaceArtifactRefV1(
            repo_id="fireviewer/dataset",
            repo_type="dataset",
            revision="main",
            path="data.json",
            byte_count=1,
        )


def test_case_rejects_future_inputs_and_calibrator_cannot_open_holdout(tmp_path: Path) -> None:
    base = {
        "case_id": "case-1",
        "incident_id": "FR-26-00001",
        "evaluation_cutoff_at": NOW.isoformat(),
        "latest_input_at": NOW.isoformat(),
        "observations": _artifact(),
        "split": "calibration",
        "split_id": "fr-v1",
        "reference_ids": ["ref-1"],
    }
    assert Part4CalibrationCaseV1.model_validate(base).split == "calibration"
    with pytest.raises(ValidationError, match="newer than the cutoff"):
        Part4CalibrationCaseV1.model_validate(
            {**base, "latest_input_at": (NOW + timedelta(seconds=1)).isoformat()}
        )
    with pytest.raises(ValueError, match="holdout"):
        ensure_calibration_repo("fireviewer/part4-france-holdout-v1")

    observation = {
        "schema": "fireviewer.spatial-observation.v2",
        "observation_id": "obs-1",
        "observation_kind": "burned_probability",
        "target_state": "affected",
        "observed_at": (NOW + timedelta(seconds=1)).isoformat(),
        "geometry_geojson": {
            "type": "Polygon",
            "coordinates": [[[5.0, 44.0], [5.01, 44.0], [5.01, 44.01], [5.0, 44.0]]],
        },
        "probability": 0.8,
        "resolution_m": 20,
        "source_family_id": "satellite:s2-msi",
        "lineage_id": "satellite:s2-msi:pair-1",
        "upstream_product_id": "pair-1",
        "source_revision_sha256": "a" * 64,
        "processor_revision": "sentinel2_nbr_change_v1",
    }
    payload_path = tmp_path / "case.json"
    payload_path.write_text(
        json.dumps({"case": base, "observations": [observation]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="cutoff"):
        load_replay_payload(payload_path)


def test_confidence_calibrator_is_deterministic_and_metrics_are_exact() -> None:
    rows = [_metric_row(index, success=index % 2 == 0) for index in range(40)]
    first = fit_confidence_calibrator(rows, calibrator_id="cal-fr-v1")
    second = fit_confidence_calibrator(rows, calibrator_id="cal-fr-v1")
    assert first.coefficients == second.coefficients
    assert first.feature_means == second.feature_means
    assert predict_confidence(first, rows[0]) > predict_confidence(first, rows[1])
    assert brier_score([0.9, 0.1], [True, False]) == pytest.approx(0.01)
    assert expected_calibration_error([0.9, 0.1], [True, False]) == pytest.approx(0.1)


def test_profile_ranking_is_incident_aggregated() -> None:
    baseline = [_metric_row(index, success=index % 2 == 0) for index in range(10)]
    improved = [_metric_row(index, success=True) for index in range(10)]
    ranked = rank_candidate_profiles({"baseline": baseline, "improved": improved})
    assert ranked[0][0] == "improved"
    assert ranked[0][1] > ranked[1][1]


def test_screening_profiles_and_grid_sweep_are_bounded() -> None:
    base = load_fusion_profile(
        "part4-baseline-v3-provenance",
        algorithm_version=FUSION_ALGORITHM_VERSION,
    )
    profiles = build_screening_profiles(base)
    assert 50 <= len(profiles) <= 100
    assert len({item.profile_sha256 for item in profiles}) == len(profiles)
    assert base.profile_id == "part4-baseline-v3-provenance"
    assert refinement_values(0.5, lower=0.35, upper=0.70) == tuple(
        sorted(refinement_values(0.5, lower=0.35, upper=0.70))
    )
    affected, active = threshold_masks(
        np.asarray([[0.49, 0.50]], dtype=np.float32),
        np.asarray([[0.59, 0.60]], dtype=np.float32),
        affected_threshold=0.50,
        active_threshold=0.60,
    )
    assert affected.tolist() == [[False, True]]
    assert active.tolist() == [[False, True]]


def test_qualification_gates_and_strict_confidence_threshold() -> None:
    rows = [_metric_row(index, success=True, confidence=0.95) for index in range(300)]
    receipt = qualification_receipt(
        qualification_id="qual-fr-v1",
        fusion_profile_id="part4-baseline-v3-provenance",
        calibrator_id="cal-fr-v1",
        campaign_id="campaign-fr-v1",
        holdout_dataset_repo="fireviewer/part4-france-holdout-v1",
        holdout_dataset_revision=HF_REVISION,
        rows=rows,
        campaign_incident_count=50,
        campaign_snapshot_count=300,
    )
    assert receipt.qualified is True
    assert all(receipt.gates.values())

    decision = Part4ComponentReleaseDecisionV1(
        decision_id="decision-1",
        decided_at=NOW,
        perimeter_receipt_id="receipt-1",
        incident_id="FR-26-00001",
        component="affected",
        qualification_revision=HF_REVISION,
        confidence_calibrated=0.85,
        reconstruction_status="fused",
        reason_codes=("confidence_not_above_0_85",),
        eligible_for_automatic_publication=False,
    )
    assert decision.eligible_for_automatic_publication is False
    with pytest.raises(ValidationError, match="eligibility"):
        Part4ComponentReleaseDecisionV1.model_validate(
            {
                **decision.model_dump(mode="json", by_alias=True),
                "confidence_calibrated": 0.86,
            }
        )


def test_bounded_scratch_is_removed(tmp_path: Path) -> None:
    with bounded_scratch(parent=tmp_path) as scratch:
        created = scratch
        assert (scratch / "scratch-policy.json").is_file()
    assert not created.exists()


def test_evaluator_opens_reference_only_after_prediction_is_frozen(tmp_path: Path) -> None:
    geometry = {
        "type": "Polygon",
        "coordinates": [[[5.0, 44.0], [5.02, 44.0], [5.02, 44.02], [5.0, 44.0]]],
    }
    observation = SpatialObservationV2(
        observation_id="official-input-1",
        observation_kind="official_perimeter",
        target_state="affected",
        observed_at=NOW,
        geometry_geojson=geometry,
        probability=1.0,
        resolution_m=20,
        source_family_id="authority:test-input",
        lineage_id="authority:test-input:revision-1",
        upstream_product_id="test-input-revision-1",
        source_revision_sha256="b" * 64,
        processor_revision="test-normalizer-v1",
        evidence_refs=("input-evidence-1",),
    )
    profile = load_fusion_profile(
        "part4-baseline-v3-provenance",
        algorithm_version=FUSION_ALGORITHM_VERSION,
    )
    fused = fuse_daily_fire_state(
        incident_id="FR-26-00001",
        episode_id="E01",
        local_date=NOW.date(),
        observations=(observation,),
        profile=profile,
    )
    state_path, grid_path = freeze_replay_result(
        fused,
        output_dir=tmp_path,
        case_id="case-frozen",
    )
    assert fused.state.perimeter.affected is not None
    reference = Part4ReferenceSnapshotV1(
        reference_id="held-out-ref-1",
        incident_id="FR-26-00001",
        episode_id="E01",
        component="affected",
        geometry_geojson=fused.state.perimeter.affected,
        valid_at=NOW,
        temporal_accuracy_seconds=3_600,
        spatial_accuracy_m=20,
        resolution_m=20,
        provider="test authority",
        product="held-out geometry",
        licence="test-only",
        source_revision="held-out-revision-1",
        grade="A",
        forbidden_input_refs=("held-out-ref-1", "held-out-revision-1"),
    )
    comparison_cache: dict[str, Mapping[str, object]] = {}
    row = evaluate_frozen_affected_prediction(
        frozen_state_path=state_path,
        frozen_grid_path=grid_path,
        reference=reference,
        comparison_cache=comparison_cache,
    )
    assert row["prediction_frozen_before_reference_open"] is True
    assert row["reference_leakage_detected"] is False
    assert row["iou"] == 1.0
    assert row["geometry_comparison_cached"] is False
    changed_state = json.loads(state_path.read_text(encoding="utf-8"))
    changed_state["perimeter"]["evidence_strength"] = 0.123
    changed_path = tmp_path / "different-evidence.state.json"
    changed_path.write_text(json.dumps(changed_state), encoding="utf-8")
    cached = evaluate_frozen_affected_prediction(
        frozen_state_path=changed_path,
        frozen_grid_path=grid_path,
        reference=reference,
        comparison_cache=comparison_cache,
    )
    assert cached["geometry_comparison_cached"] is True
    assert cached["iou"] == row["iou"]
    assert cached["evidence_strength"] == 0.123 != row["evidence_strength"]
    assert len(comparison_cache) == 1
