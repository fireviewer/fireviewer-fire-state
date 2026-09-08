from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon

from fireviewer_contracts.backend.part4_calibration_schemas import Part4ReferenceSnapshotV1
from fireviewer_fire_state.part4_calibration_evaluation import (
    _boundary_coverage,
    _geometry_valid,
    _grid_features,
    _metric_float,
    _polygonal_result,
    _source_resolution,
    _unjustified_regression,
    simplify_reference_for_evaluation,
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


def _reference() -> Part4ReferenceSnapshotV1:
    return Part4ReferenceSnapshotV1(
        reference_id="REF-EDGE",
        incident_id="INC-EDGE",
        component="affected",
        geometry_geojson=_polygon(5.10, 5.11),
        valid_at=datetime(2026, 7, 7, 12, tzinfo=UTC),
        temporal_accuracy_seconds=3_600,
        spatial_accuracy_m=20.0,
        resolution_m=20.0,
        provider="Test authority",
        product="Test perimeter",
        licence="Test-only fixture",
        source_revision="reference-edge",
        grade="A",
        forbidden_input_refs=("REF-EDGE",),
    )


def test_evaluation_grid_receipts_and_scalar_guards(tmp_path: Path) -> None:
    assert _grid_features(None) == (0.0, 0.0)
    invalid = tmp_path / "invalid.support.json"
    invalid.write_text('{"schema":"wrong"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="support receipt is invalid"):
        _grid_features(invalid)

    support = tmp_path / "valid.support.json"
    support.write_text(
        json.dumps(
            {
                "schema": "fireviewer.part4-frozen-grid-support.v1",
                "observable_fraction": 0.75,
                "uncertainty_fraction": 0.25,
            }
        ),
        encoding="utf-8",
    )
    assert _grid_features(support) == (0.75, 0.25)
    assert _source_resolution(support) is None
    assert _source_resolution(None) is None
    support.write_text(json.dumps({"best_observation_resolution_m": 375.0}), encoding="utf-8")
    assert _source_resolution(support) == 375.0
    for invalid_resolution in (-1, True, "20", 0, float("nan")):
        support.write_text(
            json.dumps({"best_observation_resolution_m": invalid_resolution}), encoding="utf-8"
        )
        with pytest.raises(ValueError, match="resolution receipt"):
            _source_resolution(support)

    grid = tmp_path / "state.npz"
    np.savez_compressed(
        grid,
        observable=np.asarray([[2.0, -1.0], [0.5, 1.0]], dtype=np.float32),
        uncertainty_support=np.asarray([[0, 1], [1, 0]], dtype=np.float32),
    )
    assert _grid_features(grid) == (0.625, 0.5)
    assert _metric_float({"score": 4}, "score") == 4.0
    assert _metric_float({"score": True}, "score") == 0.0
    assert _metric_float({}, "score") == 0.0


def test_evaluation_geometry_and_regression_guards() -> None:
    small = _polygon(5.10, 5.11)
    large = _polygon(5.09, 5.12)
    assert _geometry_valid(small) is True
    assert _geometry_valid(None) is False
    assert _geometry_valid({"type": "invalid"}) is False
    assert _polygonal_result(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])) is not None
    assert _polygonal_result(Point(0, 0)) is None
    collection = GeometryCollection(
        [Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), LineString([(0, 0), (1, 1)])]
    )
    assert _polygonal_result(collection) is not None

    assert _boundary_coverage(None, small) == 0.0
    assert 0.0 <= _boundary_coverage(large, small) <= 1.0
    assert _unjustified_regression(small, None, correction_recorded=False) is False
    assert _unjustified_regression(small, large, correction_recorded=True) is False
    assert _unjustified_regression(small, large, correction_recorded=False) is True
    assert _unjustified_regression(large, small, correction_recorded=False) is False
    assert (
        _unjustified_regression(
            {"type": "Point", "coordinates": [5.1, 44.7]},
            small,
            correction_recorded=False,
        )
        is True
    )


def test_reference_simplification_records_resolution_floor() -> None:
    simplified, tolerance = simplify_reference_for_evaluation(_reference())
    assert tolerance == 20.0
    assert simplified.geometry_geojson is not None
    missing = _reference().model_copy(update={"geometry_geojson": None})
    unchanged, missing_tolerance = simplify_reference_for_evaluation(missing)
    assert unchanged is missing
    assert missing_tolerance == 0.0
