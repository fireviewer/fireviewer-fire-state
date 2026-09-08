from shapely.geometry import box, shape
from shapely.ops import unary_union

from fireviewer_fire_state.fire_state_fusion import (
    _canonical_multipolygon,
    _cumulative_affected_geometry,
    _projected_geojson,
)


def test_cumulative_union_preserves_wgs84_edges_without_tolerance() -> None:
    prior = _canonical_multipolygon(box(718800, 6298140, 742160, 6323660))
    current = _canonical_multipolygon(box(730480, 6298140, 765520, 6349180))
    legacy = _canonical_multipolygon(
        unary_union([_projected_geojson(prior), _projected_geojson(current)])
    )
    assert not shape(prior).within(shape(legacy).buffer(1e-7))

    result = _cumulative_affected_geometry(current, None, prior)
    assert result is not None
    assert shape(result).is_valid
    assert shape(result).covers(shape(prior))
    assert shape(prior).difference(shape(result)).is_empty
    # GEOS may represent a zero-area line difference at an intersection, but
    # no new affected surface may be lost. The prior above is preserved exactly.
    assert shape(current).difference(shape(result)).area == 0
    assert _cumulative_affected_geometry(result, None, prior) == result
    assert _cumulative_affected_geometry(prior, None, current) == result


def test_cumulative_union_keeps_prior_without_new_observations() -> None:
    prior = _canonical_multipolygon(box(718800, 6298140, 742160, 6323660))
    result = _cumulative_affected_geometry(None, None, prior)
    assert result is not None
    assert shape(result).equals(shape(prior))
    assert _cumulative_affected_geometry(None, None, None) is None
