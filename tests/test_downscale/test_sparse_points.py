"""Tests for sparse points mode (bbox-less point setup with per-group DEM patches)."""

import pytest

from topopyscale2.config.schema import (
    DomainConfig,
    PointCoordinate,
    PointsConfig,
    TPS2Config,
)
from topopyscale2.domain import Domain

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_points(*coords):
    """Create PointCoordinate list from (name, lon, lat) tuples."""
    return [PointCoordinate(name=n, lon=lo, lat=la) for n, lo, la in coords]


# ---------------------------------------------------------------------------
# Config schema: bbox=None allowed for sparse points
# ---------------------------------------------------------------------------

class TestSparsePointsConfig:
    def test_bbox_none_with_points_allowed(self):
        """bbox=None is valid when spatial_mode='points' with coordinates."""
        cfg = DomainConfig(
            spatial_mode="points",
            points=PointsConfig(coordinates=_make_points(
                ("a", 10.0, 45.0),
                ("b", 20.0, 50.0),
            )),
        )
        assert cfg.bbox is None
        assert len(cfg.points.coordinates) == 2

    def test_bbox_none_without_points_raises(self):
        """bbox=None with spatial_mode='clusters' must still raise."""
        with pytest.raises(ValueError, match="Either 'dem'"):
            DomainConfig(spatial_mode="clusters")

    def test_bbox_none_points_empty_raises(self):
        """bbox=None with spatial_mode='points' but empty coordinates must raise."""
        with pytest.raises(ValueError, match="Either 'dem'"):
            DomainConfig(
                spatial_mode="points",
                points=PointsConfig(coordinates=[]),
            )

    def test_dem_buffer_m_default(self):
        """dem_buffer_m defaults to None (auto)."""
        cfg = DomainConfig(
            spatial_mode="points",
            points=PointsConfig(coordinates=_make_points(("a", 10.0, 45.0))),
        )
        assert cfg.dem_buffer_m is None

    def test_dem_buffer_m_explicit(self):
        cfg = DomainConfig(
            spatial_mode="points",
            dem_buffer_m=5000.0,
            points=PointsConfig(coordinates=_make_points(("a", 10.0, 45.0))),
        )
        assert cfg.dem_buffer_m == 5000.0


# ---------------------------------------------------------------------------
# _group_points
# ---------------------------------------------------------------------------

class TestGroupPoints:
    def test_single_point(self):
        pts = _make_points(("a", 10.0, 45.0))
        groups = Domain._group_points(pts, threshold_m=6000.0)
        assert len(groups) == 1
        assert len(groups[0]) == 1

    def test_two_close_points_same_group(self):
        """Points ~1 km apart should be in the same group at 6 km threshold."""
        pts = _make_points(
            ("a", 10.0, 45.0),
            ("b", 10.01, 45.0),  # ~0.8 km east
        )
        groups = Domain._group_points(pts, threshold_m=6000.0)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_two_far_points_separate_groups(self):
        """Points ~1100 km apart should be in separate groups."""
        pts = _make_points(
            ("pamir", 72.5, 38.6),
            ("north_kaz", 71.4, 51.1),
        )
        groups = Domain._group_points(pts, threshold_m=6000.0)
        assert len(groups) == 2

    def test_three_points_chain_linkage(self):
        """A-B close, B-C close, A-C far → all in one group (single-linkage)."""
        # Points spaced ~4 km apart, threshold 6 km
        pts = _make_points(
            ("a", 10.0, 45.0),
            ("b", 10.05, 45.0),   # ~3.9 km from a
            ("c", 10.10, 45.0),   # ~3.9 km from b, ~7.8 km from a
        )
        groups = Domain._group_points(pts, threshold_m=6000.0)
        # Single linkage: a-b linked, b-c linked → all in one group
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_mixed_groups(self):
        """Two clusters of two points each, well separated."""
        pts = _make_points(
            ("a1", 10.0, 45.0),
            ("a2", 10.01, 45.0),
            ("b1", 30.0, 60.0),
            ("b2", 30.01, 60.0),
        )
        groups = Domain._group_points(pts, threshold_m=6000.0)
        assert len(groups) == 2
        names_per_group = [sorted(p.name for p in g) for g in groups]
        names_per_group.sort()
        assert names_per_group == [["a1", "a2"], ["b1", "b2"]]


# ---------------------------------------------------------------------------
# _compute_patch_bbox
# ---------------------------------------------------------------------------

class TestComputePatchBbox:
    def test_single_point_buffer(self):
        pts = _make_points(("a", 10.0, 45.0))
        bbox = Domain._compute_patch_bbox(pts, buffer_m=5000.0)
        w, s, e, n = bbox
        # Buffer is ~5 km; at lat 45°, 1° lon ≈ 78.7 km → 5 km ≈ 0.064°
        assert w < 10.0
        assert e > 10.0
        assert s < 45.0
        assert n > 45.0
        # Symmetric
        assert abs((10.0 - w) - (e - 10.0)) < 1e-6
        assert abs((45.0 - s) - (n - 45.0)) < 1e-6

    def test_buffer_size_reasonable(self):
        """5 km buffer at lat 45° should be ~0.045° in latitude."""
        pts = _make_points(("a", 10.0, 45.0))
        bbox = Domain._compute_patch_bbox(pts, buffer_m=5000.0)
        lat_buf = bbox[3] - 45.0
        # 5000 m / 111320 m/deg ≈ 0.0449°
        assert abs(lat_buf - 0.0449) < 0.001

    def test_two_points_bbox_spans_both(self):
        pts = _make_points(("a", 10.0, 45.0), ("b", 10.5, 45.5))
        bbox = Domain._compute_patch_bbox(pts, buffer_m=1000.0)
        w, s, e, n = bbox
        assert w < 10.0
        assert e > 10.5
        assert s < 45.0
        assert n > 45.5


# ---------------------------------------------------------------------------
# _get_bbox fallback for sparse points
# ---------------------------------------------------------------------------

class TestGetBboxSparsePoints:
    def test_get_bbox_from_points(self):
        """_get_bbox should derive bbox from point coordinates when bbox=None."""
        cfg = TPS2Config(
            domain=DomainConfig(
                spatial_mode="points",
                points=PointsConfig(coordinates=_make_points(
                    ("a", 10.0, 45.0),
                    ("b", 20.0, 55.0),
                )),
            ),
        )
        domain = Domain(cfg)
        bbox = domain._get_bbox()
        # Should be point extent + 1° ERA5 padding
        assert bbox == (9.0, 44.0, 21.0, 56.0)


# ---------------------------------------------------------------------------
# _domain_config_hash includes points
# ---------------------------------------------------------------------------

class TestDomainConfigHash:
    def test_hash_changes_with_points(self):
        """Config hash must change when point coordinates change."""
        cfg1 = TPS2Config(
            domain=DomainConfig(
                spatial_mode="points",
                points=PointsConfig(coordinates=_make_points(("a", 10.0, 45.0))),
            ),
        )
        cfg2 = TPS2Config(
            domain=DomainConfig(
                spatial_mode="points",
                points=PointsConfig(coordinates=_make_points(("a", 20.0, 55.0))),
            ),
        )
        d1 = Domain(cfg1)
        d2 = Domain(cfg2)
        assert d1._domain_config_hash() != d2._domain_config_hash()

    def test_hash_stable_same_config(self):
        cfg = TPS2Config(
            domain=DomainConfig(
                spatial_mode="points",
                points=PointsConfig(coordinates=_make_points(("a", 10.0, 45.0))),
            ),
        )
        d = Domain(cfg)
        assert d._domain_config_hash() == d._domain_config_hash()


# ---------------------------------------------------------------------------
# Point units must live in the DEM CRS (projected-domain regression)
# ---------------------------------------------------------------------------

class TestPointUnitsCRS:
    """Points are configured as lon/lat but centroids must be in the DEM CRS.

    The downscaler reprojects centroids *from* ``dem_data.attrs['crs']`` to pick
    ERA5 cells, so storing raw lon/lat under a projected domain CRS makes it read
    degrees as metres: every point collapses onto the projection origin and all
    units draw the same ERA5 column.
    """

    # Central Asia albers-equal-area, as the CA snowmapper uses
    AEA = ("+proj=aea +lat_1=35 +lat_2=42 +lat_0=38 +lon_0=70 "
           "+x_0=0 +y_0=0 +datum=WGS84 +units=m")

    def _domain_with_dem(self, crs, points):
        import numpy as np
        import xarray as xr

        cfg = TPS2Config(
            domain=DomainConfig(
                bbox=[61.0, 33.6, 79.3, 43.1],
                spatial_mode="points",
                crs=crs,
                points=PointsConfig(coordinates=points),
            )
        )
        d = Domain(cfg)
        # A DEM spanning the projected domain; values are irrelevant here.
        if crs is None:
            x = np.linspace(61.0, 79.3, 40)
            y = np.linspace(33.6, 43.1, 40)
        else:
            x = np.linspace(-4.0e5, 4.0e5, 40)
            y = np.linspace(-5.0e5, 5.0e5, 40)
        d.dem_data = xr.Dataset(
            {"elevation": (["y", "x"], np.zeros((40, 40)))},
            coords={"x": x, "y": y},
            attrs={"crs": crs} if crs else {},
        )
        return d

    def test_projected_crs_keeps_points_distinct(self):
        pts = _make_points(("a", 69.0, 38.7), ("b", 75.4, 35.3), ("c", 78.1, 42.5))
        units = self._domain_with_dem(self.AEA, pts)._create_point_units()

        xs = [u.x for u in units]
        ys = [u.y for u in units]
        # Distinct points must stay distinct, and land in metres not degrees.
        assert len(set(xs)) == 3
        assert max(xs) - min(xs) > 100_000  # hundreds of km apart
        assert any(abs(v) > 1000 for v in xs + ys)

    def test_projected_crs_roundtrips_to_input_lonlat(self):
        from pyproj import CRS, Transformer

        pts = _make_points(("a", 69.0, 38.7), ("b", 75.4, 35.3))
        units = self._domain_with_dem(self.AEA, pts)._create_point_units()

        back = Transformer.from_crs(
            CRS.from_user_input(self.AEA), CRS.from_epsg(4326), always_xy=True
        )
        for unit, pt in zip(units, pts):
            lon, lat = back.transform(unit.x, unit.y)
            assert lon == pytest.approx(pt.lon, abs=1e-6)
            assert lat == pytest.approx(pt.lat, abs=1e-6)

    def test_geographic_crs_is_unchanged(self):
        """A geographic domain needs no conversion — lon/lat pass through."""
        pts = _make_points(("a", 69.0, 38.7), ("b", 75.4, 35.3))
        units = self._domain_with_dem(None, pts)._create_point_units()

        assert [u.x for u in units] == [69.0, 75.4]
        assert [u.y for u in units] == [38.7, 35.3]
