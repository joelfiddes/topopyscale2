"""Tests for bilinear horizontal interpolation of ERA5 grid cells."""

import numpy as np
import pytest

from topopyscale2.core.downscale import Downscaler
from topopyscale2.spatial.units import SpatialUnit


def _make_units(coords):
    """Create SpatialUnit list from (lon, lat, elev) tuples."""
    return [
        SpatialUnit(f"u{i}", centroid=(lon, lat, elev))
        for i, (lon, lat, elev) in enumerate(coords)
    ]


class TestComputeBilinearWeights:
    """Tests for Downscaler.compute_bilinear_weights()."""

    def test_exact_grid_point_gets_weight_one(self):
        """Unit exactly on a grid node should get w00=1, rest=0."""
        lat = np.array([42.0, 42.25, 42.5, 42.75, 43.0])
        lon = np.array([76.0, 76.25, 76.5, 76.75, 77.0])
        units = _make_units([(76.5, 42.5, 3000)])  # Exactly on grid node

        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        # One weight should be 1.0, rest 0.0
        weights = [w["w00"][0], w["w01"][0], w["w10"][0], w["w11"][0]]
        assert max(weights) == pytest.approx(1.0)
        assert sum(weights) == pytest.approx(1.0)
        # The exact node should have weight 1.0
        assert min(weights) == pytest.approx(0.0, abs=1e-15)

    def test_midpoint_equal_weights(self):
        """Unit at center of 4 cells should have equal weights (0.25 each)."""
        lat = np.array([42.0, 42.5])
        lon = np.array([76.0, 76.5])
        units = _make_units([(76.25, 42.25, 3000)])  # Center of the 4 cells

        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        np.testing.assert_allclose(w["w00"][0], 0.25)
        np.testing.assert_allclose(w["w01"][0], 0.25)
        np.testing.assert_allclose(w["w10"][0], 0.25)
        np.testing.assert_allclose(w["w11"][0], 0.25)

    def test_weights_sum_to_one(self):
        """Weights should sum to 1.0 for any unit position."""
        lat = np.array([40.0, 41.0, 42.0, 43.0, 44.0])
        lon = np.array([75.0, 76.0, 77.0, 78.0, 79.0])
        units = _make_units([
            (76.3, 42.7, 2000),
            (75.1, 40.9, 3000),
            (78.8, 43.2, 1500),
            (77.0, 42.0, 2500),  # Exact grid point
        ])

        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        weight_sums = w["w00"] + w["w01"] + w["w10"] + w["w11"]
        np.testing.assert_allclose(weight_sums, 1.0, atol=1e-14)

    def test_descending_latitude(self):
        """Descending latitude should produce the same weights as ascending."""
        lat_asc = np.array([42.0, 42.25, 42.5, 42.75, 43.0])
        lat_desc = lat_asc[::-1]
        lon = np.array([76.0, 76.25, 76.5, 76.75, 77.0])
        units = _make_units([(76.3, 42.6, 3000)])

        w_asc = Downscaler.compute_bilinear_weights(units, lat_asc, lon)
        w_desc = Downscaler.compute_bilinear_weights(units, lat_desc, lon)

        # Weights should be identical (the indices will differ but weights match)
        np.testing.assert_allclose(w_asc["w00"], w_desc["w00"], atol=1e-14)
        np.testing.assert_allclose(w_asc["w01"], w_desc["w01"], atol=1e-14)
        np.testing.assert_allclose(w_asc["w10"], w_desc["w10"], atol=1e-14)
        np.testing.assert_allclose(w_asc["w11"], w_desc["w11"], atol=1e-14)

    def test_boundary_clipping(self):
        """Unit beyond grid edge should be clipped; weights still sum to 1."""
        lat = np.array([42.0, 42.25, 42.5])
        lon = np.array([76.0, 76.25, 76.5])
        # Unit beyond north edge and east edge
        units = _make_units([(76.8, 42.8, 3000)])

        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        weight_sum = w["w00"][0] + w["w01"][0] + w["w10"][0] + w["w11"][0]
        assert weight_sum == pytest.approx(1.0)

    def test_multiple_units(self):
        """Vectorized computation for multiple units."""
        lat = np.array([42.0, 42.5, 43.0])
        lon = np.array([76.0, 76.5, 77.0])
        units = _make_units([
            (76.25, 42.25, 2000),
            (76.75, 42.75, 3000),
        ])

        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        # Both should have shape (2,)
        assert w["w00"].shape == (2,)
        assert w["lat_idx0"].shape == (2,)

        # Both should sum to 1
        sums = w["w00"] + w["w01"] + w["w10"] + w["w11"]
        np.testing.assert_allclose(sums, 1.0, atol=1e-14)


class TestInterpolateBilinearUnit:
    """Tests for Downscaler.interpolate_bilinear_unit()."""

    def test_linear_field_exact(self):
        """Bilinear interp of f = 2*lat + 3*lon should be exact at any point."""
        lat = np.array([42.0, 42.5, 43.0])
        lon = np.array([76.0, 76.5, 77.0])
        nt = 5

        # f(lat, lon) = 2*lat + 3*lon, broadcast to (time, lat, lon)
        lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
        field_2d = 2.0 * lat_grid + 3.0 * lon_grid
        field = np.broadcast_to(field_2d[np.newaxis, :, :], (nt, len(lat), len(lon))).copy()

        # Test at an arbitrary interior point
        test_lon, test_lat = 76.3, 42.7
        units = _make_units([(test_lon, test_lat, 3000)])
        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        result = Downscaler.interpolate_bilinear_unit(field, w, 0)

        expected = 2.0 * test_lat + 3.0 * test_lon
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_4d_field(self):
        """4D field (time, level, lat, lon) should return (time, level)."""
        lat = np.array([42.0, 42.5])
        lon = np.array([76.0, 76.5])
        nt, nl = 4, 3

        # Midpoint of 4 cells: result should be mean of 4 corners
        field = np.random.RandomState(42).rand(nt, nl, 2, 2)
        units = _make_units([(76.25, 42.25, 3000)])
        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        result = Downscaler.interpolate_bilinear_unit(field, w, 0)

        assert result.shape == (nt, nl)
        # At midpoint, all weights = 0.25, so result = mean of 4 corners
        expected = (field[:, :, 0, 0] + field[:, :, 0, 1] +
                    field[:, :, 1, 0] + field[:, :, 1, 1]) / 4.0
        np.testing.assert_allclose(result, expected, atol=1e-14)

    def test_grid_node_returns_exact_value(self):
        """Unit on exact grid node should return the field value at that node."""
        lat = np.array([42.0, 42.5, 43.0])
        lon = np.array([76.0, 76.5, 77.0])
        nt = 3

        field = np.random.RandomState(7).rand(nt, 3, 3)
        # Unit at (76.5, 42.5) = grid node [1, 1]
        units = _make_units([(76.5, 42.5, 3000)])
        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        result = Downscaler.interpolate_bilinear_unit(field, w, 0)

        np.testing.assert_allclose(result, field[:, 1, 1], atol=1e-14)

    def test_invalid_ndim_raises(self):
        """Field with wrong number of dimensions should raise ValueError."""
        lat = np.array([42.0, 42.5])
        lon = np.array([76.0, 76.5])
        units = _make_units([(76.25, 42.25, 3000)])
        w = Downscaler.compute_bilinear_weights(units, lat, lon)

        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            Downscaler.interpolate_bilinear_unit(np.array([1.0, 2.0]), w, 0)


class TestDescendingLatInterpolation:
    """Test that descending latitude arrays produce identical interpolation results."""

    def test_descending_lat_interpolation_matches(self):
        """Same physical field in ascending vs descending lat order → identical result."""
        lat_asc = np.array([42.0, 42.5, 43.0])
        lat_desc = lat_asc[::-1]
        lon = np.array([76.0, 76.5, 77.0])
        nt = 5

        # Create field with known structure: f = lat^2 + lon
        lat_g, lon_g = np.meshgrid(lat_asc, lon, indexing="ij")
        field_2d = lat_g**2 + lon_g
        field_asc = np.broadcast_to(field_2d[np.newaxis, :, :], (nt, 3, 3)).copy()
        # Descending: flip the lat axis
        field_desc = field_asc[:, ::-1, :]

        test_lon, test_lat = 76.3, 42.7
        units = _make_units([(test_lon, test_lat, 3000)])

        w_asc = Downscaler.compute_bilinear_weights(units, lat_asc, lon)
        w_desc = Downscaler.compute_bilinear_weights(units, lat_desc, lon)

        result_asc = Downscaler.interpolate_bilinear_unit(field_asc, w_asc, 0)
        result_desc = Downscaler.interpolate_bilinear_unit(field_desc, w_desc, 0)

        np.testing.assert_allclose(result_asc, result_desc, atol=1e-10)
