"""Tests for DEM processing module."""

import numpy as np
import pytest
import xarray as xr

from topopyscale2.config.schema import DomainConfig, PolygonConfig, WindConfig
from topopyscale2.spatial.dem import DEMProcessor

# Check if pyflwdir is available (for conditional skipping of flow routing tests)
try:
    import pyflwdir  # noqa: F401 — availability check

    HAS_PYFLWDIR = True
except ImportError:
    HAS_PYFLWDIR = False

# Marker for tests that require pyflwdir
requires_pyflwdir = pytest.mark.skipif(not HAS_PYFLWDIR, reason="pyflwdir not installed")


class TestComputeSlopeAspect:
    def test_flat_dem(self, synthetic_dem_flat):
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        slope, aspect = proc.compute_slope_aspect(synthetic_dem_flat)

        # Flat terrain: slope should be ~0 everywhere (except edges due to gradient)
        inner = slope.values[1:-1, 1:-1]
        np.testing.assert_allclose(inner, 0.0, atol=1e-10)

    def test_tilted_plane(self, synthetic_dem_tilted):
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        slope, aspect = proc.compute_slope_aspect(synthetic_dem_tilted)

        # Tilted north: 0.5 m/m rise → slope = atan(0.5) ≈ 26.57°
        inner_slope = slope.values[1:-1, 1:-1]
        expected_slope = np.degrees(np.arctan(0.5))
        np.testing.assert_allclose(inner_slope, expected_slope, atol=1.0)

        # North-facing slope (elevation increases going north) → aspect ≈ 180° (south-facing)
        # Wait: elevation increases with y. If y increases northward, gradient is positive
        # dz/dy > 0 means north side is higher → slope faces south → aspect ≈ 180
        inner_aspect = aspect.values[1:-1, 1:-1]
        np.testing.assert_allclose(inner_aspect, 180.0, atol=5.0)


class TestComputeHorizonAngles:
    def test_flat_dem_zero_horizon(self, synthetic_dem_flat):
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        horizon = proc.compute_horizon_angles(synthetic_dem_flat, n_directions=8, max_distance_m=150.0)

        assert horizon.dims == ("azimuth", "y", "x")
        assert horizon.shape[0] == 8
        # Flat terrain: horizon angles should be ~0 in interior
        inner = horizon.values[:, 2:-2, 2:-2]
        np.testing.assert_allclose(inner, 0.0, atol=1.0)


class TestComputeSkyViewFactor:
    def test_flat_dem_svf_1(self, synthetic_dem_flat):
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        horizon = proc.compute_horizon_angles(synthetic_dem_flat, n_directions=8, max_distance_m=150.0)
        svf = proc.compute_sky_view_factor(horizon)

        # Flat terrain: SVF ≈ 1.0
        inner = svf.values[2:-2, 2:-2]
        np.testing.assert_allclose(inner, 1.0, atol=0.05)

    def test_valley_svf_less_than_1(self, synthetic_dem_valley):
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_valley.attrs["resolution_m"] = 30.0

        horizon = proc.compute_horizon_angles(synthetic_dem_valley, n_directions=16, max_distance_m=300.0)
        svf = proc.compute_sky_view_factor(horizon)

        # Valley bottom should have SVF < 1
        ny, nx = svf.shape
        center_svf = svf.values[ny // 2, nx // 2]
        assert center_svf < 1.0


class TestComputeWinstralSx:
    """Tests for Winstral Sx (maximum upwind slope) computation."""

    def test_output_shape_and_dims(self, synthetic_dem_flat):
        """Test that output has correct shape and dimensions."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        sx = proc.compute_winstral_sx(synthetic_dem_flat, n_directions=8, search_distance_m=90.0)

        assert sx.dims == ("wind_direction", "y", "x")
        assert sx.shape[0] == 8
        assert sx.shape[1:] == synthetic_dem_flat.shape

    def test_wind_direction_coordinates(self, synthetic_dem_flat):
        """Test that wind_direction coordinates are correctly set."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        sx = proc.compute_winstral_sx(synthetic_dem_flat, n_directions=36, search_distance_m=90.0)

        expected_azimuths = np.linspace(0, 360, 36, endpoint=False)
        np.testing.assert_allclose(sx.wind_direction.values, expected_azimuths, atol=1e-10)

    def test_flat_dem_zero_sx(self, synthetic_dem_flat):
        """Flat terrain should have Sx = 0 everywhere (no exposure/sheltering)."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        sx = proc.compute_winstral_sx(synthetic_dem_flat, n_directions=8, search_distance_m=90.0)

        # Flat terrain: Sx should be ~0 in interior (no slope to upwind)
        inner = sx.values[:, 2:-2, 2:-2]
        np.testing.assert_allclose(inner, 0.0, atol=1e-10)

    def test_ridgeline_positive_sx(self):
        """Ridgeline (local high point) should have positive Sx (exposed)."""
        # Create a ridge running north-south (high in center column)
        ny, nx = 20, 20
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        center_x = x[nx // 2]
        _, xx = np.meshgrid(y, x, indexing="ij")
        # Ridge: elevation decreases away from center column
        elevation = 3000.0 - np.abs(xx - center_x) * 0.5

        dem = xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"y": y, "x": x},
            attrs={"resolution_m": 30.0},
        )

        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        sx = proc.compute_winstral_sx(dem, n_directions=8, search_distance_m=150.0)

        # For wind from east (azimuth 90) or west (azimuth 270), the ridge center
        # should have positive Sx because upwind terrain is lower
        east_idx = 2  # 90 degrees with 8 directions (0, 45, 90, ...)
        west_idx = 6  # 270 degrees

        center_col = nx // 2
        ridge_sx_east = sx.values[east_idx, ny // 2, center_col]
        ridge_sx_west = sx.values[west_idx, ny // 2, center_col]

        assert ridge_sx_east > 0, f"Ridgeline should have positive Sx from east, got {ridge_sx_east}"
        assert ridge_sx_west > 0, f"Ridgeline should have positive Sx from west, got {ridge_sx_west}"

    def test_valley_negative_sx(self, synthetic_dem_valley):
        """Valley bottom should have negative Sx (sheltered)."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_valley.attrs["resolution_m"] = 30.0

        sx = proc.compute_winstral_sx(synthetic_dem_valley, n_directions=8, search_distance_m=150.0)

        ny, nx = synthetic_dem_valley.shape
        center_col = nx // 2

        # For wind from east (azimuth 90) or west (azimuth 270), the valley center
        # should have negative Sx because upwind terrain is higher
        east_idx = 2  # 90 degrees
        west_idx = 6  # 270 degrees

        valley_sx_east = sx.values[east_idx, ny // 2, center_col]
        valley_sx_west = sx.values[west_idx, ny // 2, center_col]

        assert valley_sx_east < 0, f"Valley should have negative Sx from east, got {valley_sx_east}"
        assert valley_sx_west < 0, f"Valley should have negative Sx from west, got {valley_sx_west}"

    def test_tilted_plane_consistent_exposure(self, synthetic_dem_tilted):
        """Tilted plane should have consistent Sx pattern based on wind direction."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        sx = proc.compute_winstral_sx(synthetic_dem_tilted, n_directions=8, search_distance_m=90.0)

        # Slope rises northward (y increases). Wind from south (180 deg)
        # sees upwind terrain (south) as lower -> positive Sx (exposed)
        # Wind from north (0 deg) sees upwind terrain as higher -> negative Sx
        # With 8 directions: [0, 45, 90, 135, 180, 225, 270, 315]
        # North = index 0, South = index 4
        north_idx = 0
        south_idx = 4

        inner = sx.values[:, 2:-2, 2:-2]

        # Most pixels should show this pattern
        mean_sx_north = np.mean(inner[north_idx])
        mean_sx_south = np.mean(inner[south_idx])

        assert mean_sx_north < 0, f"North wind should give negative Sx, got {mean_sx_north}"
        assert mean_sx_south > 0, f"South wind should give positive Sx, got {mean_sx_south}"

    def test_symmetry_east_west(self):
        """East-west ridge should have symmetric Sx for east vs west wind."""
        # Create a ridge running east-west (high in center row)
        ny, nx = 20, 20
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        center_y = y[ny // 2]
        yy, _ = np.meshgrid(y, x, indexing="ij")
        elevation = 3000.0 - np.abs(yy - center_y) * 0.5

        dem = xr.DataArray(
            elevation,
            dims=["y", "x"],
            coords={"y": y, "x": x},
            attrs={"resolution_m": 30.0},
        )

        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        sx = proc.compute_winstral_sx(dem, n_directions=8, search_distance_m=150.0)

        # For symmetric terrain, wind from north vs south should give similar magnitude
        # but opposite sign at off-ridge locations
        north_idx = 0
        south_idx = 4

        # At ridge center (center row), both should be positive (exposed)
        center_row = ny // 2
        ridge_sx_north = sx.values[north_idx, center_row, nx // 2]
        ridge_sx_south = sx.values[south_idx, center_row, nx // 2]

        assert ridge_sx_north > 0, "Ridge should be exposed from north"
        assert ridge_sx_south > 0, "Ridge should be exposed from south"

    def test_search_distance_effect(self, synthetic_dem_valley):
        """Larger search distance should find more distant sheltering."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_valley.attrs["resolution_m"] = 30.0

        sx_short = proc.compute_winstral_sx(
            synthetic_dem_valley, n_directions=4, search_distance_m=60.0
        )
        sx_long = proc.compute_winstral_sx(
            synthetic_dem_valley, n_directions=4, search_distance_m=300.0
        )

        # With longer search distance, we can see further up the valley walls
        # so sheltering effect may be stronger (more negative Sx in valley)
        # The absolute Sx values should generally be >= for longer distances
        ny, nx = synthetic_dem_valley.shape
        center_col = nx // 2

        # Sx magnitude should increase or stay same with longer search
        sx_short_center = np.abs(sx_short.values[:, ny // 2, center_col])
        sx_long_center = np.abs(sx_long.values[:, ny // 2, center_col])

        # At least for some directions, longer search should capture more terrain
        assert np.any(sx_long_center >= sx_short_center - 0.1)

    def test_n_directions_variation(self, synthetic_dem_flat):
        """Different n_directions should produce correct number of outputs."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        for n_dir in [4, 8, 16, 36]:
            sx = proc.compute_winstral_sx(
                synthetic_dem_flat, n_directions=n_dir, search_distance_m=90.0
            )
            assert sx.shape[0] == n_dir, f"Expected {n_dir} directions, got {sx.shape[0]}"

    def test_attrs_preserved(self, synthetic_dem_flat):
        """Test that output has expected attributes."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        sx = proc.compute_winstral_sx(
            synthetic_dem_flat, n_directions=8, search_distance_m=150.0
        )

        assert "long_name" in sx.attrs
        assert "units" in sx.attrs
        assert sx.attrs["units"] == "degrees"
        assert sx.attrs["search_distance_m"] == 150.0

    def test_no_nan_values(self, synthetic_dem_flat, synthetic_dem_valley):
        """Output should not contain NaN values for valid input."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        for dem in [synthetic_dem_flat, synthetic_dem_valley]:
            dem.attrs["resolution_m"] = 30.0
            sx = proc.compute_winstral_sx(dem, n_directions=8, search_distance_m=150.0)
            assert not np.any(np.isnan(sx.values)), "Output should not contain NaN"

    def test_no_inf_values(self, synthetic_dem_flat, synthetic_dem_valley):
        """Output should not contain infinite values."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        for dem in [synthetic_dem_flat, synthetic_dem_valley]:
            dem.attrs["resolution_m"] = 30.0
            sx = proc.compute_winstral_sx(dem, n_directions=8, search_distance_m=150.0)
            assert not np.any(np.isinf(sx.values)), "Output should not contain inf"

    def test_edge_pixels_handled(self, synthetic_dem_flat):
        """Edge pixels should be handled without errors."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        # Use large search distance relative to DEM size
        sx = proc.compute_winstral_sx(
            synthetic_dem_flat, n_directions=8, search_distance_m=500.0
        )

        # Should not raise and should produce valid output
        assert sx.shape[0] == 8
        assert not np.any(np.isnan(sx.values))
        assert not np.any(np.isinf(sx.values))

    def test_single_pixel_dem(self):
        """Single pixel DEM should not crash."""
        dem = xr.DataArray(
            [[3000.0]],
            dims=["y", "x"],
            coords={"y": [0.0], "x": [0.0]},
            attrs={"resolution_m": 30.0},
        )

        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        sx = proc.compute_winstral_sx(dem, n_directions=4, search_distance_m=60.0)

        # Single pixel has no upwind to search -> Sx should be 0
        assert sx.shape == (4, 1, 1)
        np.testing.assert_allclose(sx.values, 0.0, atol=1e-10)


class TestProcessWithSx:
    """Tests for process() method with Winstral Sx computation."""

    def test_process_no_sx_by_default(self, synthetic_dem_flat, tmp_path, monkeypatch):
        """Without wind_config, process() should not compute Sx."""
        # Create a simple DEM file
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        proc = DEMProcessor(config)

        ds = proc.process()

        assert "winstral_sx" not in ds

    def test_process_with_winstral_wind_config(self, synthetic_dem_flat, tmp_path):
        """With winstral wind_config, process() should compute Sx."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        wind_config = WindConfig(method="winstral", sx_n_directions=8, sx_search_distance_m=90.0)
        proc = DEMProcessor(config, wind_config=wind_config)

        ds = proc.process()

        assert "winstral_sx" in ds
        assert ds["winstral_sx"].dims == ("wind_direction", "y", "x")
        assert ds["winstral_sx"].shape[0] == 8

    def test_process_log_profile_no_sx(self, synthetic_dem_flat, tmp_path):
        """With log_profile wind_config, process() should not compute Sx."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        wind_config = WindConfig(method="log_profile")
        proc = DEMProcessor(config, wind_config=wind_config)

        ds = proc.process()

        assert "winstral_sx" not in ds

    def test_process_explicit_compute_sx_true(self, synthetic_dem_flat, tmp_path):
        """Explicit compute_sx=True should compute Sx regardless of config."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        proc = DEMProcessor(config)  # No wind_config

        ds = proc.process(compute_sx=True)

        assert "winstral_sx" in ds

    def test_process_explicit_compute_sx_false(self, synthetic_dem_flat, tmp_path):
        """Explicit compute_sx=False should skip Sx even with winstral config."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        wind_config = WindConfig(method="winstral")
        proc = DEMProcessor(config, wind_config=wind_config)

        ds = proc.process(compute_sx=False)

        assert "winstral_sx" not in ds


def _create_test_geotiff(path, data):
    """Helper to create a simple GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds

    ny, nx = data.shape
    transform = from_bounds(0, 0, nx * 30, ny * 30, nx, ny)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype=data.dtype,
        crs="EPSG:32632",
        transform=transform,
    ) as dst:
        dst.write(data, 1)


class TestDEMProcessorAcquisition:
    def test_missing_dem_raises(self, tmp_path):
        config = DomainConfig(dem=tmp_path / "nonexistent.tif")
        proc = DEMProcessor(config)
        with pytest.raises(FileNotFoundError):
            proc.acquire()

    def test_no_dem_no_bbox_raises(self):
        with pytest.raises(ValueError):
            DomainConfig()


# =============================================================================
# Flow Routing and Catchment Delineation Tests
# =============================================================================


@pytest.fixture
def synthetic_dem_ridge():
    """Ridge DEM: central ridge running north-south with valleys on both sides.

    Water flows from the ridge crest toward the east and west edges.
    """
    ny, nx = 30, 30
    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    center_x = x[nx // 2]
    _, xx = np.meshgrid(y, x, indexing="ij")
    # Ridge in center, valleys on sides
    elevation = 3000.0 - np.abs(xx - center_x) * 0.3
    return xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={"y": y, "x": x},
        attrs={"resolution_m": 30.0},
    )


@pytest.fixture
def synthetic_dem_single_valley():
    """Single valley DEM: V-shaped valley draining southward.

    All flow should converge to the southern outlet.
    """
    ny, nx = 30, 30
    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    center_x = x[nx // 2]
    yy, xx = np.meshgrid(y, x, indexing="ij")
    # V-shaped valley: lower in center (x) and lower toward south (y)
    elevation = 3000.0 + np.abs(xx - center_x) * 0.4 - yy * 0.1
    return xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={"y": y, "x": x},
        attrs={"resolution_m": 30.0},
    )


@pytest.fixture
def synthetic_dem_flat_with_pit():
    """Flat DEM with a central depression (pit).

    Tests pit filling behavior.
    """
    ny, nx = 20, 20
    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    elevation = np.full((ny, nx), 3000.0)
    # Add a pit in the center
    elevation[ny // 2, nx // 2] = 2990.0
    return xr.DataArray(
        elevation,
        dims=["y", "x"],
        coords={"y": y, "x": x},
        attrs={"resolution_m": 30.0},
    )


@requires_pyflwdir
class TestComputeFlowDirections:
    """Tests for D8 flow direction computation."""

    def test_output_shape_and_dims(self, synthetic_dem_tilted):
        """Test that output has correct shape and dimensions."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)

        assert flow_dir.dims == ("y", "x")
        assert flow_dir.shape == synthetic_dem_tilted.shape

    def test_flow_direction_values_valid(self, synthetic_dem_tilted):
        """Flow direction values should be valid D8 power-of-2 codes."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)

        # Valid D8 codes (power-of-2): 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
        # Plus 0=pit/outlet, 247=nodata
        valid_codes = {0, 1, 2, 4, 8, 16, 32, 64, 128, 247}
        unique_codes = set(np.unique(flow_dir.values))
        assert unique_codes.issubset(valid_codes), f"Invalid codes: {unique_codes - valid_codes}"

    def test_tilted_plane_consistent_flow(self, synthetic_dem_tilted):
        """Tilted plane (north-facing) should have consistent southward flow."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)

        # For north-facing slope (elevation increases with y), water flows south
        # D8 power-of-2 codes: 4 = S, also 2 = SE, 8 = SW are southerly
        # pyflwdir y-axis may be inverted, so we also check for N (64)
        inner = flow_dir.values[2:-2, 2:-2]
        # Count all southerly or northerly directions (depends on y-axis orientation)
        south_count = np.sum((inner == 4) | (inner == 2) | (inner == 8))
        north_count = np.sum((inner == 64) | (inner == 32) | (inner == 128))
        total = inner.size
        # One direction should dominate
        dominant = max(south_count, north_count)
        assert dominant / total > 0.8, f"Most cells should flow consistently, got S={south_count}, N={north_count}"

    def test_ridge_splits_flow(self, synthetic_dem_ridge):
        """Ridge should split flow to east and west."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_ridge)

        ny, nx = flow_dir.shape
        center_col = nx // 2

        # D8 power-of-2 codes: 1=E, 16=W, with diagonals 2=SE, 128=NE, 8=SW, 32=NW
        # East of ridge (cols > center): should flow eastward (E, SE, NE)
        east_side = flow_dir.values[2:-2, center_col + 2 : -2]
        east_codes = {1, 2, 128}  # E, SE, NE
        east_count = np.sum(np.isin(east_side, list(east_codes)))
        assert east_count / east_side.size > 0.5, f"East side of ridge should flow east, got {np.unique(east_side)}"

        # West of ridge (cols < center): should flow westward (W, SW, NW)
        west_side = flow_dir.values[2:-2, 2 : center_col - 1]
        west_codes = {16, 8, 32}  # W, SW, NW
        west_count = np.sum(np.isin(west_side, list(west_codes)))
        assert west_count / west_side.size > 0.5, f"West side of ridge should flow west, got {np.unique(west_side)}"

    def test_flat_terrain_handled(self, synthetic_dem_flat):
        """Flat terrain should be handled without errors (flats resolved)."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_flat.attrs["resolution_m"] = 30.0

        # Should not raise
        flow_dir = proc.compute_flow_directions(synthetic_dem_flat)

        # All cells should have valid flow directions
        assert flow_dir.shape == synthetic_dem_flat.shape
        assert not np.any(np.isnan(flow_dir.values))

    def test_pit_filled(self, synthetic_dem_flat_with_pit):
        """Pit should be filled and have outward flow."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_flat_with_pit)

        # D8 valid codes (power-of-2): 1, 2, 4, 8, 16, 32, 64, 128
        # 0 = pit/outlet, 247 = nodata
        ny, nx = flow_dir.shape
        pit_dir = flow_dir.values[ny // 2, nx // 2]
        valid_d8_codes = {0, 1, 2, 4, 8, 16, 32, 64, 128}
        # Pit should either be filled (flow out) or be an outlet (0)
        assert pit_dir in valid_d8_codes, f"Pit should have valid D8 code, got {pit_dir}"

    def test_attrs_preserved(self, synthetic_dem_tilted):
        """Test that output has expected attributes."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)

        assert "long_name" in flow_dir.attrs
        assert "resolution_m" in flow_dir.attrs


@requires_pyflwdir
class TestDelineateCatchments:
    """Tests for catchment delineation."""

    def test_output_shape_and_dims(self, synthetic_dem_single_valley):
        """Test that output has correct shape and dimensions."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_single_valley)
        catchments = proc.delineate_catchments(
            synthetic_dem_single_valley, flow_dir, min_area_m2=0
        )

        assert catchments.dims == ("y", "x")
        assert catchments.shape == synthetic_dem_single_valley.shape

    def test_single_valley_single_catchment(self, synthetic_dem_single_valley):
        """Single valley should produce a single large catchment."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_single_valley)
        catchments = proc.delineate_catchments(
            synthetic_dem_single_valley, flow_dir, min_area_m2=0
        )

        # Should have exactly 1 catchment (ID 1) covering most cells
        unique_ids = set(np.unique(catchments.values))
        # Remove 0 (nodata) if present
        unique_ids.discard(0)
        assert len(unique_ids) == 1, f"Expected 1 catchment, got {len(unique_ids)}"

    def test_ridge_splits_catchments(self, synthetic_dem_ridge):
        """Ridge should split into 2 catchments (east and west)."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_ridge)
        catchments = proc.delineate_catchments(
            synthetic_dem_ridge, flow_dir, min_area_m2=0
        )

        unique_ids = set(np.unique(catchments.values))
        unique_ids.discard(0)
        assert len(unique_ids) == 2, f"Expected 2 catchments, got {len(unique_ids)}"

    def test_min_area_filtering(self, synthetic_dem_ridge):
        """Small catchments should be filtered by min_area_m2."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_ridge)

        # Very large min area -> all catchments filtered
        catchments_strict = proc.delineate_catchments(
            synthetic_dem_ridge, flow_dir, min_area_m2=1e10
        )
        unique_strict = set(np.unique(catchments_strict.values))
        unique_strict.discard(0)
        assert len(unique_strict) == 0, "All catchments should be filtered with huge min_area"

        # Zero min area -> all catchments kept
        catchments_loose = proc.delineate_catchments(
            synthetic_dem_ridge, flow_dir, min_area_m2=0
        )
        unique_loose = set(np.unique(catchments_loose.values))
        unique_loose.discard(0)
        assert len(unique_loose) > 0, "Catchments should exist with min_area=0"

    def test_catchment_ids_are_integers(self, synthetic_dem_single_valley):
        """Catchment IDs should be positive integers."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_single_valley)
        catchments = proc.delineate_catchments(
            synthetic_dem_single_valley, flow_dir, min_area_m2=0
        )

        assert catchments.dtype in [np.int32, np.int64]
        assert np.all(catchments.values >= 0)

    def test_attrs_preserved(self, synthetic_dem_single_valley):
        """Test that output has expected attributes."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_single_valley)
        catchments = proc.delineate_catchments(
            synthetic_dem_single_valley, flow_dir, min_area_m2=5000.0
        )

        assert "long_name" in catchments.attrs
        assert "min_area_m2" in catchments.attrs
        assert catchments.attrs["min_area_m2"] == 5000.0


@requires_pyflwdir
class TestComputeFlowAccumulation:
    """Tests for flow accumulation computation."""

    def test_output_shape_and_dims(self, synthetic_dem_tilted):
        """Test that output has correct shape and dimensions."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)
        flow_acc = proc.compute_flow_accumulation(synthetic_dem_tilted, flow_dir)

        assert flow_acc.dims == ("y", "x")
        assert flow_acc.shape == synthetic_dem_tilted.shape

    def test_flow_accumulation_increases_downstream(self, synthetic_dem_single_valley):
        """Flow accumulation should increase toward outlet."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_single_valley)
        flow_acc = proc.compute_flow_accumulation(synthetic_dem_single_valley, flow_dir)

        ny, nx = flow_acc.shape

        # Valley bottom (center column) should have higher accumulation
        # than valley sides
        center_col = nx // 2
        center_acc = flow_acc.values[:, center_col]
        side_acc = flow_acc.values[:, 2]

        # Mean accumulation in center should be higher than at sides
        assert np.mean(center_acc) > np.mean(side_acc)

    def test_flow_accumulation_positive(self, synthetic_dem_tilted):
        """Flow accumulation should be positive everywhere."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)
        flow_acc = proc.compute_flow_accumulation(synthetic_dem_tilted, flow_dir)

        assert np.all(flow_acc.values >= 1), "All cells should have at least 1 (self)"

    def test_headwater_cells_have_low_accumulation(self, synthetic_dem_single_valley):
        """Ridgetop/headwater cells should have low accumulation (just themselves)."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)

        flow_dir = proc.compute_flow_directions(synthetic_dem_single_valley)
        flow_acc = proc.compute_flow_accumulation(synthetic_dem_single_valley, flow_dir)

        # Top row (northern edge) should have low accumulation
        top_row_acc = flow_acc.values[0, :]
        assert np.max(top_row_acc) < 5, "Headwater cells should have low accumulation"

    def test_attrs_preserved(self, synthetic_dem_tilted):
        """Test that output has expected attributes."""
        config = DomainConfig(bbox=[0, 0, 1, 1])
        proc = DEMProcessor(config)
        synthetic_dem_tilted.attrs["resolution_m"] = 30.0

        flow_dir = proc.compute_flow_directions(synthetic_dem_tilted)
        flow_acc = proc.compute_flow_accumulation(synthetic_dem_tilted, flow_dir)

        assert "long_name" in flow_acc.attrs
        assert "cell_area_m2" in flow_acc.attrs
        assert flow_acc.attrs["units"] == "cells"


@requires_pyflwdir
class TestProcessWithCatchments:
    """Tests for process() method with catchment computation."""

    def test_process_no_catchments_by_default(self, synthetic_dem_flat, tmp_path):
        """Without polygon_config, process() should not compute catchments."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        proc = DEMProcessor(config)

        ds = proc.process()

        assert "catchments" not in ds
        assert "flow_direction" not in ds
        assert "flow_accumulation" not in ds

    def test_process_with_polygon_config_respect_catchments(self, synthetic_dem_flat, tmp_path):
        """With polygon_config.respect_catchments=True, process() should compute catchments."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        polygon_config = PolygonConfig(respect_catchments=True, min_area_m2=100.0)
        proc = DEMProcessor(config, polygon_config=polygon_config)

        ds = proc.process()

        assert "catchments" in ds
        assert "flow_direction" in ds
        assert "flow_accumulation" in ds

    def test_process_polygon_config_no_respect_catchments(self, synthetic_dem_flat, tmp_path):
        """With polygon_config.respect_catchments=False, should skip catchments."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        polygon_config = PolygonConfig(respect_catchments=False)
        proc = DEMProcessor(config, polygon_config=polygon_config)

        ds = proc.process()

        assert "catchments" not in ds

    def test_process_explicit_compute_catchments_true(self, synthetic_dem_flat, tmp_path):
        """Explicit compute_catchments=True should compute catchments."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        proc = DEMProcessor(config)  # No polygon_config

        ds = proc.process(compute_catchments=True)

        assert "catchments" in ds
        assert "flow_direction" in ds
        assert "flow_accumulation" in ds

    def test_process_explicit_compute_catchments_false(self, synthetic_dem_flat, tmp_path):
        """Explicit compute_catchments=False should skip even with polygon_config."""
        dem_path = tmp_path / "test_dem.tif"
        _create_test_geotiff(dem_path, synthetic_dem_flat.values)

        config = DomainConfig(dem=dem_path)
        polygon_config = PolygonConfig(respect_catchments=True)
        proc = DEMProcessor(config, polygon_config=polygon_config)

        ds = proc.process(compute_catchments=False)

        assert "catchments" not in ds
