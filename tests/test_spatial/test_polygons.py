"""Tests for HRU polygon generation."""

import numpy as np
import xarray as xr

from topopyscale2.config.schema import PolygonConfig
from topopyscale2.spatial.polygons import HRUPolygonGenerator


def make_dem_dataset(
    ny: int = 20,
    nx: int = 20,
    res: float = 30.0,
    elev_min: float = 2000.0,
    elev_range: float = 500.0,
    pattern: str = "gradient",
) -> xr.Dataset:
    """Create a synthetic DEM dataset for testing.

    Args:
        ny: Number of rows.
        nx: Number of columns.
        res: Pixel resolution in meters.
        elev_min: Minimum elevation.
        elev_range: Range of elevation.
        pattern: Elevation pattern - 'flat', 'gradient', 'ridge', 'valley'.

    Returns:
        xr.Dataset with elevation, slope, aspect, svf variables.
    """
    x = np.arange(nx) * res
    y = np.arange(ny) * res
    yy, xx = np.meshgrid(y, x, indexing="ij")

    if pattern == "flat":
        elevation = np.full((ny, nx), elev_min)
    elif pattern == "gradient":
        # Elevation increases with x (east) and y (north)
        elevation = elev_min + (xx / (nx * res)) * elev_range * 0.5 + (yy / (ny * res)) * elev_range * 0.5
    elif pattern == "ridge":
        # Ridge running north-south in center
        center_x = x[nx // 2]
        elevation = elev_min + elev_range - np.abs(xx - center_x) / (nx * res / 2) * elev_range
    elif pattern == "valley":
        # Valley running north-south in center
        center_x = x[nx // 2]
        elevation = elev_min + np.abs(xx - center_x) / (nx * res / 2) * elev_range
    else:
        raise ValueError(f"Unknown pattern: {pattern}")

    # Compute slope and aspect (handle small arrays)
    if ny >= 2 and nx >= 2:
        dy, dx = np.gradient(elevation, res)
        slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
        slope_deg = np.degrees(slope_rad)

        aspect_rad = np.arctan2(-dx, -dy)
        aspect_deg = np.degrees(aspect_rad)
        aspect_deg = (aspect_deg + 360.0) % 360.0
    else:
        # For single-pixel DEM, use default values
        slope_deg = np.zeros((ny, nx))
        aspect_deg = np.zeros((ny, nx))

    # SVF (simplified - just use 1.0 for flat, decrease with slope)
    svf = 1.0 - slope_deg / 180.0

    ds = xr.Dataset(
        {
            "elevation": (["y", "x"], elevation),
            "slope": (["y", "x"], slope_deg),
            "aspect": (["y", "x"], aspect_deg),
            "svf": (["y", "x"], svf),
        },
        coords={"y": y, "x": x},
    )
    ds["elevation"].attrs["resolution_m"] = res
    return ds


def make_catchments(ny: int, nx: int, n_catchments: int = 2) -> xr.DataArray:
    """Create a synthetic catchment raster.

    Args:
        ny: Number of rows.
        nx: Number of columns.
        n_catchments: Number of catchments (split horizontally).

    Returns:
        xr.DataArray with catchment IDs.
    """
    catchments = np.zeros((ny, nx), dtype=np.int32)
    col_per_catchment = nx // n_catchments

    for i in range(n_catchments):
        start_col = i * col_per_catchment
        end_col = (i + 1) * col_per_catchment if i < n_catchments - 1 else nx
        catchments[:, start_col:end_col] = i + 1

    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    return xr.DataArray(catchments, dims=["y", "x"], coords={"y": y, "x": x})


# =============================================================================
# Test Elevation Band Binning
# =============================================================================


class TestElevationBinning:
    """Tests for _bin_elevation method."""

    def test_flat_dem_single_band(self):
        """Flat DEM should produce a single elevation band."""
        ds = make_dem_dataset(pattern="flat", elev_min=3000.0)
        config = PolygonConfig(elevation_bands=200.0)
        gen = HRUPolygonGenerator(config)

        bands = gen._bin_elevation(ds["elevation"], 200.0)

        # All values should be the same band
        unique_bands = np.unique(bands[bands >= 0])
        assert len(unique_bands) == 1

    def test_gradient_dem_multiple_bands(self):
        """Gradient DEM should produce multiple elevation bands."""
        ds = make_dem_dataset(pattern="gradient", elev_min=2000.0, elev_range=600.0)
        config = PolygonConfig(elevation_bands=200.0)
        gen = HRUPolygonGenerator(config)

        bands = gen._bin_elevation(ds["elevation"], 200.0)

        # Should have at least 3 bands (600m range / 200m bands)
        unique_bands = np.unique(bands[bands >= 0])
        assert len(unique_bands) >= 3

    def test_band_boundaries(self):
        """Test that band boundaries are correct."""
        # Create DEM with known elevations
        ny, nx = 10, 10
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        elevation = np.array([
            [1000, 1199, 1200, 1201, 1399, 1400, 1401, 1599, 1600, 1601]
        ] * ny, dtype=np.float64)

        dem = xr.DataArray(elevation, dims=["y", "x"], coords={"y": y, "x": x})
        config = PolygonConfig(elevation_bands=200.0)
        gen = HRUPolygonGenerator(config)

        bands = gen._bin_elevation(dem, 200.0)

        # Check band assignments
        assert bands[0, 0] == bands[0, 1]  # 1000 and 1199 in same band
        assert bands[0, 1] != bands[0, 2]  # 1199 and 1200 in different bands
        assert bands[0, 2] == bands[0, 3]  # 1200 and 1201 in same band

    def test_nan_handling(self):
        """NaN values should be assigned to band -1."""
        ds = make_dem_dataset(pattern="gradient")
        ds["elevation"].values[5, 5] = np.nan

        config = PolygonConfig(elevation_bands=200.0)
        gen = HRUPolygonGenerator(config)

        bands = gen._bin_elevation(ds["elevation"], 200.0)

        assert bands[5, 5] == -1

    def test_small_band_size(self):
        """Small band size should produce many bands."""
        ds = make_dem_dataset(pattern="gradient", elev_range=500.0)
        config = PolygonConfig(elevation_bands=50.0)
        gen = HRUPolygonGenerator(config)

        bands = gen._bin_elevation(ds["elevation"], 50.0)

        # 500m range / 50m bands = at least 10 bands
        unique_bands = np.unique(bands[bands >= 0])
        assert len(unique_bands) >= 8

    def test_large_band_size(self):
        """Large band size should produce few bands."""
        ds = make_dem_dataset(pattern="gradient", elev_range=500.0)
        config = PolygonConfig(elevation_bands=1000.0)
        gen = HRUPolygonGenerator(config)

        bands = gen._bin_elevation(ds["elevation"], 1000.0)

        # 500m range / 1000m bands = 1 band
        unique_bands = np.unique(bands[bands >= 0])
        assert len(unique_bands) == 1


# =============================================================================
# Test Aspect Classification
# =============================================================================


class TestAspectClassification:
    """Tests for _classify_aspect method."""

    def test_four_classes_cardinal(self):
        """4 classes should produce N, E, S, W."""
        config = PolygonConfig(aspect_classes=4)
        gen = HRUPolygonGenerator(config)

        # Create aspect array with cardinal directions
        ny, nx = 4, 4
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        aspect = np.array([
            [0, 90, 180, 270],    # N, E, S, W
            [45, 135, 225, 315],  # NE->N or E, SE->E or S, SW->S or W, NW->W or N
            [1, 89, 181, 269],
            [359, 91, 179, 271],
        ], dtype=np.float64)

        asp_da = xr.DataArray(aspect, dims=["y", "x"], coords={"y": y, "x": x})
        classes = gen._classify_aspect(asp_da, 4)

        # N=0, E=1, S=2, W=3 (class width = 90 degrees)
        assert classes[0, 0] == 0  # 0 deg -> N
        assert classes[0, 1] == 1  # 90 deg -> E
        assert classes[0, 2] == 2  # 180 deg -> S
        assert classes[0, 3] == 3  # 270 deg -> W

    def test_eight_classes(self):
        """8 classes should produce N, NE, E, SE, S, SW, W, NW."""
        config = PolygonConfig(aspect_classes=8)
        gen = HRUPolygonGenerator(config)

        # Create aspect array with 8 directions
        ny, nx = 2, 8
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        aspect = np.array([
            [0, 45, 90, 135, 180, 225, 270, 315],
            [0, 45, 90, 135, 180, 225, 270, 315],
        ], dtype=np.float64)

        asp_da = xr.DataArray(aspect, dims=["y", "x"], coords={"y": y, "x": x})
        classes = gen._classify_aspect(asp_da, 8)

        # Check first row covers all 8 classes
        unique_classes = np.unique(classes[0, :])
        assert len(unique_classes) == 8

    def test_aspect_class_names(self):
        """Test human-readable aspect class names."""
        config4 = PolygonConfig(aspect_classes=4)
        gen4 = HRUPolygonGenerator(config4)

        assert gen4.get_aspect_class_name(0) == "N"
        assert gen4.get_aspect_class_name(1) == "E"
        assert gen4.get_aspect_class_name(2) == "S"
        assert gen4.get_aspect_class_name(3) == "W"

        config8 = PolygonConfig(aspect_classes=8)
        gen8 = HRUPolygonGenerator(config8)

        assert gen8.get_aspect_class_name(0) == "N"
        assert gen8.get_aspect_class_name(1) == "NE"
        assert gen8.get_aspect_class_name(4) == "S"

    def test_nan_handling(self):
        """NaN aspect values should be assigned to class -1."""
        ny, nx = 5, 5
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        aspect = np.full((ny, nx), 90.0)
        aspect[2, 2] = np.nan

        asp_da = xr.DataArray(aspect, dims=["y", "x"], coords={"y": y, "x": x})
        config = PolygonConfig(aspect_classes=4)
        gen = HRUPolygonGenerator(config)

        classes = gen._classify_aspect(asp_da, 4)

        assert classes[2, 2] == -1


# =============================================================================
# Test Polygon Generation
# =============================================================================


class TestPolygonGeneration:
    """Tests for the main generate() method."""

    def test_basic_generation(self):
        """Basic polygon generation should work."""
        ds = make_dem_dataset(pattern="gradient", elev_range=400.0)
        config = PolygonConfig(elevation_bands=200.0, aspect_classes=4, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        assert len(units) > 0
        for u in units:
            assert u.id.startswith("hru_")
            assert u.area_m2 > 0
            assert "elevation" in u.attributes

    def test_unit_attributes(self):
        """Generated units should have correct attributes."""
        ds = make_dem_dataset(pattern="gradient")
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        for u in units:
            assert "elevation" in u.attributes
            assert "slope" in u.attributes
            assert "aspect" in u.attributes
            assert "svf" in u.attributes

    def test_centroid_within_domain(self):
        """Unit centroids should be within the domain."""
        ds = make_dem_dataset(nx=30, ny=30, res=30.0)
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        x_min, x_max = float(ds.x.min()), float(ds.x.max())
        y_min, y_max = float(ds.y.min()), float(ds.y.max())

        for u in units:
            assert x_min <= u.x <= x_max, f"x={u.x} outside [{x_min}, {x_max}]"
            assert y_min <= u.y <= y_max, f"y={u.y} outside [{y_min}, {y_max}]"

    def test_elevation_within_range(self):
        """Unit elevations should be within the DEM range."""
        ds = make_dem_dataset(pattern="gradient", elev_min=2000.0, elev_range=500.0)
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        elev_min = float(ds["elevation"].min())
        elev_max = float(ds["elevation"].max())

        for u in units:
            assert elev_min <= u.elevation <= elev_max, (
                f"elevation={u.elevation} outside [{elev_min}, {elev_max}]"
            )

    def test_total_area_conserved(self):
        """Total area of all units should equal the DEM area (minus filtered)."""
        ds = make_dem_dataset(nx=20, ny=20, res=30.0)
        config = PolygonConfig(elevation_bands=500.0, min_area_m2=1.0)  # Large bands to get fewer units
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        total_unit_area = sum(u.area_m2 for u in units)
        expected_area = 20 * 20 * 30.0 * 30.0

        # Should be approximately equal (may differ slightly due to nodata)
        assert abs(total_unit_area - expected_area) / expected_area < 0.01

    def test_geometry_valid(self):
        """Generated geometries should be valid."""
        ds = make_dem_dataset(pattern="gradient")
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        for u in units:
            if u.geometry is not None:
                assert u.geometry.is_valid, f"Invalid geometry for {u.id}"


# =============================================================================
# Test Catchment Integration
# =============================================================================


class TestCatchmentIntegration:
    """Tests for catchment boundary respecting."""

    def test_with_catchments(self):
        """Polygons should be split by catchment boundaries."""
        ds = make_dem_dataset(nx=20, ny=20, pattern="flat")
        catchments = make_catchments(20, 20, n_catchments=2)

        config = PolygonConfig(
            elevation_bands=500.0,  # Large to get single band
            aspect_classes=4,
            respect_catchments=True,
            min_area_m2=1.0,
        )
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds, catchments=catchments)

        # Flat DEM with 2 catchments should produce at least 2 units
        # (one per catchment, possibly more due to aspect variation)
        assert len(units) >= 2

    def test_without_catchments(self):
        """Without catchments, should produce fewer units."""
        ds = make_dem_dataset(nx=20, ny=20, pattern="flat")

        config = PolygonConfig(
            elevation_bands=500.0,
            aspect_classes=4,
            respect_catchments=True,  # Still True, but no catchments provided
            min_area_m2=1.0,
        )
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds, catchments=None)

        # Flat DEM without catchments should produce fewer units
        assert len(units) >= 1

    def test_respect_catchments_false(self):
        """respect_catchments=False should ignore catchment boundaries."""
        ds = make_dem_dataset(nx=20, ny=20, pattern="flat")
        catchments = make_catchments(20, 20, n_catchments=4)

        config_respect = PolygonConfig(
            elevation_bands=1000.0,
            aspect_classes=4,
            respect_catchments=True,
            min_area_m2=1.0,
        )
        config_ignore = PolygonConfig(
            elevation_bands=1000.0,
            aspect_classes=4,
            respect_catchments=False,
            min_area_m2=1.0,
        )

        gen_respect = HRUPolygonGenerator(config_respect)
        gen_ignore = HRUPolygonGenerator(config_ignore)

        units_respect = gen_respect.generate(ds, catchments=catchments)
        units_ignore = gen_ignore.generate(ds, catchments=catchments)

        # Ignoring catchments should produce fewer units
        assert len(units_ignore) <= len(units_respect)


# =============================================================================
# Test Neighbor Identification
# =============================================================================


class TestNeighborIdentification:
    """Tests for _compute_adjacency method."""

    def test_simple_adjacency(self):
        """Test adjacency computation with known pattern."""
        # Create a simple 3x3 label grid
        labels = np.array([
            [0, 0, 1],
            [0, 1, 1],
            [2, 2, 1],
        ], dtype=np.int32)

        config = PolygonConfig()
        gen = HRUPolygonGenerator(config)

        adjacency = gen._compute_adjacency(labels)

        # 0 neighbors 1 and 2
        assert 1 in adjacency[0]
        assert 2 in adjacency[0]

        # 1 neighbors 0 and 2
        assert 0 in adjacency[1]
        assert 2 in adjacency[1]

        # 2 neighbors 0 and 1
        assert 0 in adjacency[2]
        assert 1 in adjacency[2]

    def test_no_self_neighbors(self):
        """A polygon should not be its own neighbor."""
        labels = np.array([
            [0, 0, 1],
            [0, 0, 1],
            [0, 1, 1],
        ], dtype=np.int32)

        config = PolygonConfig()
        gen = HRUPolygonGenerator(config)

        adjacency = gen._compute_adjacency(labels)

        for label, neighbors in adjacency.items():
            assert label not in neighbors

    def test_nodata_excluded(self):
        """Nodata (-1) should not appear in adjacency."""
        labels = np.array([
            [0, 0, -1],
            [0, 1, -1],
            [1, 1, -1],
        ], dtype=np.int32)

        config = PolygonConfig()
        gen = HRUPolygonGenerator(config)

        adjacency = gen._compute_adjacency(labels)

        # -1 should not be in adjacency
        assert -1 not in adjacency
        for neighbors in adjacency.values():
            assert -1 not in neighbors

    def test_neighbors_in_generated_units(self):
        """Generated units should have valid neighbor references."""
        ds = make_dem_dataset(pattern="gradient", elev_range=600.0)
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)
        unit_ids = {u.id for u in units}

        for u in units:
            if u.neighbors:
                for n in u.neighbors:
                    assert n in unit_ids, f"Neighbor {n} not in unit set"


# =============================================================================
# Test Min Area Filtering
# =============================================================================


class TestMinAreaFiltering:
    """Tests for minimum area filtering."""

    def test_min_area_small(self):
        """Very small min_area should keep most polygons."""
        ds = make_dem_dataset(pattern="gradient")
        config = PolygonConfig(min_area_m2=1.0)  # Smallest allowed value
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        # Should have polygons
        assert len(units) > 0

    def test_min_area_large(self):
        """Large min_area should filter small polygons."""
        ds = make_dem_dataset(nx=10, ny=10, res=30.0, pattern="gradient", elev_range=400.0)

        # Total area = 10*10*30*30 = 90000 m^2
        # With min_area = 50000, should get fewer units
        config_small = PolygonConfig(elevation_bands=100.0, min_area_m2=1.0)
        config_large = PolygonConfig(elevation_bands=100.0, min_area_m2=50000.0)

        gen_small = HRUPolygonGenerator(config_small)
        gen_large = HRUPolygonGenerator(config_large)

        units_small = gen_small.generate(ds)
        units_large = gen_large.generate(ds)

        assert len(units_large) < len(units_small)

    def test_min_area_filters_correctly(self):
        """All returned units should meet min_area requirement."""
        ds = make_dem_dataset(pattern="gradient")
        min_area = 5000.0
        config = PolygonConfig(min_area_m2=min_area)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        for u in units:
            assert u.area_m2 >= min_area, f"Unit {u.id} has area {u.area_m2} < {min_area}"


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_band_dem(self):
        """DEM within single elevation band should produce units."""
        ds = make_dem_dataset(pattern="flat", elev_min=3000.0)
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        # Should still have units (based on aspect classes)
        assert len(units) >= 1

    def test_uniform_aspect(self):
        """Uniform aspect should produce fewer units."""
        # Create DEM with uniform aspect (tilted plane)
        ny, nx = 20, 20
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0
        yy, _ = np.meshgrid(y, x, indexing="ij")
        elevation = 2000.0 + yy * 0.1  # North-facing slope

        ds = xr.Dataset(
            {
                "elevation": (["y", "x"], elevation),
                "slope": (["y", "x"], np.full((ny, nx), 5.71)),  # atan(0.1) in degrees
                "aspect": (["y", "x"], np.full((ny, nx), 180.0)),  # South-facing
                "svf": (["y", "x"], np.ones((ny, nx)) * 0.99),
            },
            coords={"y": y, "x": x},
        )
        ds["elevation"].attrs["resolution_m"] = 30.0

        config = PolygonConfig(elevation_bands=200.0, aspect_classes=4, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        # With uniform aspect, number of units ~ number of elevation bands
        elev_range = float(ds["elevation"].max() - ds["elevation"].min())
        expected_bands = max(1, int(np.ceil(elev_range / 200.0)))

        # Should have approximately expected_bands units (within 2x)
        assert len(units) <= expected_bands * 2

    def test_small_dem(self):
        """Small DEM (e.g., 3x3) should work."""
        ds = make_dem_dataset(nx=3, ny=3, pattern="gradient")
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        assert len(units) >= 1

    def test_single_pixel_dem(self):
        """Single pixel DEM should produce one unit."""
        ds = make_dem_dataset(nx=1, ny=1, pattern="flat")
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        assert len(units) == 1

    def test_membership_raster(self):
        """get_membership should return valid raster after generate()."""
        ds = make_dem_dataset(pattern="gradient")
        config = PolygonConfig(elevation_bands=200.0, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        # Before generate, membership should be None
        assert gen.get_membership() is None

        units = gen.generate(ds)

        membership = gen.get_membership()
        assert membership is not None
        assert membership.shape == ds["elevation"].shape


# =============================================================================
# Test Polygon Statistics
# =============================================================================


class TestPolygonStats:
    """Tests for _polygon_stats method."""

    def test_elevation_mean(self):
        """Mean elevation should be correctly computed."""
        ds = make_dem_dataset(pattern="gradient")
        config = PolygonConfig()
        gen = HRUPolygonGenerator(config)

        # Create a simple mask
        mask = np.zeros(ds["elevation"].shape, dtype=bool)
        mask[5:10, 5:10] = True

        stats = gen._polygon_stats(ds, mask)

        expected_elev = float(ds["elevation"].values[mask].mean())
        assert abs(stats["elevation"] - expected_elev) < 0.01

    def test_aspect_circular_mean(self):
        """Aspect should use circular mean."""
        ny, nx = 10, 10
        x = np.arange(nx) * 30.0
        y = np.arange(ny) * 30.0

        # Create aspect with values spanning 0/360 boundary
        aspect = np.full((ny, nx), 350.0)
        aspect[5:, :] = 10.0  # Mean should be ~0 (North)

        ds = xr.Dataset(
            {
                "elevation": (["y", "x"], np.full((ny, nx), 3000.0)),
                "slope": (["y", "x"], np.zeros((ny, nx))),
                "aspect": (["y", "x"], aspect),
                "svf": (["y", "x"], np.ones((ny, nx))),
            },
            coords={"y": y, "x": x},
        )

        config = PolygonConfig()
        gen = HRUPolygonGenerator(config)

        mask = np.ones((ny, nx), dtype=bool)
        stats = gen._polygon_stats(ds, mask)

        # Circular mean of 350 and 10 should be close to 0 (North)
        # Actually mean of half 350, half 10 should be close to 0
        assert stats["aspect"] < 20.0 or stats["aspect"] > 340.0


# =============================================================================
# Test Known Geometries
# =============================================================================


class TestKnownGeometries:
    """Tests with known geometric patterns."""

    def test_ridge_produces_two_aspect_groups(self):
        """Ridge should produce distinct east and west facing units."""
        ds = make_dem_dataset(nx=30, ny=30, pattern="ridge", elev_range=300.0)

        config = PolygonConfig(
            elevation_bands=500.0,  # Large to minimize band splitting
            aspect_classes=4,
            min_area_m2=1.0,
        )
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        # Get aspect classes
        aspects = [u.attributes.get("aspect", 0) for u in units]

        # Should have both east-facing (~90) and west-facing (~270) units
        has_east = any(45 < a < 135 for a in aspects)
        has_west = any(225 < a < 315 for a in aspects)

        assert has_east and has_west, f"Missing east or west aspects: {aspects}"

    def test_valley_produces_two_aspect_groups(self):
        """Valley should produce distinct east and west facing units."""
        ds = make_dem_dataset(nx=30, ny=30, pattern="valley", elev_range=300.0)

        config = PolygonConfig(
            elevation_bands=500.0,
            aspect_classes=4,
            min_area_m2=1.0,
        )
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        aspects = [u.attributes.get("aspect", 0) for u in units]

        # Should have both east-facing and west-facing units
        has_east = any(45 < a < 135 for a in aspects)
        has_west = any(225 < a < 315 for a in aspects)

        assert has_east and has_west, f"Missing east or west aspects: {aspects}"

    def test_elevation_bands_match_dem_range(self):
        """Elevation bands should cover the DEM's elevation range."""
        elev_min = 1500.0
        elev_range = 800.0
        band_size = 200.0

        ds = make_dem_dataset(pattern="gradient", elev_min=elev_min, elev_range=elev_range)

        config = PolygonConfig(elevation_bands=band_size, min_area_m2=1.0)
        gen = HRUPolygonGenerator(config)

        units = gen.generate(ds)

        # Get unique elevation bands (rounded to band size)
        elevations = [u.attributes["elevation"] for u in units]
        min_unit_elev = min(elevations)
        max_unit_elev = max(elevations)

        # Unit elevations should span most of the DEM range
        actual_range = max_unit_elev - min_unit_elev
        expected_range = elev_range * 0.8  # Allow some margin

        assert actual_range >= expected_range * 0.5, (
            f"Elevation range {actual_range} too small (expected ~{expected_range})"
        )
