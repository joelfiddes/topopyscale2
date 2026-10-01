"""Tests for RasterMapper - cluster to raster mapping."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.outputs.raster_mapper import RasterMapper
from topopyscale2.spatial.units import SpatialUnit


def make_membership(ny=10, nx=10, n_clusters=3):
    """Create test membership array with n_clusters."""
    # Create simple pattern: divide grid into n_clusters vertical stripes
    membership = np.zeros((ny, nx), dtype=np.int32)
    stripe_width = nx // n_clusters
    for i in range(n_clusters):
        start = i * stripe_width
        end = (i + 1) * stripe_width if i < n_clusters - 1 else nx
        membership[:, start:end] = i

    # Add some nodata (-1) in corners
    membership[0, 0] = -1
    membership[-1, -1] = -1

    y_coords = np.linspace(46.0, 46.9, ny)
    x_coords = np.linspace(7.0, 7.9, nx)

    return xr.DataArray(
        membership,
        dims=["y", "x"],
        coords={"y": y_coords, "x": x_coords},
    )


def make_units(n=3):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(7.0 + i * 0.3, 46.5, 2000.0 + i * 100),
            surface_type="open",
            area_m2=900.0,
        )
        for i in range(n)
    ]


def make_forcing_dataset(nt=24, n_units=3):
    """Create test forcing dataset."""
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    unit_ids = [f"unit_{i:04d}" for i in range(n_units)]

    # Create data with distinct values per unit for easy verification
    temp_data = np.zeros((nt, n_units))
    precip_data = np.zeros((nt, n_units))
    for i in range(n_units):
        temp_data[:, i] = 270.0 + i * 5.0  # 270, 275, 280 K
        precip_data[:, i] = 0.001 * (i + 1)  # 0.001, 0.002, 0.003 m

    return xr.Dataset(
        {
            "temperature": (["time", "unit_id"], temp_data),
            "precipitation": (["time", "unit_id"], precip_data),
        },
        coords={"time": times, "unit_id": unit_ids},
    )


class TestRasterMapper:
    def test_init(self):
        """Test RasterMapper initialization."""
        membership = make_membership()
        units = make_units()

        mapper = RasterMapper(membership, units)

        assert mapper.membership is membership
        assert len(mapper.units) == 3
        assert mapper.crs == "EPSG:4326"

    def test_init_custom_crs(self):
        """Test RasterMapper with custom CRS."""
        membership = make_membership()
        units = make_units()

        mapper = RasterMapper(membership, units, crs="EPSG:32632")

        assert mapper.crs == "EPSG:32632"

    def test_map_to_raster_shape(self):
        """Test that map_to_raster produces correct shape."""
        membership = make_membership(ny=10, nx=15, n_clusters=3)
        units = make_units(3)
        forcing = make_forcing_dataset(nt=24, n_units=3)

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        assert result.dims == {"time": 24, "y": 10, "x": 15}
        assert "temperature" in result
        assert "precipitation" in result

    def test_map_to_raster_values(self):
        """Test that values are correctly mapped from units to pixels."""
        membership = make_membership(ny=10, nx=10, n_clusters=3)
        units = make_units(3)
        forcing = make_forcing_dataset(nt=24, n_units=3)

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        # Check that pixels in each cluster have the correct temperature
        # Cluster 0: columns 0-2, temp = 270
        # Cluster 1: columns 3-5, temp = 275
        # Cluster 2: columns 6-9, temp = 280
        temp = result["temperature"].isel(time=0).values

        # Cluster 0 pixels (excluding nodata at [0,0])
        assert temp[1, 0] == 270.0
        assert temp[1, 2] == 270.0

        # Cluster 1 pixels
        assert temp[0, 3] == 275.0
        assert temp[5, 5] == 275.0

        # Cluster 2 pixels (excluding nodata at [-1,-1])
        assert temp[0, 6] == 280.0
        assert temp[5, 8] == 280.0

    def test_map_to_raster_nan_for_nodata(self):
        """Test that nodata pixels (membership=-1) become NaN."""
        membership = make_membership(ny=10, nx=10, n_clusters=3)
        units = make_units(3)
        forcing = make_forcing_dataset(nt=24, n_units=3)

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        temp = result["temperature"].isel(time=0).values

        # Nodata pixels should be NaN
        assert np.isnan(temp[0, 0])
        assert np.isnan(temp[-1, -1])

    def test_map_to_raster_preserves_time(self):
        """Test that time coordinate is preserved."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset(nt=48)

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        np.testing.assert_array_equal(result.time.values, forcing.time.values)

    def test_map_to_raster_preserves_coords(self):
        """Test that y and x coordinates are preserved from membership."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        np.testing.assert_array_equal(result.y.values, membership.y.values)
        np.testing.assert_array_equal(result.x.values, membership.x.values)

    def test_map_to_raster_selected_variables(self):
        """Test mapping only selected variables."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing, variables=["temperature"])

        assert "temperature" in result
        assert "precipitation" not in result

    def test_map_to_raster_copies_attrs(self):
        """Test that variable attributes are copied."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()
        forcing["temperature"].attrs["units"] = "K"
        forcing["temperature"].attrs["standard_name"] = "air_temperature"

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        assert result["temperature"].attrs["units"] == "K"
        assert result["temperature"].attrs["standard_name"] == "air_temperature"

    def test_round_trip_at_centroids(self):
        """Test that sampling raster at cluster centroids matches original values."""
        ny, nx, n_clusters = 20, 20, 5
        membership = make_membership(ny=ny, nx=nx, n_clusters=n_clusters)
        units = make_units(n_clusters)
        forcing = make_forcing_dataset(nt=12, n_units=n_clusters)

        mapper = RasterMapper(membership, units)
        raster = mapper.map_to_raster(forcing)

        # For each cluster, find a pixel in that cluster and verify value
        temp_raster = raster["temperature"].isel(time=0).values
        temp_forcing = forcing["temperature"].isel(time=0).values

        for cluster_id in range(n_clusters):
            # Find pixels belonging to this cluster
            mask = membership.values == cluster_id
            if np.any(mask):
                # Get value from raster at a pixel in this cluster
                y_idx, x_idx = np.where(mask)
                raster_val = temp_raster[y_idx[0], x_idx[0]]

                # Should match forcing value for this cluster
                forcing_val = temp_forcing[cluster_id]
                assert raster_val == forcing_val

    def test_write_raster_dataset_zarr(self, tmp_path):
        """Test writing raster dataset to zarr."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        output_path = tmp_path / "raster.zarr"
        mapper.write_raster_dataset(forcing, output_path, format="zarr")

        assert output_path.exists()

        # Read back and verify
        result = xr.open_zarr(str(output_path))
        assert result.dims == {"time": 24, "y": 10, "x": 10}
        result.close()

    def test_write_raster_dataset_netcdf(self, tmp_path):
        """Test writing raster dataset to NetCDF."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        output_path = tmp_path / "raster.nc"
        mapper.write_raster_dataset(forcing, output_path, format="netcdf")

        assert output_path.exists()

        # Read back and verify
        result = xr.open_dataset(output_path)
        assert result.dims == {"time": 24, "y": 10, "x": 10}
        result.close()

    def test_write_raster_dataset_invalid_format(self, tmp_path):
        """Test that invalid format raises error."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        output_path = tmp_path / "raster.foo"

        with pytest.raises(ValueError, match="Unknown format"):
            mapper.write_raster_dataset(forcing, output_path, format="foo")


# Check if rasterio is available for GeoTIFF tests
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


@pytest.mark.skipif(not HAS_RASTERIO, reason="rasterio not installed")
class TestRasterMapperGeoTIFF:
    def test_write_geotiff_2d(self, tmp_path):
        """Test writing 2D data to GeoTIFF."""
        membership = make_membership(ny=10, nx=10)
        units = make_units()
        forcing = make_forcing_dataset(nt=1)

        mapper = RasterMapper(membership, units)
        raster = mapper.map_to_raster(forcing)

        output_path = tmp_path / "temp.tif"
        mapper.write_geotiff(
            raster["temperature"].isel(time=0),
            output_path,
            variable="temperature",
        )

        assert output_path.exists()

        # Read back and verify
        with rasterio.open(output_path) as src:
            data = src.read(1)
            assert data.shape == (10, 10)
            assert src.crs.to_string() == "EPSG:4326"

    def test_write_geotiff_3d_with_time_idx(self, tmp_path):
        """Test writing 3D data to GeoTIFF with time index."""
        membership = make_membership(ny=10, nx=10)
        units = make_units()
        forcing = make_forcing_dataset(nt=24)

        mapper = RasterMapper(membership, units)
        raster = mapper.map_to_raster(forcing)

        output_path = tmp_path / "temp.tif"
        mapper.write_geotiff(
            raster["temperature"],
            output_path,
            variable="temperature",
            time_idx=5,
        )

        assert output_path.exists()

    def test_write_geotiff_3d_without_time_idx_fails(self, tmp_path):
        """Test that 3D data without time_idx raises error."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        raster = mapper.map_to_raster(forcing)

        output_path = tmp_path / "temp.tif"
        with pytest.raises(ValueError, match="time_idx required"):
            mapper.write_geotiff(raster["temperature"], output_path)

    def test_write_geotiff_series(self, tmp_path):
        """Test writing GeoTIFF series."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset(nt=6)

        mapper = RasterMapper(membership, units)
        output_dir = tmp_path / "geotiffs"
        paths = mapper.write_geotiff_series(forcing, output_dir, "temperature")

        assert len(paths) == 6
        assert output_dir.exists()
        for p in paths:
            assert p.exists()
            assert p.suffix == ".tif"

    def test_write_geotiff_series_selected_times(self, tmp_path):
        """Test writing GeoTIFF series for selected times."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset(nt=24)

        mapper = RasterMapper(membership, units)
        output_dir = tmp_path / "geotiffs"

        # Select first 3 times
        times = forcing.time.values[:3].tolist()
        paths = mapper.write_geotiff_series(
            forcing, output_dir, "temperature", times=times
        )

        assert len(paths) == 3

    def test_geotiff_crs_preserved(self, tmp_path):
        """Test that custom CRS is preserved in GeoTIFF."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset(nt=1)

        mapper = RasterMapper(membership, units, crs="EPSG:32632")
        raster = mapper.map_to_raster(forcing)

        output_path = tmp_path / "temp.tif"
        mapper.write_geotiff(
            raster["temperature"].isel(time=0),
            output_path,
        )

        with rasterio.open(output_path) as src:
            assert src.crs.to_string() == "EPSG:32632"


class TestRasterMapperEdgeCases:
    def test_single_cluster(self):
        """Test mapping with a single cluster."""
        ny, nx = 5, 5
        membership = xr.DataArray(
            np.zeros((ny, nx), dtype=np.int32),
            dims=["y", "x"],
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )
        units = [SpatialUnit(id="unit_0000", centroid=(0, 0, 100))]
        forcing = make_forcing_dataset(nt=3, n_units=1)

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        # All pixels should have same value
        assert np.all(result["temperature"].isel(time=0).values == 270.0)

    def test_all_nodata(self):
        """Test mapping with all nodata membership."""
        ny, nx = 5, 5
        membership = xr.DataArray(
            np.full((ny, nx), -1, dtype=np.int32),
            dims=["y", "x"],
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )
        units = make_units(3)
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        # All pixels should be NaN
        assert np.all(np.isnan(result["temperature"].values))

    def test_sparse_clusters(self):
        """Test mapping with sparse cluster IDs (not consecutive)."""
        ny, nx = 6, 6
        # Clusters 0, 2, 5 (skipping 1, 3, 4)
        membership_vals = np.zeros((ny, nx), dtype=np.int32)
        membership_vals[:, :2] = 0
        membership_vals[:, 2:4] = 2
        membership_vals[:, 4:] = 5

        membership = xr.DataArray(
            membership_vals,
            dims=["y", "x"],
            coords={"y": np.arange(ny), "x": np.arange(nx)},
        )

        # Create 6 units (including unused ones)
        units = [
            SpatialUnit(id=f"unit_{i:04d}", centroid=(0, 0, 100 + i * 10))
            for i in range(6)
        ]

        # Create forcing with 6 units
        times = pd.date_range("2020-01-01", periods=3, freq="h").values
        unit_ids = [f"unit_{i:04d}" for i in range(6)]
        temp_data = np.array([[270 + i * 2 for i in range(6)]] * 3)
        forcing = xr.Dataset(
            {"temperature": (["time", "unit_id"], temp_data)},
            coords={"time": times, "unit_id": unit_ids},
        )

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing)

        temp = result["temperature"].isel(time=0).values

        # Cluster 0 -> temp 270
        assert temp[0, 0] == 270.0

        # Cluster 2 -> temp 274
        assert temp[0, 2] == 274.0

        # Cluster 5 -> temp 280
        assert temp[0, 5] == 280.0

    def test_empty_forcing_variables(self):
        """Test mapping with no variables."""
        membership = make_membership()
        units = make_units()
        forcing = make_forcing_dataset()

        mapper = RasterMapper(membership, units)
        result = mapper.map_to_raster(forcing, variables=[])

        # Should return empty dataset with coords
        assert len(result.data_vars) == 0
        assert "y" in result.coords
        assert "x" in result.coords
