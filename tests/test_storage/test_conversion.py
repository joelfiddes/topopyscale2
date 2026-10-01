"""Tests for format conversion utilities."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")

from topopyscale2.storage.chunking import (
    chunks_for_nwp,
    compute_optimal_chunks,
    resolve_chunks,
)
from topopyscale2.storage.conversion import (
    get_netcdf_info,
    get_zarr_info,
    netcdf_to_zarr,
    zarr_to_netcdf,
)


def make_test_dataset(nt=24, nlat=10, nlon=20):
    """Create test dataset."""
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    lat = np.linspace(45, 47, nlat)
    lon = np.linspace(10, 12, nlon)

    rng = np.random.default_rng(42)

    return xr.Dataset(
        {
            "temperature": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((nt, nlat, nlon)) + 270,
            ),
            "precipitation": (
                ["time", "latitude", "longitude"],
                np.abs(rng.standard_normal((nt, nlat, nlon))) * 0.001,
            ),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
        attrs={"source": "test"},
    )


class TestNetCDFToZarr:
    """NetCDF to Zarr conversion tests."""

    def test_basic_conversion(self, tmp_path):
        """Test basic NetCDF to Zarr conversion."""
        ds = make_test_dataset()
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)

        zarr_path = netcdf_to_zarr(nc_path, tmp_path / "test.zarr", compression=None)

        assert zarr_path.exists()
        result = xr.open_zarr(zarr_path)
        assert "temperature" in result.data_vars
        result.close()

    def test_data_preserved(self, tmp_path):
        """Test data values are preserved in conversion."""
        ds = make_test_dataset()
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)

        zarr_path = netcdf_to_zarr(nc_path, tmp_path / "test.zarr", compression=None)
        result = xr.open_zarr(zarr_path)

        np.testing.assert_allclose(
            result["temperature"].values,
            ds["temperature"].values,
            rtol=1e-10,
        )
        result.close()

    def test_custom_chunks(self, tmp_path):
        """Test conversion with custom chunks."""
        ds = make_test_dataset(nt=48)
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)

        zarr_path = netcdf_to_zarr(
            nc_path,
            tmp_path / "test.zarr",
            chunks={"time": 12},
            compression=None,
        )

        result = xr.open_zarr(zarr_path)
        assert result.sizes["time"] == 48
        result.close()

    def test_compression(self, tmp_path):
        """Test conversion with compression."""
        pytest.importorskip("numcodecs")

        ds = make_test_dataset()
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)

        zarr_path = netcdf_to_zarr(
            nc_path,
            tmp_path / "test.zarr",
            compression="zstd",
        )

        assert zarr_path.exists()
        result = xr.open_zarr(zarr_path)
        assert result.sizes["time"] == 24
        result.close()

    def test_missing_file_raises(self, tmp_path):
        """Test missing NetCDF file raises error."""
        with pytest.raises(FileNotFoundError):
            netcdf_to_zarr(
                tmp_path / "nonexistent.nc",
                tmp_path / "test.zarr",
            )


class TestZarrToNetCDF:
    """Zarr to NetCDF conversion tests."""

    def test_basic_conversion(self, tmp_path):
        """Test basic Zarr to NetCDF conversion."""
        ds = make_test_dataset()
        zarr_path = tmp_path / "test.zarr"
        ds.to_zarr(zarr_path)

        nc_path = zarr_to_netcdf(zarr_path, tmp_path / "test.nc")

        assert nc_path.exists()
        result = xr.open_dataset(nc_path)
        assert "temperature" in result.data_vars
        result.close()

    def test_data_preserved(self, tmp_path):
        """Test data values are preserved in conversion."""
        ds = make_test_dataset()
        zarr_path = tmp_path / "test.zarr"
        ds.to_zarr(zarr_path)

        nc_path = zarr_to_netcdf(zarr_path, tmp_path / "test.nc")
        result = xr.open_dataset(nc_path)

        np.testing.assert_allclose(
            result["temperature"].values,
            ds["temperature"].values,
            rtol=1e-10,
        )
        result.close()

    def test_unlimited_dims(self, tmp_path):
        """Test conversion with unlimited dimensions."""
        ds = make_test_dataset()
        zarr_path = tmp_path / "test.zarr"
        ds.to_zarr(zarr_path)

        nc_path = zarr_to_netcdf(
            zarr_path,
            tmp_path / "test.nc",
            unlimited_dims=["time"],
        )

        assert nc_path.exists()

    def test_missing_store_raises(self, tmp_path):
        """Test missing Zarr store raises error."""
        with pytest.raises(FileNotFoundError):
            zarr_to_netcdf(
                tmp_path / "nonexistent.zarr",
                tmp_path / "test.nc",
            )


class TestRoundTrip:
    """Round-trip conversion tests."""

    def test_netcdf_zarr_netcdf(self, tmp_path):
        """Test NetCDF -> Zarr -> NetCDF round trip."""
        ds = make_test_dataset()
        nc1_path = tmp_path / "original.nc"
        ds.to_netcdf(nc1_path)

        zarr_path = netcdf_to_zarr(nc1_path, tmp_path / "intermediate.zarr", compression=None)
        nc2_path = zarr_to_netcdf(zarr_path, tmp_path / "final.nc")

        result = xr.open_dataset(nc2_path)

        np.testing.assert_allclose(
            result["temperature"].values,
            ds["temperature"].values,
            rtol=1e-10,
        )
        result.close()


class TestGetZarrInfo:
    """get_zarr_info tests."""

    def test_returns_info(self, tmp_path):
        """Test get_zarr_info returns store information."""
        ds = make_test_dataset()
        zarr_path = tmp_path / "test.zarr"
        ds.to_zarr(zarr_path)

        info = get_zarr_info(zarr_path)

        assert info["path"] == str(zarr_path)
        assert info["size_bytes"] > 0
        assert "temperature" in info["variables"]
        assert info["dimensions"]["time"] == 24

    def test_missing_store_raises(self, tmp_path):
        """Test missing store raises error."""
        with pytest.raises(FileNotFoundError):
            get_zarr_info(tmp_path / "nonexistent.zarr")


class TestGetNetCDFInfo:
    """get_netcdf_info tests."""

    def test_returns_info(self, tmp_path):
        """Test get_netcdf_info returns file information."""
        ds = make_test_dataset()
        nc_path = tmp_path / "test.nc"
        ds.to_netcdf(nc_path)

        info = get_netcdf_info(nc_path)

        assert info["path"] == str(nc_path)
        assert info["size_bytes"] > 0
        assert "temperature" in info["variables"]
        assert info["dimensions"]["time"] == 24

    def test_missing_file_raises(self, tmp_path):
        """Test missing file raises error."""
        with pytest.raises(FileNotFoundError):
            get_netcdf_info(tmp_path / "nonexistent.nc")


class TestComputeOptimalChunks:
    """compute_optimal_chunks tests."""

    def test_small_array_uses_full_shape(self):
        """Test small arrays use full dimension size."""
        shape = (100, 50, 50)
        dtype = np.float64
        # 100 * 50 * 50 * 8 bytes = 2 MB, well under 100 MB target

        chunks = compute_optimal_chunks(shape, dtype, target_mb=100)

        # Should use full dimensions
        assert chunks["dim_0"] == 100
        assert chunks["dim_1"] == 50
        assert chunks["dim_2"] == 50

    def test_large_array_chunks_reduced(self):
        """Test large arrays have reduced chunk sizes."""
        shape = (10000, 500, 500)
        dtype = np.float64
        # 10000 * 500 * 500 * 8 bytes = 20 GB

        chunks = compute_optimal_chunks(shape, dtype, target_mb=100)

        # At least one dimension should be chunked
        total_chunk_size = (
            chunks["dim_0"] * chunks["dim_1"] * chunks["dim_2"] * 8
        )
        assert total_chunk_size <= 200 * 1024 * 1024  # Allow some margin

    def test_custom_dim_names(self):
        """Test custom dimension names."""
        shape = (100, 50, 50)
        dtype = np.float32
        dim_names = ("time", "lat", "lon")

        chunks = compute_optimal_chunks(shape, dtype, target_mb=100, dim_names=dim_names)

        assert "time" in chunks
        assert "lat" in chunks
        assert "lon" in chunks

    def test_mismatched_dim_names_raises(self):
        """Test mismatched dimension names raises error."""
        shape = (100, 50, 50)
        dtype = np.float32
        dim_names = ("time", "lat")  # Only 2 names for 3D array

        with pytest.raises(ValueError, match="dim_names length"):
            compute_optimal_chunks(shape, dtype, dim_names=dim_names)


class TestChunksForNWP:
    """chunks_for_nwp preset tests."""

    def test_era5_preset(self):
        """Test ERA5 preset chunks."""
        chunks = chunks_for_nwp("era5")

        assert chunks["time"] == 24
        assert "latitude" in chunks
        assert "longitude" in chunks

    def test_downscaled_preset(self):
        """Test downscaled data preset."""
        chunks = chunks_for_nwp("downscaled")

        assert chunks["time"] == 168
        assert chunks["unit_id"] == -1

    def test_unknown_falls_back_to_default(self):
        """Test unknown source type falls back to default."""
        chunks = chunks_for_nwp("unknown_source")

        assert "time" in chunks

    def test_variation_matching(self):
        """Test variations are matched correctly."""
        # ERA5-Land should match era5_land
        chunks = chunks_for_nwp("era5-land")
        assert chunks["time"] == 24


class TestResolveChunks:
    """resolve_chunks tests."""

    def test_full_dimension_marker(self):
        """Test -1 resolves to full dimension size."""
        chunks = {"time": 24, "unit_id": -1}
        shape_dict = {"time": 100, "unit_id": 50}

        resolved = resolve_chunks(chunks, shape_dict)

        assert resolved["time"] == 24
        assert resolved["unit_id"] == 50

    def test_chunk_exceeds_dimension(self):
        """Test chunk larger than dimension is clamped."""
        chunks = {"time": 100}
        shape_dict = {"time": 24, "space": 10}

        resolved = resolve_chunks(chunks, shape_dict)

        assert resolved["time"] == 24  # Clamped
        assert resolved["space"] == 10  # Default to full

    def test_missing_dims_use_full(self):
        """Test missing dimensions default to full size."""
        chunks = {"time": 12}
        shape_dict = {"time": 100, "lat": 50, "lon": 100}

        resolved = resolve_chunks(chunks, shape_dict)

        assert resolved["time"] == 12
        assert resolved["lat"] == 50
        assert resolved["lon"] == 100
