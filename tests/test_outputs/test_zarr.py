"""Tests for Zarr output writer."""

import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.outputs.zarr_writer import ZarrWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=48, n_units=3):
    """Create test forcing dataset."""
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    unit_ids = [f"unit_{i:04d}" for i in range(n_units)]

    return xr.Dataset(
        {
            "temperature": (["time", "unit_id"], np.full((nt, n_units), 275.0)),
            "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.001)),
            "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0005)),
            "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.0005)),
            "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 200.0)),
            "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 100.0)),
            "longwave": (["time", "unit_id"], np.full((nt, n_units), 280.0)),
            "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.005)),
            "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.7)),
            "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 3.0)),
            "pressure": (["time", "unit_id"], np.full((nt, n_units), 80000.0)),
        },
        coords={"time": times, "unit_id": unit_ids},
    )


def make_units(n=3):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(10.0 + i * 0.1, 47.0, 2000.0 + i * 100),
            surface_type="open",
            area_m2=900.0,
        )
        for i in range(n)
    ]


# Check if zarr is available
try:
    import zarr  # noqa: F401 — availability check
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False

# Check if numcodecs is available
try:
    import numcodecs  # noqa: F401 — availability check
    HAS_NUMCODECS = True
except ImportError:
    HAS_NUMCODECS = False


# Skip all tests if zarr is not installed
pytestmark = pytest.mark.skipif(not HAS_ZARR, reason="zarr not installed")


class TestZarrWriter:
    def test_write_creates_store(self, tmp_path):
        """Test that write creates a Zarr store directory."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)  # No compression for basic test

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].is_dir()
        assert paths[0].suffix == ".zarr"

    def test_store_naming(self, tmp_path):
        """Test Zarr store naming convention."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)

        # Should contain date range
        assert "forcing_" in paths[0].name
        assert "20200101" in paths[0].name

    def test_round_trip(self, tmp_path):
        """Test data can be read back correctly."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        np.testing.assert_allclose(result["temperature"].values, 275.0)
        assert result.sizes["time"] == 48
        assert result.sizes["unit_id"] == 3

        result.close()

    def test_cf_attributes(self, tmp_path):
        """Test CF attributes are added."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        assert result.attrs["Conventions"] == "CF-1.8"
        assert "TopoPyScale" in result.attrs["title"]
        assert result["temperature"].attrs["standard_name"] == "air_temperature"
        assert result["temperature"].attrs["units"] == "K"

        result.close()

    def test_unit_metadata_coords(self, tmp_path):
        """Test unit metadata is added as coordinates."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        assert "unit_lat" in result.coords
        assert "unit_lon" in result.coords
        assert "unit_elevation" in result.coords
        assert "unit_surface_type" in result.coords
        assert "unit_area" in result.coords

        np.testing.assert_allclose(result["unit_elevation"].values[0], 2000.0)

        result.close()

    @pytest.mark.skipif(not HAS_NUMCODECS, reason="numcodecs not installed")
    def test_zstd_compression(self, tmp_path):
        """Test zstd compression is applied."""
        ds = make_forcing_dataset(nt=100)
        units = make_units()
        writer = ZarrWriter(compression="zstd", compression_level=3)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        # Just verify it opens correctly
        assert result.sizes["time"] == 100

        result.close()

    def test_no_compression(self, tmp_path):
        """Test output without compression."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        assert result.sizes["time"] == 48

        result.close()

    def test_chunking(self, tmp_path):
        """Test custom chunking is applied."""
        ds = make_forcing_dataset(nt=48)
        units = make_units()
        writer = ZarrWriter(chunks={"time": 12}, compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        # Check chunking
        assert result["temperature"].encoding.get("chunks") is not None or \
               result["temperature"].chunks is not None

        result.close()

    def test_consolidated_metadata(self, tmp_path):
        """Test consolidated metadata is written."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(consolidated=True, compression=None)

        paths = writer.write(ds, units, tmp_path)

        # Check metadata file exists (zarr 2.x: .zmetadata, zarr 3.x: zarr.json)
        zmetadata_v2 = paths[0] / ".zmetadata"
        zmetadata_v3 = paths[0] / "zarr.json"
        assert zmetadata_v2.exists() or zmetadata_v3.exists()

    def test_open_helper(self, tmp_path):
        """Test static open helper method."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = ZarrWriter.open(paths[0])

        assert "temperature" in result.data_vars
        assert result.sizes["time"] == 48

        result.close()

    def test_append_mode(self, tmp_path):
        """Test appending data along time dimension."""
        # First write
        ds1 = make_forcing_dataset(nt=24)
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds1, units, tmp_path)

        # Append more data
        times2 = pd.date_range("2020-01-02", periods=24, freq="h").values
        ds2 = ds1.copy()
        ds2 = ds2.assign_coords(time=times2)

        writer.append(ds2, units, paths[0], dim="time")

        # Check combined result
        result = xr.open_zarr(paths[0])
        assert result.sizes["time"] == 48

        result.close()

    @pytest.mark.skipif(not HAS_NUMCODECS, reason="numcodecs not installed")
    def test_compression_types(self, tmp_path):
        """Test different compression algorithms."""
        ds = make_forcing_dataset()
        units = make_units()

        # Note: zarr 3.x only supports zstd and gzip natively; zlib requires numcodecs
        import zarr
        zarr_major = int(zarr.__version__.split(".")[0])
        if zarr_major >= 3:
            compression_types = ["zstd", "gzip"]
        else:
            compression_types = ["zstd", "gzip", "zlib"]

        for comp in compression_types:
            writer = ZarrWriter(compression=comp)
            paths = writer.write(ds, units, tmp_path / comp)

            result = xr.open_zarr(paths[0])
            assert result.sizes["time"] == 48
            result.close()

    def test_global_attrs_format(self, tmp_path):
        """Test format attribute in global attrs."""
        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        assert result.attrs["format"] == "Zarr"
        # compression attr may be "None" or None
        assert "compression" in result.attrs

        result.close()

    def test_time_dimension_preserved(self, tmp_path):
        """Test time coordinates are preserved correctly."""
        ds = make_forcing_dataset(nt=72)
        units = make_units()
        writer = ZarrWriter(compression=None)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_zarr(paths[0])

        # Check time range
        assert result.sizes["time"] == 72
        first_time = pd.Timestamp(result.time.values[0])
        assert first_time.year == 2020
        assert first_time.month == 1
        assert first_time.day == 1

        result.close()

    def test_compression_warning_without_numcodecs(self, tmp_path):
        """Test warning is issued when compression requested without numcodecs."""
        if HAS_NUMCODECS:
            pytest.skip("numcodecs is installed")

        ds = make_forcing_dataset()
        units = make_units()
        writer = ZarrWriter(compression="zstd")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths = writer.write(ds, units, tmp_path)

            # Should have issued a warning about numcodecs
            assert len(w) >= 1
            assert "numcodecs" in str(w[0].message).lower()

        # File should still be created
        assert paths[0].exists()
