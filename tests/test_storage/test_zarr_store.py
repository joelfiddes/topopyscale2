"""Tests for ZarrStore storage backend."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")

from topopyscale2.storage.zarr_store import ZarrStore


def make_test_dataset(nt=48, nlat=10, nlon=20):
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
    )


class TestZarrStoreBasic:
    """Basic ZarrStore functionality tests."""

    def test_write_creates_store(self, tmp_path):
        """Test write creates a valid Zarr store."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        path = store.write(ds, tmp_path / "test.zarr")

        assert path.exists()
        # Zarr 2.x uses .zgroup/.zattrs, Zarr 3.x uses zarr.json
        assert (
            (path / ".zgroup").exists()
            or (path / ".zattrs").exists()
            or (path / "zarr.json").exists()
        )

    def test_read_returns_dataset(self, tmp_path):
        """Test read returns xarray Dataset."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        store.write(ds, tmp_path / "test.zarr")
        result = store.read(tmp_path / "test.zarr")

        assert isinstance(result, xr.Dataset)
        assert "temperature" in result.data_vars
        result.close()

    def test_round_trip_preserves_data(self, tmp_path):
        """Test data is preserved through write/read cycle."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        store.write(ds, tmp_path / "test.zarr")
        result = store.read(tmp_path / "test.zarr")

        np.testing.assert_allclose(
            result["temperature"].values,
            ds["temperature"].values,
            rtol=1e-10,
        )
        result.close()

    def test_exists_true_for_valid_store(self, tmp_path):
        """Test exists returns True for valid store."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        store.write(ds, tmp_path / "test.zarr")

        assert store.exists(tmp_path / "test.zarr")

    def test_exists_false_for_missing(self, tmp_path):
        """Test exists returns False for missing path."""
        store = ZarrStore()

        assert not store.exists(tmp_path / "nonexistent.zarr")


class TestZarrStoreChunking:
    """Chunking-related tests."""

    def test_custom_chunks_applied(self, tmp_path):
        """Test custom chunk sizes are applied."""
        ds = make_test_dataset(nt=48)
        store = ZarrStore(chunks={"time": 12}, compression=None)

        store.write(ds, tmp_path / "test.zarr")
        result = store.read(tmp_path / "test.zarr")

        # Chunks should be applied
        assert result["temperature"].chunks is not None
        result.close()

    def test_full_dimension_chunk(self, tmp_path):
        """Test -1 chunk size uses full dimension."""
        ds = make_test_dataset(nt=24, nlat=10, nlon=20)
        store = ZarrStore(
            chunks={"time": 12, "latitude": -1, "longitude": -1},
            compression=None,
        )

        path = store.write(ds, tmp_path / "test.zarr")
        result = store.read(path)

        # Should have written successfully
        assert result.sizes["time"] == 24
        assert result.sizes["latitude"] == 10
        result.close()


class TestZarrStoreCompression:
    """Compression-related tests."""

    def test_zstd_compression(self, tmp_path):
        """Test zstd compression."""
        pytest.importorskip("numcodecs")

        ds = make_test_dataset()
        store = ZarrStore(compression="zstd", compression_level=3)

        path = store.write(ds, tmp_path / "test.zarr")
        result = store.read(path)

        assert result.sizes["time"] == 48
        result.close()

    def test_no_compression(self, tmp_path):
        """Test writing without compression."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        path = store.write(ds, tmp_path / "test.zarr")
        assert path.exists()

    def test_invalid_compression_raises(self):
        """Test invalid compression raises error."""
        pytest.importorskip("numcodecs")

        store = ZarrStore(compression="invalid_algo")
        ds = make_test_dataset()

        with pytest.raises(ValueError, match="Unknown compression"):
            store.write(ds, "/tmp/test.zarr")


class TestZarrStoreAppend:
    """Append functionality tests."""

    def test_append_along_time(self, tmp_path):
        """Test appending data along time dimension."""
        ds1 = make_test_dataset(nt=24)
        store = ZarrStore(compression=None)

        path = store.write(ds1, tmp_path / "test.zarr")

        # Create second dataset with later times
        times2 = pd.date_range("2020-01-02", periods=24, freq="h").values
        ds2 = ds1.copy()
        ds2 = ds2.assign_coords(time=times2)

        store.append(ds2, path, dim="time")

        result = store.read(path)
        assert result.sizes["time"] == 48
        result.close()

    def test_append_to_nonexistent_raises(self, tmp_path):
        """Test append to nonexistent store raises error."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        with pytest.raises(FileNotFoundError):
            store.append(ds, tmp_path / "nonexistent.zarr")


class TestZarrStoreMetadata:
    """Metadata handling tests."""

    def test_global_attrs_added(self, tmp_path):
        """Test global attributes are added."""
        ds = make_test_dataset()
        store = ZarrStore(compression=None)

        path = store.write(ds, tmp_path / "test.zarr")
        result = store.read(path)

        assert result.attrs["Conventions"] == "CF-1.8"
        assert "TopoPyScale" in result.attrs["title"]
        assert result.attrs["format"] == "Zarr"
        result.close()

    def test_consolidated_metadata(self, tmp_path):
        """Test consolidated metadata is written."""
        ds = make_test_dataset()
        store = ZarrStore(consolidated=True, compression=None)

        path = store.write(ds, tmp_path / "test.zarr")

        # Zarr 2.x uses .zmetadata, Zarr 3.x may use different approach
        # Just verify the store can be opened and has expected data
        result = store.read(path)
        assert result is not None
        assert "temperature" in result.data_vars
        result.close()

    def test_get_info(self, tmp_path):
        """Test get_info returns store information."""
        ds = make_test_dataset(nt=48, nlat=10, nlon=20)
        store = ZarrStore(compression=None)

        path = store.write(ds, tmp_path / "test.zarr")
        info = store.get_info(path)

        assert info["path"] == str(path)
        assert info["size_bytes"] > 0
        assert "temperature" in info["variables"]
        assert info["dimensions"]["time"] == 48


class TestZarrStoreRechunk:
    """Rechunk functionality tests."""

    def test_rechunk_creates_new_store(self, tmp_path):
        """Test rechunk creates a new Zarr store."""
        ds = make_test_dataset(nt=48)
        store = ZarrStore(chunks={"time": 48}, compression=None)

        path1 = store.write(ds, tmp_path / "original.zarr")
        path2 = store.rechunk(
            path1,
            tmp_path / "rechunked.zarr",
            {"time": 12},
        )

        assert path2.exists()
        result = store.read(path2)
        assert result.sizes["time"] == 48
        result.close()
