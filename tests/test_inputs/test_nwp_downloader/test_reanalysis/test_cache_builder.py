"""Tests for the CacheBuilder."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr


def _make_synthetic_day(date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
    """Create synthetic surface + plev datasets for one day."""
    times = pd.date_range(date, periods=24, freq="1h")
    lats = np.array([46.0, 45.75, 45.5])
    lons = np.array([7.0, 7.5, 8.0])
    levels = np.array([700, 1000])

    np.random.seed(int(date.day_of_year))
    shape_3d = (24, 3, 3)
    shape_4d = (24, 2, 3, 3)

    ds_surf = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], 273 + np.random.rand(*shape_3d)),
            "sp": (["time", "latitude", "longitude"], 90000 * np.ones(shape_3d)),
            "z": (["time", "latitude", "longitude"], 15000 * np.ones(shape_3d)),
            "tp": (["time", "latitude", "longitude"], 0.001 * np.random.rand(*shape_3d)),
        },
        coords={"time": times, "latitude": lats, "longitude": lons},
    )

    ds_plev = xr.Dataset(
        {
            "t": (["time", "level", "latitude", "longitude"], 260 + np.random.rand(*shape_4d)),
            "z": (["time", "level", "latitude", "longitude"], 50000 * np.random.rand(*shape_4d)),
            "q": (["time", "level", "latitude", "longitude"], 0.005 * np.ones(shape_4d)),
        },
        coords={"time": times, "level": levels, "latitude": lats, "longitude": lons},
    )

    return ds_surf, ds_plev


class FakeBackend:
    """Fake backend that returns synthetic data."""

    def __init__(self, **kwargs):
        pass

    def fetch_day(self, date):
        return _make_synthetic_day(date)

    def probe_date(self, date):
        return True

    def close(self):
        pass


@pytest.fixture
def existing_cache(tmp_path):
    """Create a small existing cache for update tests."""
    times = pd.date_range("2020-06-15", periods=48, freq="1h")  # 2 days
    lats = np.array([46.0, 45.75, 45.5])
    lons = np.array([7.0, 7.5, 8.0])
    levels = np.array([700, 1000])

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], np.ones((48, 3, 3))),
            "sp": (["time", "latitude", "longitude"], 90000 * np.ones((48, 3, 3))),
            "z_surf": (["time", "latitude", "longitude"], 15000 * np.ones((48, 3, 3))),
            "tp": (["time", "latitude", "longitude"], 0.001 * np.ones((48, 3, 3))),
            "t": (["time", "level", "latitude", "longitude"], 260 * np.ones((48, 2, 3, 3))),
            "z": (["time", "level", "latitude", "longitude"], 50000 * np.ones((48, 2, 3, 3))),
            "q": (["time", "level", "latitude", "longitude"], 0.005 * np.ones((48, 2, 3, 3))),
        },
        coords={"time": times, "latitude": lats, "longitude": lons, "level": levels},
        attrs={
            "tps2_cache_version": "1.0",
            "tps2_surf_vars": ["t2m", "sp", "z_surf", "tp"],
            "tps2_plev_vars": ["t", "z", "q"],
            "tps2_bbox": [7.0, 45.5, 8.0, 46.0],
            "tps2_pressure_levels": [700, 1000],
            "tps2_time_resolution": "1H",
        },
    )

    store_path = tmp_path / "existing.zarr"
    ds.to_zarr(str(store_path))
    return store_path


class TestCacheBuilderInfo:
    """Test CacheBuilder.info() static method."""

    def test_info_returns_metadata(self, existing_cache):
        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        meta = CacheBuilder.info(existing_cache)

        assert meta["cache_version"] == "1.0"
        assert meta["bbox"] == [7.0, 45.5, 8.0, 46.0]
        assert meta["time_range"] == ["2020-06-15", "2020-06-16"]
        assert meta["n_days"] == 2
        assert meta["expected_days"] == 2
        assert meta["missing_days"] == []
        assert meta["pressure_levels"] == [700, 1000]
        assert "t2m" in meta["surf_vars"]
        assert "t" in meta["plev_vars"]
        assert meta["time_resolution"] == "1H"
        assert meta["size_gb"] >= 0  # May round to 0.0 for tiny test stores

    def test_info_nonexistent_raises(self, tmp_path):
        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        with pytest.raises(FileNotFoundError):
            CacheBuilder.info(tmp_path / "nonexistent.zarr")


class TestCacheBuilderBuild:
    """Test CacheBuilder.build() with mocked ERA5Loader."""

    def test_build_creates_zarr_store(self, tmp_path):
        """_merge_to_cache creates a valid Zarr store with metadata attrs."""
        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        output_path = tmp_path / "test_build.zarr"

        builder = CacheBuilder(
            backend="google",
            bbox=(7.0, 45.5, 8.0, 46.0),
            start_date="2020-06-15",
            end_date="2020-06-16",
            pressure_levels=[700, 1000],
            output_path=output_path,
            max_workers=1,
        )

        # Directly create staging and call _merge_to_cache
        staging_dir = tmp_path / ".cache_staging"
        daily_dir = staging_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        for date_str in ["20200615", "20200616"]:
            date = pd.Timestamp(date_str)
            ds_surf, ds_plev = _make_synthetic_day(date)
            if "z" in ds_surf:
                ds_surf = ds_surf.rename({"z": "z_surf"})
            ds = xr.merge([ds_surf, ds_plev])
            day_path = daily_dir / f"day_{date_str}.zarr"
            ds.to_zarr(str(day_path))

        builder._merge_to_cache(staging_dir)

        # Verify output
        assert output_path.exists()

        ds = xr.open_zarr(str(output_path))
        assert "tps2_cache_version" in ds.attrs
        assert ds.attrs["tps2_cache_version"] == "1.0"
        assert ds.attrs["tps2_pressure_levels"] == [700, 1000]
        assert ds.sizes["time"] == 48  # 2 days * 24 hours

        # Verify data variables exist
        assert "t2m" in ds
        assert "t" in ds
        assert "z_surf" in ds

        # Staging should be cleaned up
        assert not staging_dir.exists()

        ds.close()

    def test_merge_includes_existing_cache(self, tmp_path, existing_cache):
        """_merge_to_cache includes existing store when updating."""
        import shutil

        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        # Copy existing cache to output path
        output_path = tmp_path / "updated.zarr"
        shutil.copytree(str(existing_cache), str(output_path))

        builder = CacheBuilder(
            backend="google",
            bbox=(7.0, 45.5, 8.0, 46.0),
            start_date="2020-06-17",
            end_date="2020-06-17",
            pressure_levels=[700, 1000],
            output_path=output_path,
        )

        # Create staging with one new day
        staging_dir = tmp_path / ".cache_staging"
        daily_dir = staging_dir / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        date = pd.Timestamp("2020-06-17")
        ds_surf, ds_plev = _make_synthetic_day(date)
        if "z" in ds_surf:
            ds_surf = ds_surf.rename({"z": "z_surf"})
        ds = xr.merge([ds_surf, ds_plev])
        ds.to_zarr(str(daily_dir / "day_20200617.zarr"))

        builder._merge_to_cache(staging_dir)

        # Should now have 3 days
        ds = xr.open_zarr(str(output_path))
        unique_days = pd.DatetimeIndex(ds.time.values).normalize().unique()
        assert len(unique_days) == 3
        ds.close()


class TestCacheBuilderUpdate:
    """Test CacheBuilder.update() class method."""

    def test_update_reads_attrs_from_existing(self, existing_cache):
        """update() reads bbox, levels, etc. from existing cache attrs."""
        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        # The existing cache ends at 2020-06-16.
        # Since that's far in the past, update should find new_start <= new_end
        # But we can't actually download, so just verify it reads attrs correctly.
        # We'll mock CacheBuilder.__init__ and build() to capture the arguments.

        init_args = {}

        def capture_init(self, **kwargs):
            init_args.update(kwargs)

        with patch.object(CacheBuilder, "__init__", capture_init), \
             patch.object(CacheBuilder, "build"):
            CacheBuilder.update(existing_cache, backend="google")

        assert init_args["bbox"] == (7.0, 45.5, 8.0, 46.0)
        assert init_args["pressure_levels"] == [700, 1000]
        assert init_args["time_resolution"] == "1H"
        assert init_args["start_date"] == "2020-06-17"

    def test_update_nonexistent_raises(self, tmp_path):
        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        with pytest.raises(FileNotFoundError):
            CacheBuilder.update(tmp_path / "nonexistent.zarr")


class TestCacheBuilderCompressor:
    """Test compression helper."""

    def test_get_compressor_returns_something(self):
        from topopyscale2.inputs.nwp_downloader.cache_builder import CacheBuilder

        compressor, key = CacheBuilder._get_compressor()
        assert key in ("compressor", "compressors")
        # Compressor may be None if no codec available, but key should be set
