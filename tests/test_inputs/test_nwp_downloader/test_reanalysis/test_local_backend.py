"""Tests for the LocalCacheBackend."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.inputs.nwp_downloader.bbox import BBox


@pytest.fixture
def cache_store(tmp_path):
    """Create a synthetic Zarr cache store with 3 days of data."""
    n_times_per_day = 24
    n_days = 3
    n_lats = 5
    n_lons = 5
    n_levels = 3

    times = pd.date_range("2020-06-15", periods=n_times_per_day * n_days, freq="1h")
    lats = np.linspace(46.5, 45.5, n_lats)  # Descending (N to S)
    lons = np.linspace(7.0, 8.0, n_lons)
    levels = np.array([300, 700, 1000])

    np.random.seed(42)
    shape_3d = (len(times), n_lats, n_lons)
    shape_4d = (len(times), n_levels, n_lats, n_lons)

    ds = xr.Dataset(
        {
            # Surface variables
            "t2m": (["time", "latitude", "longitude"], 273 + 15 * np.random.rand(*shape_3d)),
            "d2m": (["time", "latitude", "longitude"], 268 + 10 * np.random.rand(*shape_3d)),
            "sp": (["time", "latitude", "longitude"], 85000 + 5000 * np.random.rand(*shape_3d)),
            "ssrd": (["time", "latitude", "longitude"], 1e6 * np.random.rand(*shape_3d)),
            "strd": (["time", "latitude", "longitude"], 0.8e6 * np.random.rand(*shape_3d)),
            "tp": (["time", "latitude", "longitude"], 0.001 * np.random.rand(*shape_3d)),
            "z_surf": (["time", "latitude", "longitude"], 9.81 * 1500 * np.ones(shape_3d)),
            "u10": (["time", "latitude", "longitude"], 5 * np.random.rand(*shape_3d)),
            "v10": (["time", "latitude", "longitude"], 5 * np.random.rand(*shape_3d)),
            "tisr": (["time", "latitude", "longitude"], 1e7 * np.random.rand(*shape_3d)),
            # Pressure-level variables
            "t": (["time", "level", "latitude", "longitude"], 250 + 30 * np.random.rand(*shape_4d)),
            "z": (["time", "level", "latitude", "longitude"], 9.81 * 5000 * np.random.rand(*shape_4d)),
            "u": (["time", "level", "latitude", "longitude"], 20 * np.random.rand(*shape_4d)),
            "v": (["time", "level", "latitude", "longitude"], 20 * np.random.rand(*shape_4d)),
            "q": (["time", "level", "latitude", "longitude"], 0.01 * np.random.rand(*shape_4d)),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
            "level": levels,
        },
        attrs={
            "tps2_cache_version": "1.0",
            "tps2_surf_vars": ["d2m", "sp", "ssrd", "strd", "t2m", "tp", "z_surf", "u10", "v10", "tisr"],
            "tps2_plev_vars": ["q", "t", "u", "v", "z"],
            "tps2_bbox": [7.0, 45.5, 8.0, 46.5],
            "tps2_pressure_levels": [300, 700, 1000],
            "tps2_time_resolution": "1H",
        },
    )

    store_path = tmp_path / "test_cache.zarr"
    ds.to_zarr(str(store_path))
    return store_path


@pytest.fixture
def cache_store_no_attrs(tmp_path):
    """Create a Zarr store without tps2 metadata attrs (legacy/external store)."""
    times = pd.date_range("2020-06-15", periods=24, freq="1h")
    lats = np.array([46.0, 45.5])
    lons = np.array([7.5, 8.0])
    levels = np.array([700, 1000])

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], np.ones((24, 2, 2))),
            "sp": (["time", "latitude", "longitude"], 90000 * np.ones((24, 2, 2))),
            "z_surf": (["time", "latitude", "longitude"], 15000 * np.ones((24, 2, 2))),
            "t": (["time", "level", "latitude", "longitude"], 260 * np.ones((24, 2, 2, 2))),
            "z": (["time", "level", "latitude", "longitude"], 50000 * np.ones((24, 2, 2, 2))),
            "q": (["time", "level", "latitude", "longitude"], 0.005 * np.ones((24, 2, 2, 2))),
        },
        coords={"time": times, "latitude": lats, "longitude": lons, "level": levels},
    )
    store_path = tmp_path / "no_attrs.zarr"
    ds.to_zarr(str(store_path))
    return store_path


class TestLocalCacheBackend:
    """Test LocalCacheBackend functionality."""

    def test_open_and_spatial_subset(self, cache_store):
        """Backend opens store and pre-selects spatial subset."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.2, 45.8, 7.6, 46.2))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        # Should have spatially subsetted
        assert backend._ds.sizes["latitude"] < 5
        assert backend._ds.sizes["longitude"] < 5
        backend.close()

    def test_fetch_day(self, cache_store):
        """fetch_day returns (ds_surf, ds_plev) with correct variables."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        ds_surf, ds_plev = backend.fetch_day(pd.Timestamp("2020-06-15"))

        # Surface should have standard vars (z_surf renamed to z)
        assert "t2m" in ds_surf
        assert "z" in ds_surf  # z_surf renamed to z
        assert "z_surf" not in ds_surf
        assert ds_surf.sizes["time"] == 24

        # Plev should have selected levels only
        assert "t" in ds_plev
        assert "q" in ds_plev
        assert set(ds_plev.level.values.tolist()) == {700, 1000}

        # Levels should be ascending
        assert list(ds_plev.level.values) == [700, 1000]

        backend.close()

    def test_fetch_day_missing_date(self, cache_store):
        """fetch_day raises FileNotFoundError for dates not in cache."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        with pytest.raises(FileNotFoundError):
            backend.fetch_day(pd.Timestamp("2025-01-01"))

        backend.close()

    def test_probe_date(self, cache_store):
        """probe_date returns True for existing dates, False for missing."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        assert backend.probe_date(pd.Timestamp("2020-06-15")) is True
        assert backend.probe_date(pd.Timestamp("2020-06-16")) is True
        assert backend.probe_date(pd.Timestamp("2025-01-01")) is False

        backend.close()

    def test_level_selection(self, cache_store):
        """Only requested levels that exist in the store are returned."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        # Request level 500 which doesn't exist in the store (has 300, 700, 1000)
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[500, 700, 1000],
            cache_path=str(cache_store),
        )

        ds_surf, ds_plev = backend.fetch_day(pd.Timestamp("2020-06-15"))

        # Should only have 700 and 1000 (500 not in store)
        assert set(ds_plev.level.values.tolist()) == {700, 1000}

        backend.close()

    def test_attr_based_variable_detection(self, cache_store):
        """Backend reads variable lists from store attrs."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        assert "t2m" in backend.surf_vars
        assert "z_surf" in backend.surf_vars
        assert "t" in backend.plev_vars
        assert "q" in backend.plev_vars

        backend.close()

    def test_fallback_to_defaults_without_attrs(self, cache_store_no_attrs):
        """Backend falls back to default var lists when attrs are missing."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            DEFAULT_PLEV_VARS,
            DEFAULT_SURF_VARS,
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.0, 8.5, 46.5))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store_no_attrs),
        )

        assert backend.surf_vars == DEFAULT_SURF_VARS
        assert backend.plev_vars == DEFAULT_PLEV_VARS

        # Should still be able to fetch
        ds_surf, ds_plev = backend.fetch_day(pd.Timestamp("2020-06-15"))
        assert "t2m" in ds_surf
        assert "t" in ds_plev

        backend.close()

    def test_missing_cache_path_raises(self):
        """Constructor raises ValueError when cache_path is empty."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        with pytest.raises(ValueError, match="cache_path is required"):
            LocalCacheBackend(
                bbox=bbox,
                pressure_levels=[700, 1000],
                cache_path="",
            )

    def test_close(self, cache_store):
        """close() sets _ds to None."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.local import (
            LocalCacheBackend,
        )

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        backend = LocalCacheBackend(
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        backend.close()
        assert backend._ds is None


class TestLocalBackendRegistry:
    """Test that local backend is properly registered."""

    def test_registry_contains_local(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import (
            BACKEND_REGISTRY,
        )

        assert "local" in BACKEND_REGISTRY

    def test_get_backend_local(self, cache_store):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import get_backend

        bbox = BBox.from_tuple((7.0, 45.5, 8.0, 46.5))
        backend = get_backend(
            "local",
            bbox=bbox,
            pressure_levels=[700, 1000],
            cache_path=str(cache_store),
        )

        assert backend is not None
        assert hasattr(backend, "fetch_day")
        backend.close()
