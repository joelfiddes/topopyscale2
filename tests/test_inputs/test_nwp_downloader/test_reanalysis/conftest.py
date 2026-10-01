"""Shared fixtures for reanalysis module tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.inputs.nwp_downloader.bbox import BBox


@pytest.fixture
def sample_bbox() -> BBox:
    """Sample bounding box for testing (Zermatt area, Switzerland)."""
    return BBox.from_tuple((7.7, 45.95, 7.85, 46.05))


@pytest.fixture
def sample_pressure_levels() -> list[int]:
    """Standard pressure levels for testing."""
    return [1000, 850, 700, 500, 300]


@pytest.fixture
def sample_date() -> pd.Timestamp:
    """Sample date for testing."""
    return pd.Timestamp("2020-06-15")


@pytest.fixture
def sample_surface_dataset(sample_date: pd.Timestamp) -> xr.Dataset:
    """Create a sample surface ERA5 dataset for one day."""
    n_times = 24
    n_lats = 3
    n_lons = 3

    times = pd.date_range(sample_date, periods=n_times, freq="1h")
    lats = np.array([46.05, 46.0, 45.95])  # Descending
    lons = np.array([7.7, 7.775, 7.85])

    np.random.seed(42)

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"],
                    273.15 + 15 * np.random.rand(n_times, n_lats, n_lons)),
            "d2m": (["time", "latitude", "longitude"],
                    268.15 + 10 * np.random.rand(n_times, n_lats, n_lons)),
            "sp": (["time", "latitude", "longitude"],
                   85000 + 5000 * np.random.rand(n_times, n_lats, n_lons)),
            "ssrd": (["time", "latitude", "longitude"],
                     np.cumsum(1e6 * np.random.rand(n_times, n_lats, n_lons), axis=0)),
            "strd": (["time", "latitude", "longitude"],
                     np.cumsum(0.8e6 * np.random.rand(n_times, n_lats, n_lons), axis=0)),
            "tp": (["time", "latitude", "longitude"],
                   np.cumsum(0.0005 * np.random.rand(n_times, n_lats, n_lons), axis=0)),
            "z": (["time", "latitude", "longitude"],
                  9.81 * 1500 * np.ones((n_times, n_lats, n_lons))),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


@pytest.fixture
def sample_pressure_dataset(
    sample_date: pd.Timestamp,
    sample_pressure_levels: list[int],
) -> xr.Dataset:
    """Create a sample pressure-level ERA5 dataset for one day."""
    n_times = 24
    n_levels = len(sample_pressure_levels)
    n_lats = 3
    n_lons = 3

    times = pd.date_range(sample_date, periods=n_times, freq="1h")
    lats = np.array([46.05, 46.0, 45.95])
    lons = np.array([7.7, 7.775, 7.85])
    levels = np.array(sorted(sample_pressure_levels))

    np.random.seed(42)

    ds = xr.Dataset(
        {
            "t": (["time", "level", "latitude", "longitude"],
                  250 + 30 * np.random.rand(n_times, n_levels, n_lats, n_lons)),
            "z": (["time", "level", "latitude", "longitude"],
                  9.81 * (5000 + 3000 * np.random.rand(n_times, n_levels, n_lats, n_lons))),
            "u": (["time", "level", "latitude", "longitude"],
                  20 * (np.random.rand(n_times, n_levels, n_lats, n_lons) - 0.5)),
            "v": (["time", "level", "latitude", "longitude"],
                  20 * (np.random.rand(n_times, n_levels, n_lats, n_lons) - 0.5)),
            "q": (["time", "level", "latitude", "longitude"],
                  0.01 * np.random.rand(n_times, n_levels, n_lats, n_lons)),
            "r": (["time", "level", "latitude", "longitude"],
                  100 * np.random.rand(n_times, n_levels, n_lats, n_lons)),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds
