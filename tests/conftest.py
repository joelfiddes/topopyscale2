"""Shared test fixtures for TopoPyScale 2.0."""

import os

# typer.rich_utils forces a rich color terminal at import time when it sees
# GITHUB_ACTIONS/FORCE_COLOR, which injects ANSI codes inside option names
# and breaks CLI --help substring assertions. Neutralize before any test
# module imports typer.
for _var in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS"):
    os.environ.pop(_var, None)

import numpy as np
import pandas as pd
import pytest
import xarray as xr

if os.environ.get("TPS2_EXPECT_INSTALLED"):
    # Set by scripts/release/clean_install_check.sh: the suite must exercise the
    # INSTALLED package (with its compiled kernels), not the source tree beside it.
    from pathlib import Path

    import topopyscale2

    _pkg = Path(topopyscale2.__file__).resolve()
    if _pkg.is_relative_to(Path(__file__).resolve().parents[1]):
        raise RuntimeError(f"topopyscale2 resolves to the source tree ({_pkg}), not the install")

@pytest.fixture(autouse=True)
def _restore_cwd():
    """CLI commands such as `run` chdir into the config directory and stay there;
    without this a test that invokes one moves every later test's working directory
    (and breaks relative paths, e.g. a relative PYTHONPATH in a subprocess)."""
    cwd = os.getcwd()
    yield
    os.chdir(cwd)


# Cross-backend agreement tolerances
RTOL = 1e-10
ATOL = 1e-12


@pytest.fixture
def synthetic_dem_flat():
    """Flat DEM at 3000m elevation, 10x10 grid, 30m resolution."""
    ny, nx = 10, 10
    elevation = np.full((ny, nx), 3000.0)
    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    return xr.DataArray(elevation, dims=["y", "x"], coords={"y": y, "x": x})


@pytest.fixture
def synthetic_dem_tilted():
    """Tilted plane DEM: south-facing slope of ~26.6 degrees (rise 1m per 2m horizontal).

    10x10 grid, 30m resolution, elevation increases northward.
    """
    ny, nx = 10, 10
    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    yy, _ = np.meshgrid(y, x, indexing="ij")
    elevation = 3000.0 + yy * 0.5  # 0.5 m/m slope northward
    return xr.DataArray(elevation, dims=["y", "x"], coords={"y": y, "x": x})


@pytest.fixture
def synthetic_dem_valley():
    """V-shaped valley DEM for SVF testing.

    Cross-section is V-shaped (east-west), constant along north-south.
    """
    ny, nx = 20, 20
    x = np.arange(nx) * 30.0
    y = np.arange(ny) * 30.0
    center_x = x[nx // 2]
    _, xx = np.meshgrid(y, x, indexing="ij")
    elevation = 3000.0 + np.abs(xx - center_x) * 0.5
    return xr.DataArray(elevation, dims=["y", "x"], coords={"y": y, "x": x})


@pytest.fixture
def synthetic_era5_surface():
    """Small synthetic ERA5 surface dataset (2x2 grid, 24 hourly timesteps)."""
    times = pd.date_range("2020-01-01", periods=24, freq="h").values
    lat = np.array([42.0, 42.25])
    lon = np.array([76.0, 76.25])

    nt, nlat, nlon = len(times), len(lat), len(lon)
    rng = np.random.default_rng(42)

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], 270.0 + rng.standard_normal((nt, nlat, nlon)) * 5),
            "d2m": (["time", "latitude", "longitude"], 265.0 + rng.standard_normal((nt, nlat, nlon)) * 3),
            "sp": (["time", "latitude", "longitude"], 80000.0 + rng.standard_normal((nt, nlat, nlon)) * 500),
            "ssrd": (["time", "latitude", "longitude"], np.maximum(0, rng.standard_normal((nt, nlat, nlon)) * 500000 + 300000)),
            "strd": (["time", "latitude", "longitude"], 800000.0 + rng.standard_normal((nt, nlat, nlon)) * 50000),
            "tp": (["time", "latitude", "longitude"], np.maximum(0, rng.standard_normal((nt, nlat, nlon)) * 0.002 + 0.001)),
            "z": (["latitude", "longitude"], np.array([[25000.0, 26000.0], [24000.0, 25500.0]])),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )
    return ds


@pytest.fixture
def synthetic_era5_pressure():
    """Small synthetic ERA5 pressure level dataset (2x2 grid, 24 timesteps, 5 levels)."""
    times = pd.date_range("2020-01-01", periods=24, freq="h").values
    lat = np.array([42.0, 42.25])
    lon = np.array([76.0, 76.25])
    levels = np.array([300, 500, 700, 850, 1000])

    nt, nlev, nlat, nlon = len(times), len(levels), len(lat), len(lon)
    rng = np.random.default_rng(42)

    # Temperature decreases with altitude (lower pressure = higher altitude)
    t_base = np.array([220.0, 250.0, 270.0, 280.0, 288.0])
    t_data = t_base[np.newaxis, :, np.newaxis, np.newaxis] + rng.standard_normal((nt, nlev, nlat, nlon)) * 2

    # Geopotential increases with altitude
    z_base = np.array([9500.0, 5500.0, 3000.0, 1500.0, 100.0]) * 9.81
    z_data = z_base[np.newaxis, :, np.newaxis, np.newaxis] + rng.standard_normal((nt, nlev, nlat, nlon)) * 50

    ds = xr.Dataset(
        {
            "t": (["time", "level", "latitude", "longitude"], t_data),
            "z": (["time", "level", "latitude", "longitude"], z_data),
            "u": (["time", "level", "latitude", "longitude"], rng.standard_normal((nt, nlev, nlat, nlon)) * 10),
            "v": (["time", "level", "latitude", "longitude"], rng.standard_normal((nt, nlev, nlat, nlon)) * 10),
            "q": (["time", "level", "latitude", "longitude"], np.maximum(0, rng.standard_normal((nt, nlev, nlat, nlon)) * 0.002 + 0.005)),
        },
        coords={"time": times, "level": levels, "latitude": lat, "longitude": lon},
    )
    return ds


@pytest.fixture
def backend(request):
    """Parametrized backend fixture for cross-backend testing."""
    return request.param
