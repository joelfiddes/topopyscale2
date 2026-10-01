"""Unit conversions for meteorological data.

This module standardizes ERA5 (and other NWP) data to consistent SI units
before downscaling. All conversions happen here, not in the downscaler.

ERA5 native units → Standard units:
- ssrd, strd: J/m² (1-hour snapshot) → W/m² (instantaneous)
- tp: m (accumulated per timestep) → mm (total per timestep)
- z: m²/s² (geopotential) → m (height)
- t2m, d2m, t: K (no change)
- sp: Pa (no change)
- u10, v10, u, v: m/s (no change)
- q: kg/kg (no change)
- r: % or 0-1 → 0-1 fractional

Backends ensure that tp is the total precipitation for the timestep
interval (e.g. 3-hour total for 3H data).  Radiation is kept as a
1-hour snapshot regardless of timestep — an acceptable approximation
for sub-daily forcing.

After this module runs, all variables are in standard units and the
downscaler can assume consistent inputs.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from topopyscale2.inputs.qc import qc_era5_surface

# Physical constants
G = 9.80665  # gravitational acceleration [m/s²]
SECONDS_PER_HOUR = 3600.0
MM_PER_M = 1000.0


def convert_era5_surface(ds: xr.Dataset) -> xr.Dataset:
    """Convert ERA5 surface variables to standard units.

    Radiation is treated as a 1-hour snapshot (always divided by 3600s)
    regardless of the data's temporal resolution.  This is an acceptable
    approximation for sub-daily forcing.

    Precipitation is converted from metres to mm.  Backends are responsible
    for ensuring tp is the total accumulation for the timestep interval
    (e.g. scaled by the step ratio for subsampled data).

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 surface dataset with native units.

    Returns
    -------
    xr.Dataset
        Dataset with standardized units.
    """
    ds = ds.copy()

    # Radiation: J/m² (1-hour snapshot) → W/m² instantaneous
    # Always use 3600s divisor: for subsampled data each value is still
    # a 1-hour accumulation, which is an acceptable approximation.
    if "ssrd" in ds:
        ds["ssrd"] = ds["ssrd"] / SECONDS_PER_HOUR
        ds["ssrd"].attrs["units"] = "W m-2"
        ds["ssrd"].attrs["long_name"] = "Surface solar radiation downwards"

    if "strd" in ds:
        ds["strd"] = ds["strd"] / SECONDS_PER_HOUR
        ds["strd"].attrs["units"] = "W m-2"
        ds["strd"].attrs["long_name"] = "Surface thermal radiation downwards"

    if "tisr" in ds:
        ds["tisr"] = ds["tisr"] / SECONDS_PER_HOUR
        ds["tisr"].attrs["units"] = "W m-2"
        ds["tisr"].attrs["long_name"] = "TOA incident solar radiation"

    # Precipitation: m accumulated per timestep → mm per timestep
    # Backends ensure tp is the total for the timestep interval.
    if "tp" in ds:
        ds["tp"] = ds["tp"] * MM_PER_M
        ds["tp"].attrs["units"] = "mm"
        ds["tp"].attrs["long_name"] = "Total precipitation per timestep"

    # Geopotential: m²/s² → m height
    if "z" in ds:
        ds["z"] = ds["z"] / G
        ds["z"].attrs["units"] = "m"
        ds["z"].attrs["long_name"] = "Surface geopotential height"

    if "z_surf" in ds:
        ds["z_surf"] = ds["z_surf"] / G
        ds["z_surf"].attrs["units"] = "m"
        ds["z_surf"].attrs["long_name"] = "Surface geopotential height"

    # Temperature: ensure attrs
    for var in ["t2m", "d2m"]:
        if var in ds:
            ds[var].attrs["units"] = "K"

    # Pressure: ensure attrs
    if "sp" in ds:
        ds["sp"].attrs["units"] = "Pa"

    # Wind: ensure attrs
    for var in ["u10", "v10"]:
        if var in ds:
            ds[var].attrs["units"] = "m s-1"

    return ds


def convert_era5_pressure(ds: xr.Dataset) -> xr.Dataset:
    """Convert ERA5 pressure level variables to standard units.

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 pressure level dataset with native units.

    Returns
    -------
    xr.Dataset
        Dataset with standardized units.
    """
    ds = ds.copy()

    # Geopotential: m²/s² → m height
    if "z" in ds:
        ds["z"] = ds["z"] / G
        ds["z"].attrs["units"] = "m"
        ds["z"].attrs["long_name"] = "Geopotential height"

    # Temperature: ensure attrs
    if "t" in ds:
        ds["t"].attrs["units"] = "K"

    # Wind: ensure attrs
    for var in ["u", "v"]:
        if var in ds:
            ds[var].attrs["units"] = "m s-1"

    # Specific humidity: ensure attrs
    if "q" in ds:
        ds["q"].attrs["units"] = "kg kg-1"

    # Relative humidity: normalize to 0-1 if in %
    if "r" in ds:
        if ds["r"].max() > 1.1:  # Likely in %
            ds["r"] = ds["r"] / 100.0
        ds["r"].attrs["units"] = "1"
        ds["r"].attrs["long_name"] = "Relative humidity"

    # Derive specific humidity from relative humidity when q is missing
    # (e.g. Open-Meteo IFS provides r but not q)
    if "r" in ds and "t" in ds and "q" not in ds and "level" in ds.dims:
        t = ds["t"]  # K
        r = ds["r"]  # 0-1
        # Pressure from level coordinate (hPa → Pa)
        p = ds["level"].astype("float64") * 100.0
        # Bolton (1980) saturation vapor pressure
        e_sat = 611.2 * np.exp(17.67 * (t - 273.15) / (t - 29.65))
        e = r * e_sat
        ds["q"] = 0.622 * e / (p - 0.378 * e)
        ds["q"].attrs["units"] = "kg kg-1"
        ds["q"].attrs["long_name"] = "Specific humidity (derived from r)"

    return ds


def convert_era5(
    ds_surface: xr.Dataset,
    ds_pressure: xr.Dataset,
    *,
    run_qc: bool = True,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Convert both ERA5 surface and pressure datasets to standard units.

    This is the main entry point for unit conversion.  When ``run_qc``
    is True (default), surface data is first passed through
    :func:`qc_era5_surface` to detect and interpolate corrupted
    single-timestep spikes before converting units.

    Parameters
    ----------
    ds_surface : xr.Dataset
        ERA5 surface dataset.
    ds_pressure : xr.Dataset
        ERA5 pressure level dataset.
    run_qc : bool
        If True, run input QC on surface data before conversion.

    Returns
    -------
    tuple[xr.Dataset, xr.Dataset]
        (surface, pressure) datasets with standardized units.
    """
    if run_qc:
        ds_surface = qc_era5_surface(ds_surface)
    return (
        convert_era5_surface(ds_surface),
        convert_era5_pressure(ds_pressure),
    )


def detect_time_step(ds: xr.Dataset) -> float:
    """Detect time step in hours from dataset.

    Uses the most common time delta (mode) to handle gaps in the data.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with time coordinate.

    Returns
    -------
    float
        Time step in hours.
    """
    if "time" not in ds.dims or ds.sizes["time"] < 2:
        return 1.0  # Default to hourly

    # Use mode of first N deltas to be robust against data gaps
    n_sample = min(48, ds.sizes["time"])
    deltas = np.diff(ds.time.values[:n_sample])
    hours = deltas / np.timedelta64(1, "h")
    # Most common delta = nominal time step
    unique, counts = np.unique(hours, return_counts=True)
    return float(unique[np.argmax(counts)])
