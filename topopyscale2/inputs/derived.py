"""Derived meteorological quantities computed from raw forcing data.

Vectorized implementations — all functions use numpy broadcasting
over (time, latitude, longitude) grids. No Python loops.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

log = logging.getLogger(__name__)


def compute_lapse_rate(ds: xr.Dataset) -> xr.DataArray:
    """Compute environmental lapse rate from pressure level data.

    For each grid cell and timestep, fits linear T vs Z from pressure
    levels using vectorized least-squares regression across all grid
    cells simultaneously.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'z' (geopotential or height) and 't' (temperature)
        on pressure levels.

    Returns
    -------
    xr.DataArray
        Lapse rate gamma [K/m] as DataArray(time, latitude, longitude).
    """
    g = 9.80665

    if "z" not in ds or "t" not in ds:
        raise ValueError("Dataset must contain 'z' (geopotential) and 't' (temperature)")

    # Get raw z values and check units
    z_raw = ds["z"]
    t = ds["t"]

    # Check if z is geopotential (J/kg, ~10000-55000) or height (m, ~1000-5500)
    z_sample = float(z_raw.isel(time=0, latitude=0, longitude=0).max())
    if z_sample > 10000:
        log.info("compute_lapse_rate: z is geopotential (max=%.0f J/kg), converting to meters", z_sample)
        z_m = z_raw / g
    else:
        log.info("compute_lapse_rate: z is already in meters (max=%.0f m)", z_sample)
        z_m = z_raw

    # Transpose to ensure (time, level, latitude, longitude) order
    target_dims = ["time", "level", "latitude", "longitude"]
    if list(z_m.dims) != target_dims:
        z_m = z_m.transpose(*target_dims)
        t = t.transpose(*target_dims)

    # shape: (nt, nlev, nlat, nlon)
    z_vals = z_m.values
    t_vals = t.values

    gamma = _vectorized_lapse_rate(z_vals, t_vals)

    log.info(
        "compute_lapse_rate: result range [%.2f, %.2f] K/km, mean=%.2f K/km",
        gamma.min() * 1000,
        gamma.max() * 1000,
        gamma.mean() * 1000,
    )

    return xr.DataArray(
        gamma,
        dims=["time", "latitude", "longitude"],
        coords={
            "time": ds.time,
            "latitude": ds.latitude,
            "longitude": ds.longitude,
        },
        attrs={"units": "K/m", "long_name": "Temperature lapse rate"},
    )


def _vectorized_lapse_rate(
    z_vals: np.ndarray, t_vals: np.ndarray, fallback: float = -0.0065
) -> np.ndarray:
    """Vectorized linear regression of T vs Z across pressure levels.

    Computes slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x²) - sum(x)²)
    for all (time, lat, lon) cells simultaneously.

    Parameters
    ----------
    z_vals : ndarray, shape (nt, nlev, nlat, nlon)
        Height values on pressure levels [m].
    t_vals : ndarray, shape (nt, nlev, nlat, nlon)
        Temperature values on pressure levels [K].
    fallback : float
        Lapse rate fallback for cells with <2 valid levels [K/m].

    Returns
    -------
    ndarray, shape (nt, nlat, nlon)
        Lapse rate gamma [K/m].
    """
    # Mask invalid (NaN) entries
    valid = ~(np.isnan(z_vals) | np.isnan(t_vals))  # (nt, nlev, nlat, nlon)
    n_valid = valid.sum(axis=1)  # (nt, nlat, nlon)

    # Zero out NaN entries for safe summation
    z_clean = np.where(valid, z_vals, 0.0)
    t_clean = np.where(valid, t_vals, 0.0)

    # Sums along the level axis
    sum_z = z_clean.sum(axis=1)
    sum_t = t_clean.sum(axis=1)
    sum_zt = (z_clean * t_clean).sum(axis=1)
    sum_z2 = (z_clean ** 2).sum(axis=1)

    # Denominator of linear regression slope
    denom = n_valid * sum_z2 - sum_z ** 2

    # Compute slope where we have >=2 valid levels and non-degenerate denominator
    enough_data = (n_valid >= 2) & (np.abs(denom) > 1e-12)
    gamma = np.where(
        enough_data,
        (n_valid * sum_zt - sum_z * sum_t) / np.where(enough_data, denom, 1.0),
        fallback,
    )

    return gamma


def compute_solar_geometry(
    time: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute solar elevation and azimuth angles (vectorized).

    Uses numpy broadcasting across (time, lat, lon) — no Python loops.

    Parameters
    ----------
    time : array of datetime64
    lat : array of latitudes [degrees]
    lon : array of longitudes [degrees]

    Returns
    -------
    tuple of (solar_elevation, solar_azimuth) ndarrays in degrees,
        each with shape (nt, nlat, nlon).
    """
    times = pd.DatetimeIndex(time)
    doy = times.dayofyear.values.astype(np.float64)  # (nt,)
    h = times.hour.values + times.minute.values / 60.0  # (nt,)

    # Solar declination — shape (nt,)
    b = np.radians(360.0 / 365.0 * (doy - 81))
    decl_rad = np.radians(23.45 * np.sin(b))

    # Equation of time [hours] — shape (nt,)
    eot = (9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)) / 60.0

    # Broadcast to (nt, nlat, nlon)
    # time axis: [:, None, None]  lat axis: [None, :, None]  lon axis: [None, None, :]
    lat_rad = np.radians(lat)[None, :, None]   # (1, nlat, 1)
    lon_1d = lon[None, None, :]                 # (1, 1, nlon)
    h_3d = h[:, None, None]                     # (nt, 1, 1)
    eot_3d = eot[:, None, None]                 # (nt, 1, 1)
    decl_3d = decl_rad[:, None, None]           # (nt, 1, 1)

    # Local solar time and hour angle — (nt, 1, nlon) broadcasts to (nt, nlat, nlon)
    lst = h_3d + lon_1d / 15.0 + eot_3d
    hour_angle = np.radians(15.0 * (lst - 12.0))

    # Solar elevation — full (nt, nlat, nlon) via broadcasting
    sin_elev = (
        np.sin(lat_rad) * np.sin(decl_3d)
        + np.cos(lat_rad) * np.cos(decl_3d) * np.cos(hour_angle)
    )
    sin_elev = np.clip(sin_elev, -1.0, 1.0)
    solar_elev = np.degrees(np.arcsin(sin_elev))

    # Solar azimuth
    cos_elev = np.cos(np.radians(solar_elev))
    above_horizon = cos_elev > 0.001

    cos_az = np.where(
        above_horizon,
        (np.sin(decl_3d) - np.sin(lat_rad) * sin_elev)
        / np.where(above_horizon, np.cos(lat_rad) * cos_elev, 1.0),
        0.0,
    )
    cos_az = np.clip(cos_az, -1.0, 1.0)
    solar_az = np.degrees(np.arccos(cos_az))

    # Afternoon: az = 360 - az
    solar_az = np.where(hour_angle > 0, 360.0 - solar_az, solar_az)
    # Below horizon: az = 0
    solar_az = np.where(above_horizon, solar_az, 0.0)

    return solar_elev, solar_az


def compute_solar_elevation(ds: xr.Dataset) -> xr.DataArray:
    """Compute solar elevation angles from time and coordinates.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with time, latitude, longitude coordinates.

    Returns
    -------
    xr.DataArray
        Solar elevation angle in degrees.
    """
    time = ds.time.values
    lat = ds.latitude.values
    lon = ds.longitude.values

    solar_elev, _ = compute_solar_geometry(time, lat, lon)

    return xr.DataArray(
        solar_elev,
        dims=["time", "latitude", "longitude"],
        coords={"time": time, "latitude": lat, "longitude": lon},
        attrs={"units": "degrees", "long_name": "Solar elevation angle"},
    )


def compute_clearness_index(ds: xr.Dataset) -> xr.DataArray:
    """Compute clearness index kt = ssrd / sw_toa.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 'ssrd' (surface solar radiation) and 'solar_elevation'.

    Returns
    -------
    xr.DataArray
        Clearness index kt, clamped to [0, 1].
    """
    SOLAR_CONSTANT = 1361.0  # W/m2

    if "solar_elevation" not in ds:
        raise ValueError("Dataset must contain 'solar_elevation'. Run compute_solar_elevation first.")

    if "ssrd" not in ds:
        raise ValueError("Dataset must contain 'ssrd' (surface solar radiation downwards)")

    # xarray ops so a member (or any extra) dimension on ssrd broadcasts cleanly
    # against the member-less solar geometry (dims inferred, not hardcoded).
    sin_elev = np.sin(np.radians(ds["solar_elevation"]))
    sw_toa = SOLAR_CONSTANT * np.maximum(sin_elev, 0.0)

    kt = xr.where(sw_toa > 1.0, ds["ssrd"] / sw_toa, 0.0)
    kt = kt.clip(0.0, 1.0)
    kt.attrs = {"units": "-", "long_name": "Clearness index"}
    return kt


def synthesize_longwave(ds: xr.Dataset) -> xr.DataArray:
    """Synthesize surface thermal radiation downwards (strd) from t2m, d2m, ssrd.

    Uses the Brutsaert (1975) clear-sky emissivity model with the
    Konzelmann et al. (1994) all-sky correction.  Cloud fraction is
    estimated from the clearness index (kt = ssrd / TOA) during daytime
    and filled with the daily mean for nighttime hours.

    This allows forecast datasets that lack strd (e.g. Open-Meteo IFS)
    to participate in the radiation downscaling group.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with 't2m', 'd2m', 'ssrd', plus 'latitude', 'longitude',
        and 'time' coordinates.

    Returns
    -------
    xr.DataArray
        Synthesized strd [W/m2] with the same dims/coords as t2m.
    """
    SIGMA = 5.670374419e-8  # Stefan-Boltzmann constant
    SOLAR_CONSTANT = 1361.0

    t2m = ds["t2m"].values  # K
    d2m = ds["d2m"].values  # K
    ssrd = ds["ssrd"].values  # W/m2

    # Vapor pressure from dewpoint (Bolton 1980)
    e_a = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))

    # Brutsaert (1975) clear-sky emissivity
    eps_cs = 0.23 + 0.43 * np.maximum(e_a / t2m, 0.0) ** (1.0 / 5.7)

    # Cloud fraction from clearness index
    time = ds.time.values
    lat = ds.latitude.values
    lon = ds.longitude.values
    solar_elev, _ = compute_solar_geometry(time, lat, lon)
    sin_elev = np.sin(np.radians(solar_elev))
    sw_toa = SOLAR_CONSTANT * np.maximum(sin_elev, 0.0)

    kt = np.where(sw_toa > 1.0, ssrd / sw_toa, np.nan)
    kt = np.clip(kt, 0.0, 1.0)
    cloud_frac = 1.0 - kt  # NaN at nighttime

    # Fill nighttime cloud fraction with daily mean
    time_pd = pd.DatetimeIndex(time)
    day_labels = time_pd.date
    # Compute daily mean cloud fraction (daytime only)
    n_time = cloud_frac.shape[0]
    spatial_shape = cloud_frac.shape[1:]
    flat_cf = cloud_frac.reshape(n_time, -1)
    unique_days = np.unique(day_labels)
    daily_mean = np.full_like(flat_cf, 0.5)  # fallback
    for day in unique_days:
        mask = day_labels == day
        day_vals = flat_cf[mask]
        with np.errstate(all="ignore"):
            day_avg = np.nanmean(day_vals, axis=0)
        # If entire day is NaN (polar night), use 0.5
        day_avg = np.where(np.isnan(day_avg), 0.5, day_avg)
        daily_mean[mask] = day_avg
    cloud_frac_filled = np.where(np.isnan(flat_cf), daily_mean, flat_cf)
    cloud_frac_filled = cloud_frac_filled.reshape(cloud_frac.shape)

    # Konzelmann et al. (1994) all-sky emissivity
    cf2 = cloud_frac_filled ** 2
    eps_all = eps_cs * (1.0 - cf2) + cf2

    strd = eps_all * SIGMA * t2m ** 4

    log.info(
        "Synthesized strd from t2m/d2m/ssrd: %.1f to %.1f W/m2",
        float(np.nanmin(strd)),
        float(np.nanmax(strd)),
    )

    return xr.DataArray(
        strd,
        dims=ds["t2m"].dims,
        coords=ds["t2m"].coords,
        attrs={
            "units": "W m-2",
            "long_name": "Surface thermal radiation downwards (synthesized)",
        },
    )
