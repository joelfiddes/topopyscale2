"""Pressure-level interpolation kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - topopyscale2/core/rust_kernels/src/interpolation.rs
  - topopyscale2/core/jax_kernels/interpolation.py
Any changes here MUST be applied to all three backends in the same commit.

This module implements the TopoPyScale vertical interpolation algorithm:
for each target elevation, find the two pressure levels that bracket it
(based on geopotential height) and interpolate using inverse distance weighting.
"""

import numpy as np


def interpolate_pressure_levels(
    values: np.ndarray,
    z_levels: np.ndarray,
    z_target: float,
) -> np.ndarray:
    """Interpolate values from pressure levels to target elevation.

    For each timestep, finds the pressure levels above and below the target
    elevation and performs inverse-distance-weighted interpolation.

    SYNC: rust_kernels/src/interpolation.rs::interpolate_pressure_levels
    SYNC: jax_kernels/interpolation.py::interpolate_pressure_levels

    Parameters
    ----------
    values : np.ndarray
        Values at pressure levels, shape (n_time, n_levels).
    z_levels : np.ndarray
        Geopotential heights at pressure levels [m], shape (n_time, n_levels).
        Must be in the same order as values (typically ascending with level index
        means descending pressure, i.e., increasing altitude).
    z_target : float
        Target elevation [m].

    Returns
    -------
    np.ndarray
        Interpolated values at target elevation, shape (n_time,).

    Notes
    -----
    When target is outside the range of pressure levels:
    - If below all levels: uses the lowest level (highest pressure)
    - If above all levels: uses the highest level (lowest pressure)

    The algorithm follows TopoPyScale's approach:
    - Find levels immediately above and below target
    - Compute inverse-distance weights
    - Weighted average of the two bracketing values
    """
    values = np.asarray(values, dtype=np.float64)
    z_levels = np.asarray(z_levels, dtype=np.float64)

    # Handle 1D case (single timestep)
    if values.ndim == 1:
        values = values.reshape(1, -1)
        z_levels = z_levels.reshape(1, -1)
        squeeze_output = True
    else:
        squeeze_output = False

    n_time, n_levels = values.shape
    result = np.zeros(n_time, dtype=np.float64)

    for t in range(n_time):
        z = z_levels[t, :]
        v = values[t, :]

        # Find levels above and below target
        above_mask = z > z_target
        below_mask = z < z_target

        if above_mask.any() and below_mask.any():
            # Target is bracketed by pressure levels - interpolate
            # Find closest level above (smallest positive difference)
            z_above = np.where(above_mask, z, np.inf)
            i_top = np.argmin(z_above - z_target)

            # Find closest level below (smallest negative difference)
            z_below = np.where(below_mask, z, -np.inf)
            i_bot = np.argmax(z_below - z_target)

            z_top = z[i_top]
            z_bot = z[i_bot]
            v_top = v[i_top]
            v_bot = v[i_bot]

            # Inverse distance weighting
            d_top = z_top - z_target
            d_bot = z_target - z_bot
            total_dist = d_top + d_bot

            if total_dist > 0:
                # Weight inversely proportional to distance
                # Closer level gets higher weight
                w_top = d_bot / total_dist
                w_bot = d_top / total_dist
                result[t] = w_bot * v_bot + w_top * v_top
            else:
                # Degenerate case: target exactly at a level
                result[t] = v_top

        elif above_mask.any():
            # Target is below all levels - use lowest level
            i_lowest = np.argmin(z)
            result[t] = v[i_lowest]

        elif below_mask.any():
            # Target is above all levels - use highest level
            i_highest = np.argmax(z)
            result[t] = v[i_highest]

        else:
            # All levels are exactly at target (unlikely)
            result[t] = v[0]

    if squeeze_output:
        return result[0]
    return result


def interpolate_wind_components(
    u_levels: np.ndarray,
    v_levels: np.ndarray,
    z_levels: np.ndarray,
    z_target: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate u and v wind components from pressure levels.

    SYNC: rust_kernels/src/interpolation.rs::interpolate_wind_components
    SYNC: jax_kernels/interpolation.py::interpolate_wind_components

    Parameters
    ----------
    u_levels : np.ndarray
        Eastward wind component at pressure levels [m/s], shape (n_time, n_levels).
    v_levels : np.ndarray
        Northward wind component at pressure levels [m/s], shape (n_time, n_levels).
    z_levels : np.ndarray
        Geopotential heights at pressure levels [m], shape (n_time, n_levels).
    z_target : float
        Target elevation [m].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (u_target, v_target) interpolated wind components [m/s].
    """
    u_target = interpolate_pressure_levels(u_levels, z_levels, z_target)
    v_target = interpolate_pressure_levels(v_levels, z_levels, z_target)
    return u_target, v_target


def compute_wind_speed_direction(
    u: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute wind speed and direction from u, v components.

    SYNC: rust_kernels/src/interpolation.rs::compute_wind_speed_direction
    SYNC: jax_kernels/interpolation.py::compute_wind_speed_direction

    Parameters
    ----------
    u : np.ndarray
        Eastward wind component [m/s].
    v : np.ndarray
        Northward wind component [m/s].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (wind_speed, wind_direction) where direction is meteorological convention
        (direction wind is FROM, degrees clockwise from north, 0-360).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    wind_speed = np.sqrt(u**2 + v**2)

    # Meteorological convention: direction FROM which wind blows
    # atan2(-u, -v) gives angle from north
    wind_direction = np.degrees(np.arctan2(-u, -v))
    wind_direction = np.mod(wind_direction, 360.0)

    return wind_speed, wind_direction


def interpolate_humidity(
    q_levels: np.ndarray,
    z_levels: np.ndarray,
    z_target: float,
    t_target: np.ndarray,
    p_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate specific humidity and compute relative humidity at target.

    SYNC: rust_kernels/src/interpolation.rs::interpolate_humidity
    SYNC: jax_kernels/interpolation.py::interpolate_humidity

    Parameters
    ----------
    q_levels : np.ndarray
        Specific humidity at pressure levels [kg/kg], shape (n_time, n_levels).
    z_levels : np.ndarray
        Geopotential heights at pressure levels [m], shape (n_time, n_levels).
    z_target : float
        Target elevation [m].
    t_target : np.ndarray
        Temperature at target elevation [K], shape (n_time,).
    p_target : np.ndarray
        Pressure at target elevation [Pa], shape (n_time,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (q_target, rh_target) specific humidity [kg/kg] and relative humidity [0-1].
    """
    # Interpolate specific humidity
    q_target = interpolate_pressure_levels(q_levels, z_levels, z_target)

    # Compute relative humidity at target using Bolton (1980)
    t_target = np.asarray(t_target, dtype=np.float64)
    p_target = np.asarray(p_target, dtype=np.float64)

    # Mixing ratio from specific humidity
    mr = q_target / (1.0 - q_target)

    # Actual vapor pressure [Pa]
    e = mr * p_target / (0.62197 + mr)

    # Bolton (1980) saturation vapor pressure [Pa]
    e_sat = 611.2 * np.exp(17.67 * (t_target - 273.15) / (t_target - 29.65))

    # Relative humidity clamped to [0, 1]
    rh_target = np.clip(e / e_sat, 0.0, 1.0)

    return q_target, rh_target
