"""Temperature downscaling kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - topopyscale2/core/rust_kernels/src/temperature.rs
  - topopyscale2/core/jax_kernels/temperature.py
Any changes here MUST be applied to all three backends in the same commit.
"""

import numpy as np


def simple_lapse_rate(
    t_surface: np.ndarray,
    z_surface: np.ndarray,
    z_target: float,
    lapse_rate: float = 0.0065,
) -> np.ndarray:
    """Apply simple lapse rate correction from surface temperature.

    This is the "simple" downscaling mode that only requires ERA5-Land
    surface data (t2m). Temperature decreases with elevation at a fixed rate.

    SYNC: rust_kernels/src/temperature.rs::simple_lapse_rate
    SYNC: jax_kernels/temperature.py::simple_lapse_rate

    Parameters
    ----------
    t_surface : array_like
        Surface temperature [K], e.g., ERA5 t2m.
    z_surface : array_like
        Surface elevation [m], e.g., ERA5 grid cell elevation.
    z_target : float
        Target elevation [m].
    lapse_rate : float, optional
        Environmental lapse rate [K/m], default 0.0065 (6.5 K/km).
        This is the rate of temperature DECREASE with altitude.

    Returns
    -------
    np.ndarray
        Temperature at target elevation [K].

    Notes
    -----
    Formula: T_target = T_surface - lapse_rate * (z_target - z_surface)

    When z_target > z_surface (going uphill), temperature decreases.
    When z_target < z_surface (going downhill), temperature increases.

    The default 6.5 K/km is the average environmental lapse rate in the
    troposphere. Actual lapse rates vary:
    - Dry adiabatic: ~9.8 K/km
    - Moist adiabatic: ~5-6 K/km
    - Inversions: negative (temperature increases with altitude)
    """
    t_surface = np.asarray(t_surface, dtype=np.float64)
    z_surface = np.asarray(z_surface, dtype=np.float64)

    dz = z_target - z_surface
    return t_surface - lapse_rate * dz


def lapse_rate_correction(
    t_source: np.ndarray,
    gamma: np.ndarray,
    z_unit: np.ndarray,
    z_source: np.ndarray,
) -> np.ndarray:
    """Apply lapse rate correction to temperature.

    SYNC: rust_kernels/src/temperature.rs::lapse_rate_correction
    SYNC: jax_kernels/temperature.py::lapse_rate_correction

    Parameters
    ----------
    t_source : array_like
        Source temperature [K].
    gamma : array_like
        Lapse rate [K/m] (typically negative, e.g., -0.0065).
    z_unit : array_like
        Target elevation [m].
    z_source : array_like
        Source elevation [m].

    Returns
    -------
    np.ndarray
        Corrected temperature [K].
    """
    t_source = np.asarray(t_source, dtype=np.float64)
    gamma = np.asarray(gamma, dtype=np.float64)
    z_unit = np.asarray(z_unit, dtype=np.float64)
    z_source = np.asarray(z_source, dtype=np.float64)

    return t_source + gamma * (z_unit - z_source)
