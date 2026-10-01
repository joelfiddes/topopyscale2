"""Precipitation downscaling kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - topopyscale2/core/rust_kernels/src/precipitation.rs
  - topopyscale2/core/jax_kernels/precipitation.py
Any changes here MUST be applied to all three backends in the same commit.
"""

import numpy as np

# Thermodynamic constants for the psychrometric wet-bulb solve.
CP_AIR = 1004.0       # specific heat of air at constant pressure [J/kg/K]
EPSILON = 0.622       # ratio of molar masses of water vapour to dry air
LV = 2.501e6          # latent heat of vaporisation [J/kg]
WET_BULB_ITERS = 20   # fixed Newton iterations (must match across backends)


def wet_bulb_temperature(
    temperature: np.ndarray,
    pressure: np.ndarray,
    specific_humidity: np.ndarray,
) -> np.ndarray:
    """Psychrometric (pressure-aware) wet-bulb temperature.

    Solves the psychrometer equation ``e = e_s(Tw) - gamma * (T - Tw)`` for the
    wet-bulb temperature ``Tw`` with a fixed number of Newton iterations. The
    psychrometric "constant" ``gamma = cp * p / (eps * Lv)`` depends on pressure,
    so the result is valid at the low pressures of high-elevation terrain (where
    a sea-level empirical formula would be biased).

    A falling hydrometeor equilibrates toward the wet-bulb temperature, so Tw is
    a better rain/snow predictor than air temperature, especially in dry air.

    SYNC: rust_kernels/src/precipitation.rs::wet_bulb_temperature
    SYNC: jax_kernels/precipitation.py::wet_bulb_temperature

    Parameters
    ----------
    temperature : array_like
        Air (dry-bulb) temperature [K].
    pressure : array_like
        Air pressure [Pa].
    specific_humidity : array_like
        Specific humidity [kg/kg].

    Returns
    -------
    np.ndarray
        Wet-bulb temperature [K], clamped <= air temperature.
    """
    t = np.asarray(temperature, dtype=np.float64)
    p = np.asarray(pressure, dtype=np.float64)
    q = np.asarray(specific_humidity, dtype=np.float64)

    # Actual vapour pressure from specific humidity [Pa].
    e = q * p / (EPSILON + (1.0 - EPSILON) * q)
    # Psychrometric "constant" gamma [Pa/K] (pressure-dependent).
    gamma = (CP_AIR * p) / (EPSILON * LV)

    tw = t.copy()  # initial guess: air temperature
    for _ in range(WET_BULB_ITERS):
        es = 611.2 * np.exp(17.67 * (tw - 273.15) / (tw - 29.65))
        des_dtw = es * (17.67 * (273.15 - 29.65)) / (tw - 29.65) ** 2
        f = es - gamma * (t - tw) - e
        fprime = des_dtw + gamma
        tw = tw - f / fprime

    return np.minimum(tw, t)


def elevation_gradient(
    p_source: np.ndarray,
    z_unit: np.ndarray,
    z_source: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    """Apply elevation-dependent precipitation gradient.

    SYNC: rust_kernels/src/precipitation.rs::elevation_gradient
    SYNC: jax_kernels/precipitation.py::elevation_gradient

    Parameters
    ----------
    p_source : array_like
        Source precipitation [m or kg/m2].
    z_unit : array_like
        Target elevation [m].
    z_source : array_like
        Source elevation [m].
    gradient : array_like
        Precipitation gradient [1/m] (e.g., 0.0003 for +3%/100m).

    Returns
    -------
    np.ndarray
        Corrected precipitation, clamped >= 0.
    """
    p_source = np.asarray(p_source, dtype=np.float64)
    z_unit = np.asarray(z_unit, dtype=np.float64)
    z_source = np.asarray(z_source, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)

    dz = z_unit - z_source
    p_corrected = p_source * (1.0 + gradient * dz)
    return np.maximum(p_corrected, 0.0)


def phase_partition(
    p_total: np.ndarray,
    temperature: np.ndarray,
    t_rain: np.ndarray,
    t_snow: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition precipitation into rainfall and snowfall.

    Linear interpolation between t_snow (all snow) and t_rain (all rain).

    SYNC: rust_kernels/src/precipitation.rs::phase_partition
    SYNC: jax_kernels/precipitation.py::phase_partition

    Parameters
    ----------
    p_total : array_like
        Total precipitation [m or kg/m2].
    temperature : array_like
        Air temperature [K].
    t_rain : array_like
        Temperature above which all precipitation is rain [K].
    t_snow : array_like
        Temperature below which all precipitation is snow [K].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rainfall, snowfall).
    """
    p_total = np.asarray(p_total, dtype=np.float64)
    temperature = np.asarray(temperature, dtype=np.float64)
    t_rain = np.asarray(t_rain, dtype=np.float64)
    t_snow = np.asarray(t_snow, dtype=np.float64)

    rain_fraction = np.where(
        temperature >= t_rain,
        1.0,
        np.where(
            temperature <= t_snow,
            0.0,
            (temperature - t_snow) / (t_rain - t_snow),
        ),
    )

    rainfall = p_total * rain_fraction
    snowfall = p_total * (1.0 - rain_fraction)
    return rainfall, snowfall
