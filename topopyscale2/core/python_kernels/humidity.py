"""Humidity downscaling kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - topopyscale2/core/rust_kernels/src/humidity.rs
  - topopyscale2/core/jax_kernels/humidity.py
Any changes here MUST be applied to all three backends in the same commit.
"""

import numpy as np


def adjust_humidity(
    q_source: np.ndarray,
    t_unit: np.ndarray,
    p_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Adjust humidity for target elevation.

    Specific humidity is conserved. Relative humidity is recomputed from
    the adjusted temperature and pressure using Bolton (1980) saturation
    vapor pressure.

    Note: not called by the Downscaler pipeline (which derives humidity
    inline per mode); kept as public kernel API for downstream use and
    exercised by the cross-backend tests.

    SYNC: rust_kernels/src/humidity.rs::adjust_humidity
    SYNC: jax_kernels/humidity.py::adjust_humidity

    Parameters
    ----------
    q_source : array_like
        Source specific humidity [kg/kg].
    t_unit : array_like
        Target temperature [K].
    p_unit : array_like
        Target surface pressure [Pa].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (q_adjusted, rh) where q_adjusted = q_source and rh in [0, 1].
    """
    q_source = np.asarray(q_source, dtype=np.float64)
    t_unit = np.asarray(t_unit, dtype=np.float64)
    p_unit = np.asarray(p_unit, dtype=np.float64)

    q_adjusted = q_source.copy()

    # Mixing ratio
    mr = q_source / (1.0 - q_source)

    # Actual vapor pressure [Pa]
    e = mr * p_unit / (0.62197 + mr)

    # Bolton (1980) saturation vapor pressure [Pa]
    e_sat = 611.2 * np.exp(17.67 * (t_unit - 273.15) / (t_unit - 29.65))

    # Relative humidity clamped to [0, 1]
    rh = np.clip(e / e_sat, 0.0, 1.0)

    return q_adjusted, rh
