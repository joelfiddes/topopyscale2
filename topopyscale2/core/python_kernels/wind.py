"""Wind downscaling kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - topopyscale2/core/rust_kernels/src/wind.rs (actually src/wind.rs)
  - topopyscale2/core/jax_kernels/wind.py
Any changes here MUST be applied to all three backends in the same commit.
"""

import numpy as np


def log_profile_correction(
    u_source: np.ndarray,
    z_source: np.ndarray,
    z_target: np.ndarray,
    z0_source: np.ndarray,
    z0_target: np.ndarray,
    d_source: np.ndarray,
    d_target: np.ndarray,
) -> np.ndarray:
    """Apply logarithmic wind profile correction.

    Adjusts wind speed from source measurement height to target height,
    accounting for different roughness lengths and displacement heights.

    Formula:
        U_target = U_source * ln((z_target - d_target) / z0_target) / ln((z_source - d_source) / z0_source)

    Guard: Returns 0 when effective height (z - d) is at or below z0, which
    would result in invalid logarithm arguments or non-physical wind speeds.

    SYNC: rust_kernels/src/wind.rs::log_profile_correction
    SYNC: jax_kernels/wind.py::log_profile_correction

    Parameters
    ----------
    u_source : array_like
        Source wind speed [m/s].
    z_source : array_like
        Source measurement height above ground [m].
    z_target : array_like
        Target height above ground [m].
    z0_source : array_like
        Roughness length at source [m].
    z0_target : array_like
        Roughness length at target [m].
    d_source : array_like
        Zero-plane displacement height at source [m].
    d_target : array_like
        Zero-plane displacement height at target [m].

    Returns
    -------
    np.ndarray
        Corrected wind speed at target height [m/s].
    """
    u_source = np.asarray(u_source, dtype=np.float64)
    z_source = np.asarray(z_source, dtype=np.float64)
    z_target = np.asarray(z_target, dtype=np.float64)
    z0_source = np.asarray(z0_source, dtype=np.float64)
    z0_target = np.asarray(z0_target, dtype=np.float64)
    d_source = np.asarray(d_source, dtype=np.float64)
    d_target = np.asarray(d_target, dtype=np.float64)

    # Effective heights above displacement
    h_source = z_source - d_source
    h_target = z_target - d_target

    # Guard: clamp to 0 when effective height <= roughness length
    # This prevents invalid log arguments and non-physical results
    valid_source = h_source > z0_source
    valid_target = h_target > z0_target
    valid = valid_source & valid_target

    # Compute logarithmic ratio where valid
    # Use np.where to avoid log of invalid values
    ln_source = np.where(valid_source, np.log(h_source / z0_source), 1.0)
    ln_target = np.where(valid_target, np.log(h_target / z0_target), 1.0)

    # Apply correction where valid, otherwise return 0
    result = np.where(valid, u_source * ln_target / ln_source, 0.0)

    return result


def winstral_wind_correction(
    u_log: np.ndarray,
    sx: np.ndarray,
    sx_scale: float = 0.5,
    sx_ref: float = 15.0,
) -> np.ndarray:
    """Apply Winstral Sx-based wind exposure/sheltering correction.

    Modifies wind speed based on terrain exposure index (Sx). Positive Sx
    indicates exposed terrain (wind enhancement), negative Sx indicates
    sheltered terrain (wind reduction).

    Formula:
        U_corrected = U_log * (1 + sx_scale * tanh(Sx / sx_ref))

    The tanh function naturally bounds the correction factor, preventing
    extreme modifications for very large |Sx| values.

    Reference:
        Winstral, A., Elder, K., & Davis, R. E. (2002). Spatial snow modeling
        of wind-redistributed snow using terrain-based parameters.
        Journal of Hydrometeorology, 3(5), 524-538.

    SYNC: rust_kernels/src/wind.rs::winstral_wind_correction
    SYNC: jax_kernels/wind.py::winstral_wind_correction

    Parameters
    ----------
    u_log : array_like
        Wind speed after log-profile correction [m/s].
    sx : array_like
        Winstral Sx terrain parameter [degrees]. Positive = exposed,
        negative = sheltered.
    sx_scale : float, optional
        Scaling factor for the Sx effect. Default is 0.5, meaning
        maximum enhancement/reduction is +/- 50%.
    sx_ref : float, optional
        Reference Sx value for normalization [degrees]. Default is 15.0.
        Larger values make the correction less sensitive to Sx.

    Returns
    -------
    np.ndarray
        Corrected wind speed [m/s].
    """
    u_log = np.asarray(u_log, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)

    # Correction factor: (1 + sx_scale * tanh(Sx / sx_ref))
    # tanh saturates at +/- 1, so factor ranges from (1 - sx_scale) to (1 + sx_scale)
    correction_factor = 1.0 + sx_scale * np.tanh(sx / sx_ref)

    return u_log * correction_factor
