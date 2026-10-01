"""Radiation downscaling kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - topopyscale2/core/rust_kernels/src/radiation.rs
  - topopyscale2/core/jax_kernels/radiation.py
Any changes here MUST be applied to all three backends in the same commit.
"""

import numpy as np

# Stefan-Boltzmann constant [W m^-2 K^-4]
SIGMA = 5.670374419e-8

# Minimum sin(solar_elevation) for slope correction denominator.
# sin(15°) ≈ 0.259; below this the cos_incidence/sin_elev ratio becomes
# numerically unstable, producing unphysical SW spikes on steep slopes.
# See TopoPyScale PR #132 (Beria et al.).
MU0_MIN = 0.259

# Solar constant [W/m2] — absolute upper bound for direct beam on any surface.
SOLAR_CONSTANT = 1361.0


def partition_shortwave(
    sw_total: np.ndarray,
    solar_elevation: np.ndarray,
    clearness_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Partition total shortwave into direct and diffuse components.

    Uses Erbs et al. (1982) diffuse fraction model.

    SYNC: rust_kernels/src/radiation.rs::partition_shortwave
    SYNC: jax_kernels/radiation.py::partition_shortwave

    Parameters
    ----------
    sw_total : array_like
        Total incoming shortwave radiation [W/m2].
    solar_elevation : array_like
        Solar elevation angle [degrees].
    clearness_index : array_like
        Clearness index kt [-], typically [0, 1].

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (sw_direct, sw_diffuse) [W/m2].
    """
    sw_total = np.asarray(sw_total, dtype=np.float64)
    solar_elevation = np.asarray(solar_elevation, dtype=np.float64)
    clearness_index = np.asarray(clearness_index, dtype=np.float64)

    kt = np.clip(clearness_index, 0.0, 1.0)

    # Erbs et al. (1982) diffuse fraction
    kd = np.where(
        kt <= 0.22,
        1.0 - 0.09 * kt,
        np.where(
            kt <= 0.80,
            0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4,
            0.165,
        ),
    )

    sw_diffuse = kd * sw_total
    sw_direct = sw_total - sw_diffuse

    # Night: all radiation is diffuse
    night = solar_elevation <= 0.0
    sw_direct = np.where(night, 0.0, sw_direct)
    sw_diffuse = np.where(night, sw_total, sw_diffuse)

    return sw_direct, sw_diffuse


def slope_correction(
    sw_direct: np.ndarray,
    solar_elevation: np.ndarray,
    solar_azimuth: np.ndarray,
    slope: np.ndarray,
    aspect: np.ndarray,
) -> np.ndarray:
    """Apply slope/aspect correction to direct shortwave radiation.

    SYNC: rust_kernels/src/radiation.rs::slope_correction
    SYNC: jax_kernels/radiation.py::slope_correction

    Parameters
    ----------
    sw_direct : array_like
        Direct shortwave on horizontal surface [W/m2].
    solar_elevation : array_like
        Solar elevation angle [degrees].
    solar_azimuth : array_like
        Solar azimuth angle [degrees], 0=N, 90=E, 180=S, 270=W.
    slope : array_like
        Surface slope [degrees].
    aspect : array_like
        Surface aspect [degrees], 0=N, 90=E, 180=S, 270=W.

    Returns
    -------
    np.ndarray
        Slope-corrected direct shortwave [W/m2].
    """
    sw_direct = np.asarray(sw_direct, dtype=np.float64)
    solar_elevation = np.asarray(solar_elevation, dtype=np.float64)
    solar_azimuth = np.asarray(solar_azimuth, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    aspect = np.asarray(aspect, dtype=np.float64)

    deg2rad = np.pi / 180.0
    solar_elev_rad = solar_elevation * deg2rad
    solar_az_rad = solar_azimuth * deg2rad
    slope_rad = slope * deg2rad
    aspect_rad = aspect * deg2rad

    cos_incidence = (
        np.sin(solar_elev_rad) * np.cos(slope_rad)
        + np.cos(solar_elev_rad) * np.sin(slope_rad) * np.cos(solar_az_rad - aspect_rad)
    )
    cos_incidence = np.maximum(cos_incidence, 0.0)

    # Clamp sin(solar_elevation) to prevent numerical instability when
    # sun is near horizon. See MU0_MIN constant and TopoPyScale PR #132.
    sin_solar_elev = np.sin(solar_elev_rad)
    sin_stable = np.maximum(sin_solar_elev, MU0_MIN)
    correction = np.where(sin_solar_elev > 0.01, cos_incidence / sin_stable, 0.0)

    return np.minimum(sw_direct * correction, SOLAR_CONSTANT)


def diffuse_correction(
    sw_diffuse: np.ndarray,
    svf: np.ndarray,
) -> np.ndarray:
    """Apply sky view factor correction to diffuse shortwave radiation.

    SYNC: rust_kernels/src/radiation.rs::diffuse_correction
    SYNC: jax_kernels/radiation.py::diffuse_correction

    Parameters
    ----------
    sw_diffuse : array_like
        Diffuse shortwave radiation [W/m2].
    svf : array_like
        Sky view factor [-], [0, 1].

    Returns
    -------
    np.ndarray
        SVF-corrected diffuse shortwave [W/m2].
    """
    sw_diffuse = np.asarray(sw_diffuse, dtype=np.float64)
    svf = np.asarray(svf, dtype=np.float64)
    return sw_diffuse * svf


def longwave_correction(
    lw_source: np.ndarray,
    t_source: np.ndarray,
    t_unit: np.ndarray,
    vp_source: np.ndarray,
    vp_unit: np.ndarray,
    svf: np.ndarray,
) -> np.ndarray:
    """Apply elevation and SVF correction to longwave radiation.

    Uses the Brutsaert (1975) clear-sky emissivity model to separate
    clear-sky and cloud contributions, then transfers cloud emissivity
    to the target location.  This avoids the T^4 ratio amplification
    that occurs with the simpler scaling approach when the elevation
    difference between ERA5 grid cell and target is large.

    SYNC: rust_kernels/src/radiation.rs::longwave_correction
    SYNC: jax_kernels/radiation.py::longwave_correction

    Parameters
    ----------
    lw_source : array_like
        Source longwave downwelling radiation [W/m2].
    t_source : array_like
        Source (ERA5 2m) temperature [K].
    t_unit : array_like
        Target unit temperature [K].
    vp_source : array_like
        Vapor pressure at source [Pa].
    vp_unit : array_like
        Vapor pressure at target [Pa].
    svf : array_like
        Sky view factor [-], [0, 1].

    Returns
    -------
    np.ndarray
        Corrected longwave radiation [W/m2].
    """
    lw_source = np.asarray(lw_source, dtype=np.float64)
    t_source = np.asarray(t_source, dtype=np.float64)
    t_unit = np.asarray(t_unit, dtype=np.float64)
    vp_source = np.asarray(vp_source, dtype=np.float64)
    vp_unit = np.asarray(vp_unit, dtype=np.float64)
    svf = np.asarray(svf, dtype=np.float64)

    # Brutsaert (1975) clear-sky emissivity: ε_cs = 0.23 + 0.43*(e/T)^(1/5.7)
    # where e is vapor pressure [Pa] and T is temperature [K]
    inv_x2 = 1.0 / 5.7
    cse_source = 0.23 + 0.43 * np.maximum(vp_source / t_source, 0.0) ** inv_x2
    cse_target = 0.23 + 0.43 * np.maximum(vp_unit / t_unit, 0.0) ** inv_x2

    # Total emissivity at source (from ERA5 LW measurement)
    eps_total = lw_source / (SIGMA * t_source**4)

    # Cloud emissivity = total - clear-sky (non-negative)
    cle = np.maximum(eps_total - cse_source, 0.0)

    # All-sky emissivity at target (capped at 1.0)
    aef = np.minimum(cse_target + cle, 1.0)

    # Sky longwave at target
    lw_sky = aef * SIGMA * t_unit**4

    # Terrain emission at local temperature
    lw_terrain = SIGMA * t_unit**4

    return svf * lw_sky + (1.0 - svf) * lw_terrain
