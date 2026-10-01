"""Longwave radiation parameterization.

Estimates incoming longwave radiation (strd) from temperature, humidity,
and shortwave radiation when direct observations are unavailable
(e.g., Open-Meteo IFS backend).

Uses Dilley-O'Brien (1998) clear-sky emissivity with Unsworth-Monteith
(1975) cloud correction derived from clearness index.

References
----------
- Dilley, A.C. and O'Brien, D.M. (1998). Estimating downward clear-sky
  long-wave irradiance at the surface from screen temperature and
  precipitable water. Q.J.R. Meteorol. Soc., 124, 1391-1401.
- Unsworth, M.H. and Monteith, J.L. (1975). Long-wave radiation at the
  ground. Q.J.R. Meteorol. Soc., 101, 13-24.
- Konzelmann, T. et al. (1994). Parameterization of global and longwave
  incoming radiation for the Greenland ice sheet. Global Planet. Change.
"""

from __future__ import annotations

import numpy as np

# Stefan-Boltzmann constant [W m-2 K-4]
SIGMA = 5.670374419e-8

# Solar constant [W m-2]
SOLAR_CONSTANT = 1361.0


def _vapor_pressure_from_dewpoint(d2m: np.ndarray) -> np.ndarray:
    """Compute vapor pressure from 2m dewpoint temperature.

    Uses the Magnus formula (Bolton 1980).

    Parameters
    ----------
    d2m : array
        2m dewpoint temperature [K].

    Returns
    -------
    array
        Vapor pressure [Pa].
    """
    t_c = d2m - 273.15
    return 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))


def _clearness_index(ssrd: np.ndarray, tisr: np.ndarray) -> np.ndarray:
    """Compute clearness index from surface and TOA solar radiation.

    Parameters
    ----------
    ssrd : array
        Surface solar radiation downwards [J/m2 per timestep].
    tisr : array
        TOA incident solar radiation [J/m2 per timestep].

    Returns
    -------
    array
        Clearness index kt, clipped to [0, 1].
    """
    kt = np.where(tisr > 10.0, ssrd / tisr, 0.0)
    return np.clip(kt, 0.0, 1.0)


def _cloud_fraction_from_kt(kt: np.ndarray) -> np.ndarray:
    """Estimate cloud fraction from clearness index.

    Uses a linear mapping:
        kt = 0.0 -> cloud_fraction = 1.0 (overcast)
        kt = 0.75 -> cloud_fraction = 0.0 (clear sky)

    Clipped to [0, 1].

    Parameters
    ----------
    kt : array
        Clearness index [0, 1].

    Returns
    -------
    array
        Cloud fraction [0, 1].
    """
    cf = 1.0 - kt / 0.75
    return np.clip(cf, 0.0, 1.0)


def estimate_longwave(
    t2m: np.ndarray,
    d2m: np.ndarray,
    ssrd: np.ndarray,
    tisr: np.ndarray,
) -> np.ndarray:
    """Estimate incoming longwave radiation from temperature and humidity.

    Uses Dilley-O'Brien (1998) clear-sky emissivity with Unsworth-Monteith
    cloud correction. Returns values in J/m2 (1-hour accumulation) to match
    ERA5 convention for strd.

    Parameters
    ----------
    t2m : array
        2m temperature [K].
    d2m : array
        2m dewpoint temperature [K].
    ssrd : array
        Surface solar radiation downwards [J/m2 per timestep].
    tisr : array
        TOA incident solar radiation [J/m2 per timestep].

    Returns
    -------
    array
        Estimated strd [J/m2 per timestep], same shape as inputs.
    """
    # Vapor pressure from dewpoint [Pa]
    e = _vapor_pressure_from_dewpoint(d2m)

    # Convert to hPa for the Dilley-O'Brien formula
    e_hpa = e / 100.0

    # Precipitable water estimate [cm]
    # Simple approximation from surface vapor pressure (Reitan 1963)
    w = 4.65 * (e_hpa / t2m)

    # Clear-sky emissivity: Dilley-O'Brien (1998) Eq. 7
    # epsilon_cs = 59.38 + 113.7 * (T/273.16)^6 + 96.96 * sqrt(w/25)
    # This gives emitted flux density in W/m2 directly when multiplied by sigma*T^4
    lw_clear = (
        59.38
        + 113.7 * (t2m / 273.16) ** 6
        + 96.96 * np.sqrt(np.maximum(w, 0.0) / 25.0)
    )

    # Cloud correction: Unsworth-Monteith (1975)
    # LW_all = LW_clear * (1 + 0.22 * cf^2.75)
    kt = _clearness_index(ssrd, tisr)
    cf = _cloud_fraction_from_kt(kt)
    cloud_factor = 1.0 + 0.22 * cf ** 2.75

    # All-sky longwave [W/m2]
    lw_allsky = lw_clear * cloud_factor

    # Physical bounds: 50-600 W/m2
    lw_allsky = np.clip(lw_allsky, 50.0, 600.0)

    # Convert to J/m2 per hour to match ERA5 strd convention
    strd = lw_allsky * 3600.0

    return strd
