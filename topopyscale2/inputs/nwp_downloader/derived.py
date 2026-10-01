"""Derived variable computations.

Computes relative humidity from specific humidity using Bolton (1980).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def compute_relative_humidity(
    temperature: xr.DataArray,
    pressure: xr.DataArray,
    specific_humidity: xr.DataArray,
) -> xr.DataArray:
    """Compute relative humidity from specific humidity using Bolton (1980).

    Args:
        temperature: Air temperature in Kelvin.
        pressure: Air pressure in Pa.
        specific_humidity: Specific humidity in kg/kg.

    Returns:
        Relative humidity as percentage (0-100).
    """
    # Mixing ratio from specific humidity
    mr = specific_humidity / (1.0 - specific_humidity)

    # Actual vapor pressure (Pa)
    e = mr * pressure / (0.62197 + mr)

    # Saturation vapor pressure (Pa) — Bolton (1980)
    es = 611.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))

    # Relative humidity (0-100%)
    rh = np.clip(e / es * 100.0, 0.0, 100.0)

    rh.name = "r"
    rh.attrs = {
        "long_name": "Relative humidity",
        "units": "%",
        "method": "Bolton (1980)",
    }
    return rh


def compute_surface_geopotential(
    surface_pressure: xr.DataArray,
    mean_sea_level_pressure: xr.DataArray,
    temperature_2m: xr.DataArray,
) -> xr.DataArray:
    """Compute surface geopotential from pressure and temperature.

    Uses the hypsometric equation: Z = R*T * ln(P_msl / P_surf)

    Args:
        surface_pressure: Surface pressure in Pa.
        mean_sea_level_pressure: Mean sea level pressure in Pa.
        temperature_2m: 2m temperature in K.

    Returns:
        Surface geopotential in m² s⁻².
    """
    R = 287.0  # Gas constant for dry air (J/kg/K)
    z = R * temperature_2m * np.log(mean_sea_level_pressure / surface_pressure)

    z.name = "z"
    z.attrs = {
        "long_name": "Surface geopotential",
        "units": "m**2 s**-2",
    }
    return z
