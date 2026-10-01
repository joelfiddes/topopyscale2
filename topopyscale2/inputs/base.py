"""Base protocol for meteorological data sources."""

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import xarray as xr


@runtime_checkable
class MeteoSource(Protocol):
    """Interface for meteorological data sources.

    All NWP data sources must implement this protocol to be usable
    with the TopoPyScale 2.0 downscaling pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the data source (e.g., 'era5', 'hres', 'icon')."""
        ...

    @property
    @abstractmethod
    def resolution_m(self) -> float:
        """Native horizontal resolution of the source in meters."""
        ...

    def fetch(
        self, bbox: tuple[float, float, float, float], time_range: list[str]
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch surface and pressure level data.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as (west, south, east, north) in degrees.
        time_range : list[str]
            Time range as [start_date, end_date] in ISO format.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Surface and pressure level datasets.
        """
        ...

    def compute_lapse_rate(
        self, ds_plev: xr.Dataset, z_surface: xr.DataArray
    ) -> xr.DataArray:
        """Compute temperature lapse rate from pressure levels.

        Parameters
        ----------
        ds_plev : xr.Dataset
            Pressure level dataset with 't' (temperature) and 'z' (geopotential).
        z_surface : xr.DataArray
            Surface elevation grid [m].

        Returns
        -------
        xr.DataArray
            Lapse rate gamma [K/m] with dims (time, latitude, longitude).
        """
        ...

    def compute_solar_geometry(
        self, time, lat, lon
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute solar elevation and azimuth angles.

        Parameters
        ----------
        time : array-like
            Timestamps for computation.
        lat : array-like
            Latitude values [degrees].
        lon : array-like
            Longitude values [degrees].

        Returns
        -------
        tuple[xr.DataArray, xr.DataArray]
            Solar elevation and azimuth angles [degrees].
        """
        ...

    def compute_clearness_index(
        self, ds_surf: xr.Dataset, solar_elevation: xr.DataArray
    ) -> xr.DataArray:
        """Compute clearness index kt = ssrd / sw_toa.

        Parameters
        ----------
        ds_surf : xr.Dataset
            Surface dataset with 'ssrd' variable.
        solar_elevation : xr.DataArray
            Solar elevation angles [degrees].

        Returns
        -------
        xr.DataArray
            Clearness index kt, clamped to [0, 1].
        """
        ...


# Standard variable names used internally by TopoPyScale 2.0
STANDARD_VARIABLES = {
    # Surface variables
    "t2m": "2m temperature [K]",
    "d2m": "2m dewpoint temperature [K]",
    "sp": "Surface pressure [Pa]",
    "ssrd": "Surface shortwave radiation downward [J/m2]",
    "strd": "Surface longwave radiation downward [J/m2]",
    "tp": "Total precipitation [m]",
    "u10": "10m u-wind component [m/s]",
    "v10": "10m v-wind component [m/s]",
    "z_surf": "Surface geopotential [m2/s2]",
    # Pressure level variables
    "t": "Temperature [K]",
    "z": "Geopotential [m2/s2]",
    "u": "U-wind component [m/s]",
    "v": "V-wind component [m/s]",
    "q": "Specific humidity [kg/kg]",
    "r": "Relative humidity [%]",
}

# Default variable mappings for common NWP sources
DEFAULT_VARIABLE_MAPPINGS = {
    "era5": {},  # ERA5 uses standard names
    "hres": {
        "2t": "t2m",
        "2d": "d2m",
        "msl": "sp",
    },
    "icon": {
        "T_2M": "t2m",
        "TD_2M": "d2m",
        "PS": "sp",
        "ASWDIFD_S": "ssrd",
        "ATHD_S": "strd",
        "TOT_PREC": "tp",
    },
    "gfs": {
        "TMP_2maboveground": "t2m",
        "DPT_2maboveground": "d2m",
        "PRES_surface": "sp",
    },
}
