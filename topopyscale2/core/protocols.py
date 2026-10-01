"""Protocol interfaces for TopoPyScale 2.0 pluggable components."""

from pathlib import Path
from typing import Protocol, runtime_checkable

import xarray as xr

from topopyscale2.spatial.units import SpatialUnit


@runtime_checkable
class MeteoSource(Protocol):
    """Interface for meteorological data sources."""

    def fetch(
        self, bbox: tuple[float, float, float, float], time_range: list[str]
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch surface and pressure level data.

        Returns (ds_surface, ds_pressure).
        """
        ...

    def compute_lapse_rate(
        self, ds_plev: xr.Dataset, z_surface: xr.DataArray
    ) -> xr.DataArray:
        """Compute temperature lapse rate from pressure levels."""
        ...

    def compute_solar_geometry(
        self, time, lat, lon
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute solar elevation and azimuth angles."""
        ...

    def compute_clearness_index(self, ds_surf: xr.Dataset) -> xr.DataArray:
        """Compute clearness index kt = ssrd / sw_toa."""
        ...


@runtime_checkable
class OutputWriter(Protocol):
    """Interface for output file writers."""

    model_name: str
    file_format: str

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write forcing data to disk. Returns list of written file paths."""
        ...


@runtime_checkable
class SpatialProcessor(Protocol):
    """Interface for spatial unit generation."""

    def generate(self, dem_data: xr.Dataset) -> list[SpatialUnit]:
        """Generate spatial units from DEM data."""
        ...
