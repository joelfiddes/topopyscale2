"""CryoGrid NetCDF format output writer."""

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from topopyscale2.spatial.units import SpatialUnit


class CryoGridWriter:
    """Write forcing in CryoGrid NetCDF format.

    CryoGrid is a permafrost and ground thermal model. The forcing format
    requires specific variable names and includes ground heat flux placeholder.

    Standard CryoGrid forcing variables:
    - Tair: Air temperature [K]
    - q: Specific humidity [kg/kg]
    - wind: Wind speed [m/s]
    - Sin: Incoming shortwave radiation [W/m2]
    - Lin: Incoming longwave radiation [W/m2]
    - p: Surface pressure [Pa]
    - snowfall: Snowfall rate [m/s water equivalent]
    - rainfall: Rainfall rate [m/s water equivalent]
    - Qh: Sensible heat flux (placeholder) [W/m2]
    - Qe: Latent heat flux (placeholder) [W/m2]
    - Qg: Ground heat flux (placeholder) [W/m2]
    """

    model_name = "cryogrid"
    file_format = "netcdf"

    def __init__(
        self,
        compression: bool = True,
        complevel: int = 4,
        include_flux_placeholders: bool = True,
        timestep_seconds: float = 3600.0,
    ):
        """Initialize CryoGrid writer.

        Parameters
        ----------
        compression : bool
            Whether to compress NetCDF output.
        complevel : int
            Compression level (1-9).
        include_flux_placeholders : bool
            Whether to include Qh, Qe, Qg placeholder variables.
        timestep_seconds : float
            Timestep in seconds for rate conversions.
        """
        self.compression = compression
        self.complevel = complevel
        self.include_flux_placeholders = include_flux_placeholders
        self.timestep_seconds = timestep_seconds

    def _create_cryogrid_dataset(
        self,
        forcing: xr.Dataset,
        unit: SpatialUnit,
        unit_idx: Optional[int] = None,
    ) -> xr.Dataset:
        """Create CryoGrid-formatted dataset for a single unit.

        Parameters
        ----------
        forcing : xr.Dataset
            Input forcing data.
        unit : SpatialUnit
            Spatial unit metadata.
        unit_idx : int, optional
            Index if forcing has unit_id dimension.

        Returns
        -------
        xr.Dataset
            CryoGrid-formatted dataset.
        """
        # Extract data for this unit
        if unit_idx is not None and "unit_id" in forcing.dims:
            ds = forcing.isel(unit_id=unit_idx)
        else:
            ds = forcing

        times = ds.time.values
        nt = len(times)

        # Convert precipitation from accumulated (kg/m2) to rate (m/s)
        # kg/m2 = mm, so divide by 1000 to get m, then by timestep for rate
        precip_to_rate = 1.0 / (1000.0 * self.timestep_seconds)

        # CryoGrid variable mapping
        data_vars = {
            "Tair": (
                ["time"],
                ds["temperature"].values,
                {
                    "units": "K",
                    "long_name": "Air temperature",
                    "standard_name": "air_temperature",
                },
            ),
            "q": (
                ["time"],
                ds["humidity_specific"].values,
                {
                    "units": "kg kg-1",
                    "long_name": "Specific humidity",
                    "standard_name": "specific_humidity",
                },
            ),
            "wind": (
                ["time"],
                ds["wind_speed"].values,
                {
                    "units": "m s-1",
                    "long_name": "Wind speed",
                    "standard_name": "wind_speed",
                },
            ),
            "Sin": (
                ["time"],
                ds["shortwave_direct"].values + ds["shortwave_diffuse"].values,
                {
                    "units": "W m-2",
                    "long_name": "Incoming shortwave radiation",
                    "standard_name": "surface_downwelling_shortwave_flux_in_air",
                },
            ),
            "Lin": (
                ["time"],
                ds["longwave"].values,
                {
                    "units": "W m-2",
                    "long_name": "Incoming longwave radiation",
                    "standard_name": "surface_downwelling_longwave_flux_in_air",
                },
            ),
            "p": (
                ["time"],
                ds["pressure"].values,
                {
                    "units": "Pa",
                    "long_name": "Surface air pressure",
                    "standard_name": "surface_air_pressure",
                },
            ),
            "snowfall": (
                ["time"],
                ds["snowfall"].values * precip_to_rate,
                {
                    "units": "m s-1",
                    "long_name": "Snowfall rate (water equivalent)",
                    "standard_name": "snowfall_flux",
                },
            ),
            "rainfall": (
                ["time"],
                ds["rainfall"].values * precip_to_rate,
                {
                    "units": "m s-1",
                    "long_name": "Rainfall rate",
                    "standard_name": "rainfall_flux",
                },
            ),
        }

        # Add flux placeholders if requested
        if self.include_flux_placeholders:
            zeros = np.zeros(nt, dtype=np.float64)
            nans = np.full(nt, np.nan, dtype=np.float64)

            data_vars["Qh"] = (
                ["time"],
                nans.copy(),
                {
                    "units": "W m-2",
                    "long_name": "Sensible heat flux (placeholder)",
                    "standard_name": "surface_upward_sensible_heat_flux",
                },
            )
            data_vars["Qe"] = (
                ["time"],
                nans.copy(),
                {
                    "units": "W m-2",
                    "long_name": "Latent heat flux (placeholder)",
                    "standard_name": "surface_upward_latent_heat_flux",
                },
            )
            data_vars["Qg"] = (
                ["time"],
                nans.copy(),
                {
                    "units": "W m-2",
                    "long_name": "Ground heat flux (placeholder)",
                    "standard_name": "downward_heat_flux_in_soil",
                },
            )

        # Create dataset
        cg_ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": times},
        )

        # Add global attributes
        cg_ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": "CryoGrid forcing from TopoPyScale 2.0",
            "institution": "TopoPyScale",
            "source": "ERA5 reanalysis, topographically downscaled",
            "history": f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            "station_id": unit.id,
            "latitude": unit.y,
            "longitude": unit.x,
            "elevation": unit.elevation,
            "surface_type": unit.surface_type,
        })

        return cg_ds

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write one CryoGrid NetCDF file per spatial unit.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data with dims (time,) or (time, unit_id).
        units : list[SpatialUnit]
            Spatial units.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written = []

        for i, unit in enumerate(units):
            # Create CryoGrid dataset
            cg_ds = self._create_cryogrid_dataset(forcing, unit, i)

            # Determine filename from time range
            times = cg_ds.time.values
            if len(times) > 0:
                t_start = str(times[0])[:10].replace("-", "")
                t_end = str(times[-1])[:10].replace("-", "")
                filename = f"cryogrid_{unit.id}_{t_start}_{t_end}.nc"
            else:
                filename = f"cryogrid_{unit.id}.nc"

            filepath = output_dir / filename

            # Encoding with compression
            encoding = {}
            if self.compression:
                for var in cg_ds.data_vars:
                    encoding[var] = {
                        "zlib": True,
                        "complevel": self.complevel,
                    }

            cg_ds.to_netcdf(filepath, encoding=encoding)
            written.append(filepath)

        return written

    def write_combined(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write all units to a single CryoGrid NetCDF file with point dimension.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data with dims (time, unit_id).
        units : list[SpatialUnit]
            Spatial units.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Written file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if "unit_id" not in forcing.dims:
            # Single unit case, call regular write
            return self.write(forcing, units, output_dir)

        times = forcing.time.values
        n_units = len(units)
        nt = len(times)

        # Convert precipitation
        precip_to_rate = 1.0 / (1000.0 * self.timestep_seconds)

        # Build combined dataset
        data_vars = {
            "Tair": (["time", "point"], forcing["temperature"].values),
            "q": (["time", "point"], forcing["humidity_specific"].values),
            "wind": (["time", "point"], forcing["wind_speed"].values),
            "Sin": (
                ["time", "point"],
                forcing["shortwave_direct"].values + forcing["shortwave_diffuse"].values,
            ),
            "Lin": (["time", "point"], forcing["longwave"].values),
            "p": (["time", "point"], forcing["pressure"].values),
            "snowfall": (["time", "point"], forcing["snowfall"].values * precip_to_rate),
            "rainfall": (["time", "point"], forcing["rainfall"].values * precip_to_rate),
        }

        if self.include_flux_placeholders:
            nans = np.full((nt, n_units), np.nan, dtype=np.float64)
            data_vars["Qh"] = (["time", "point"], nans.copy())
            data_vars["Qe"] = (["time", "point"], nans.copy())
            data_vars["Qg"] = (["time", "point"], nans.copy())

        coords = {
            "time": times,
            "point": [u.id for u in units],
            "latitude": ("point", [u.y for u in units]),
            "longitude": ("point", [u.x for u in units]),
            "elevation": ("point", [u.elevation for u in units]),
        }

        cg_ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Global attributes
        cg_ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": "CryoGrid forcing from TopoPyScale 2.0",
            "institution": "TopoPyScale",
            "source": "ERA5 reanalysis, topographically downscaled",
            "history": f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        })

        # Filename
        t_start = str(times[0])[:10].replace("-", "")
        t_end = str(times[-1])[:10].replace("-", "")
        filename = f"cryogrid_forcing_{t_start}_{t_end}.nc"

        filepath = output_dir / filename

        # Encoding
        encoding = {}
        if self.compression:
            for var in cg_ds.data_vars:
                encoding[var] = {"zlib": True, "complevel": self.complevel}

        cg_ds.to_netcdf(filepath, encoding=encoding)
        return [filepath]
