"""Crocus/SAFRAN NetCDF format output writer."""

import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from topopyscale2.spatial.units import SpatialUnit


class CrocusWriter:
    """Write forcing in Crocus/SAFRAN NetCDF format.

    Crocus is a detailed snowpack model developed by Meteo-France.
    SAFRAN (Systeme d'Analyse Fournissant des Renseignements Atmospheriques
    a la Neige) provides the forcing format for Crocus.

    Standard SAFRAN/Crocus forcing variables:
    - Tair: Air temperature at 2m [K]
    - Qair: Specific humidity [kg/kg]
    - Wind: Wind speed [m/s]
    - Wind_DIR: Wind direction [degrees from north] (optional)
    - PSurf: Surface pressure [Pa]
    - DIR_SWdown: Direct shortwave radiation [W/m2]
    - SCA_SWdown: Scattered/diffuse shortwave radiation [W/m2]
    - LWdown: Downwelling longwave radiation [W/m2]
    - Rainf: Rainfall rate [kg/m2/s]
    - Snowf: Snowfall rate [kg/m2/s]
    - CO2air: CO2 concentration (optional, usually constant)

    Uses SURFEX/Crocus conventions.
    """

    model_name = "crocus"
    file_format = "netcdf"

    def __init__(
        self,
        compression: bool = True,
        complevel: int = 4,
        timestep_seconds: float = 3600.0,
        include_co2: bool = False,
        co2_ppm: float = 420.0,
        include_wind_dir: bool = False,
    ):
        """Initialize Crocus writer.

        Parameters
        ----------
        compression : bool
            Whether to compress NetCDF output.
        complevel : int
            Compression level (1-9).
        timestep_seconds : float
            Timestep in seconds for rate conversions.
        include_co2 : bool
            Whether to include CO2 concentration variable.
        co2_ppm : float
            CO2 concentration in ppm if include_co2 is True.
        include_wind_dir : bool
            Whether to include wind direction variable (filled with NaN).
        """
        self.compression = compression
        self.complevel = complevel
        self.timestep_seconds = timestep_seconds
        self.include_co2 = include_co2
        self.co2_ppm = co2_ppm
        self.include_wind_dir = include_wind_dir

    def _create_crocus_dataset(
        self,
        forcing: xr.Dataset,
        unit: SpatialUnit,
        unit_idx: Optional[int] = None,
    ) -> xr.Dataset:
        """Create Crocus/SAFRAN-formatted dataset for a single unit.

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
            Crocus-formatted dataset.
        """
        # Extract data for this unit
        if unit_idx is not None and "unit_id" in forcing.dims:
            ds = forcing.isel(unit_id=unit_idx)
        else:
            ds = forcing

        times = ds.time.values
        nt = len(times)

        # Convert precipitation from accumulated (kg/m2 per timestep) to rate (kg/m2/s)
        precip_to_rate = 1.0 / self.timestep_seconds

        # Crocus variable mapping with SURFEX conventions
        data_vars = {
            "Tair": (
                ["time"],
                ds["temperature"].values,
                {
                    "units": "K",
                    "long_name": "Near surface air temperature",
                    "standard_name": "air_temperature",
                },
            ),
            "Qair": (
                ["time"],
                ds["humidity_specific"].values,
                {
                    "units": "kg kg-1",
                    "long_name": "Near surface specific humidity",
                    "standard_name": "specific_humidity",
                },
            ),
            "Wind": (
                ["time"],
                ds["wind_speed"].values,
                {
                    "units": "m s-1",
                    "long_name": "Wind speed",
                    "standard_name": "wind_speed",
                },
            ),
            "PSurf": (
                ["time"],
                ds["pressure"].values,
                {
                    "units": "Pa",
                    "long_name": "Surface pressure",
                    "standard_name": "surface_air_pressure",
                },
            ),
            "DIR_SWdown": (
                ["time"],
                ds["shortwave_direct"].values,
                {
                    "units": "W m-2",
                    "long_name": "Surface incident direct shortwave radiation",
                    "standard_name": "surface_direct_downwelling_shortwave_flux_in_air",
                },
            ),
            "SCA_SWdown": (
                ["time"],
                ds["shortwave_diffuse"].values,
                {
                    "units": "W m-2",
                    "long_name": "Surface incident diffuse shortwave radiation",
                    "standard_name": "surface_diffuse_downwelling_shortwave_flux_in_air",
                },
            ),
            "LWdown": (
                ["time"],
                ds["longwave"].values,
                {
                    "units": "W m-2",
                    "long_name": "Surface incident longwave radiation",
                    "standard_name": "surface_downwelling_longwave_flux_in_air",
                },
            ),
            "Rainf": (
                ["time"],
                ds["rainfall"].values * precip_to_rate,
                {
                    "units": "kg m-2 s-1",
                    "long_name": "Rainfall rate",
                    "standard_name": "rainfall_flux",
                },
            ),
            "Snowf": (
                ["time"],
                ds["snowfall"].values * precip_to_rate,
                {
                    "units": "kg m-2 s-1",
                    "long_name": "Snowfall rate",
                    "standard_name": "snowfall_flux",
                },
            ),
        }

        # Optional wind direction
        if self.include_wind_dir:
            data_vars["Wind_DIR"] = (
                ["time"],
                np.full(nt, np.nan, dtype=np.float64),
                {
                    "units": "degrees",
                    "long_name": "Wind direction from north",
                    "standard_name": "wind_from_direction",
                },
            )

        # Optional CO2
        if self.include_co2:
            # Convert ppm to kg/kg (approximate for dry air)
            co2_kgkg = self.co2_ppm * 44.01 / (28.97 * 1e6)
            data_vars["CO2air"] = (
                ["time"],
                np.full(nt, co2_kgkg, dtype=np.float64),
                {
                    "units": "kg kg-1",
                    "long_name": "Near surface CO2 concentration",
                    "standard_name": "mass_fraction_of_carbon_dioxide_in_air",
                },
            )

        # Create dataset
        crocus_ds = xr.Dataset(
            data_vars=data_vars,
            coords={"time": times},
        )

        # Add point coordinates (SURFEX-style)
        crocus_ds = crocus_ds.expand_dims({"Number_of_points": 1})
        crocus_ds = crocus_ds.assign_coords(
            LAT=(["Number_of_points"], [unit.y]),
            LON=(["Number_of_points"], [unit.x]),
            ZS=(["Number_of_points"], [unit.elevation]),
        )

        # Add global attributes
        crocus_ds.attrs.update({
            "Conventions": "CF-1.6",
            "title": "Crocus/SAFRAN forcing from TopoPyScale 2.0",
            "institution": "TopoPyScale",
            "source": "ERA5 reanalysis, topographically downscaled",
            "history": f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            "references": "SURFEX/Crocus forcing format",
            "station_id": unit.id,
            "surface_type": unit.surface_type,
        })

        return crocus_ds

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write one Crocus NetCDF file per spatial unit.

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
            # Create Crocus dataset
            crocus_ds = self._create_crocus_dataset(forcing, unit, i)

            # Determine filename from time range
            times = forcing.time.values
            if len(times) > 0:
                t_start = str(times[0])[:10].replace("-", "")
                t_end = str(times[-1])[:10].replace("-", "")
                filename = f"FORCING_{unit.id}_{t_start}_{t_end}.nc"
            else:
                filename = f"FORCING_{unit.id}.nc"

            filepath = output_dir / filename

            # Encoding with compression
            encoding = {}
            if self.compression:
                for var in crocus_ds.data_vars:
                    encoding[var] = {
                        "zlib": True,
                        "complevel": self.complevel,
                    }

            crocus_ds.to_netcdf(filepath, encoding=encoding)
            written.append(filepath)

        return written

    def write_combined(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write all units to a single Crocus NetCDF file (multi-point).

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
            return self.write(forcing, units, output_dir)

        times = forcing.time.values
        n_units = len(units)
        nt = len(times)

        # Convert precipitation
        precip_to_rate = 1.0 / self.timestep_seconds

        # Build combined dataset with Number_of_points dimension
        data_vars = {
            "Tair": (["time", "Number_of_points"], forcing["temperature"].values),
            "Qair": (["time", "Number_of_points"], forcing["humidity_specific"].values),
            "Wind": (["time", "Number_of_points"], forcing["wind_speed"].values),
            "PSurf": (["time", "Number_of_points"], forcing["pressure"].values),
            "DIR_SWdown": (
                ["time", "Number_of_points"],
                forcing["shortwave_direct"].values,
            ),
            "SCA_SWdown": (
                ["time", "Number_of_points"],
                forcing["shortwave_diffuse"].values,
            ),
            "LWdown": (["time", "Number_of_points"], forcing["longwave"].values),
            "Rainf": (
                ["time", "Number_of_points"],
                forcing["rainfall"].values * precip_to_rate,
            ),
            "Snowf": (
                ["time", "Number_of_points"],
                forcing["snowfall"].values * precip_to_rate,
            ),
        }

        if self.include_wind_dir:
            data_vars["Wind_DIR"] = (
                ["time", "Number_of_points"],
                np.full((nt, n_units), np.nan, dtype=np.float64),
            )

        if self.include_co2:
            co2_kgkg = self.co2_ppm * 44.01 / (28.97 * 1e6)
            data_vars["CO2air"] = (
                ["time", "Number_of_points"],
                np.full((nt, n_units), co2_kgkg, dtype=np.float64),
            )

        coords = {
            "time": times,
            "Number_of_points": np.arange(n_units),
            "LAT": (["Number_of_points"], [u.y for u in units]),
            "LON": (["Number_of_points"], [u.x for u in units]),
            "ZS": (["Number_of_points"], [u.elevation for u in units]),
        }

        crocus_ds = xr.Dataset(data_vars=data_vars, coords=coords)

        # Global attributes
        crocus_ds.attrs.update({
            "Conventions": "CF-1.6",
            "title": "Crocus/SAFRAN forcing from TopoPyScale 2.0",
            "institution": "TopoPyScale",
            "source": "ERA5 reanalysis, topographically downscaled",
            "history": f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
            "references": "SURFEX/Crocus forcing format",
        })

        # Filename
        t_start = str(times[0])[:10].replace("-", "")
        t_end = str(times[-1])[:10].replace("-", "")
        filename = f"FORCING_{t_start}_{t_end}.nc"

        filepath = output_dir / filename

        # Encoding
        encoding = {}
        if self.compression:
            for var in crocus_ds.data_vars:
                encoding[var] = {"zlib": True, "complevel": self.complevel}

        crocus_ds.to_netcdf(filepath, encoding=encoding)
        return [filepath]
