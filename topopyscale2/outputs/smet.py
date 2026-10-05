"""SNOWPACK SMET format output writer."""

from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.spatial.units import SpatialUnit


class SMETWriter:
    """Write forcing in SMET format for SNOWPACK.

    SMET (Swiss Meteorological Exchange format) is the standard input format
    for SNOWPACK and Alpine3D. It consists of a header section with metadata
    followed by a data section with columns.

    Standard variables:
    - TA: Air temperature [K]
    - RH: Relative humidity [%]
    - VW: Wind speed [m/s]
    - ISWR: Incoming shortwave radiation [W/m2]
    - ILWR: Incoming longwave radiation [W/m2]
    - PSUM: Precipitation [mm]
    - PSUM_PH: Precipitation phase (0=snow, 1=rain) [fraction]

    One file per spatial unit.
    """

    model_name = "snowpack"
    file_format = "smet"

    def __init__(
        self,
        include_precip_phase: bool = True,
        nodata: float = -999.0,
        station_name: Optional[str] = None,
    ):
        """Initialize SMET writer.

        Parameters
        ----------
        include_precip_phase : bool
            Whether to include PSUM_PH column (precipitation phase fraction).
        nodata : float
            Nodata value to use in SMET files.
        station_name : str, optional
            Base station name for SMET headers.
        """
        self.include_precip_phase = include_precip_phase
        self.nodata = nodata
        self.station_name = station_name

    def _format_timestamp(self, dt: pd.Timestamp) -> str:
        """Format timestamp in SMET ISO format."""
        return dt.strftime("%Y-%m-%dT%H:%M:%S")

    def _compute_precip_phase(
        self,
        rainfall: np.ndarray,
        snowfall: np.ndarray,
    ) -> np.ndarray:
        """Compute precipitation phase fraction (0=snow, 1=rain).

        Parameters
        ----------
        rainfall : array
            Rainfall amount.
        snowfall : array
            Snowfall amount.

        Returns
        -------
        array
            Phase fraction where 0=pure snow, 1=pure rain.
        """
        total = rainfall + snowfall
        phase = np.where(total > 0, rainfall / total, 0.5)
        return phase

    def _write_smet_file(
        self,
        filepath: Path,
        times: pd.DatetimeIndex,
        data: dict[str, np.ndarray],
        unit: SpatialUnit,
    ) -> None:
        """Write a single SMET file.

        Parameters
        ----------
        filepath : Path
            Output file path.
        times : DatetimeIndex
            Time values.
        data : dict
            Data arrays keyed by variable name.
        unit : SpatialUnit
            Spatial unit for metadata.
        """
        station_name = self.station_name or f"TPS2_{unit.id}"

        with open(filepath, "w", encoding="utf-8") as f:
            # SMET header
            f.write("SMET 1.1 ASCII\n")
            f.write("[HEADER]\n")
            f.write(f"station_id       = {unit.id}\n")
            f.write(f"station_name     = {station_name}\n")
            f.write(f"latitude         = {unit.y:.6f}\n")
            f.write(f"longitude        = {unit.x:.6f}\n")
            f.write(f"altitude         = {unit.elevation:.1f}\n")
            f.write(f"nodata           = {self.nodata}\n")
            f.write("tz               = 0\n")  # UTC
            f.write("source           = TopoPyScale 2.0\n")
            f.write(f"creation_date    = {datetime.utcnow().isoformat()}\n")

            # Field definitions
            fields = ["timestamp", "TA", "RH", "VW", "ISWR", "ILWR", "PSUM"]
            if self.include_precip_phase:
                fields.append("PSUM_PH")

            f.write(f"fields           = {' '.join(fields)}\n")
            f.write("[DATA]\n")

            # Data rows
            for i, t in enumerate(times):
                timestamp = self._format_timestamp(t)
                ta = data["temperature"][i]
                rh = data["humidity_relative"][i] * 100.0  # Convert to %
                vw = data["wind_speed"][i]
                iswr = data["shortwave_total"][i]
                ilwr = data["longwave"][i]
                psum = data["precipitation"][i]  # Already in mm

                row = f"{timestamp} {ta:.2f} {rh:.2f} {vw:.2f} {iswr:.2f} {ilwr:.2f} {psum:.4f}"

                if self.include_precip_phase:
                    phase = data["precip_phase"][i]
                    row += f" {phase:.4f}"

                f.write(row + "\n")

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write one SMET file per spatial unit.

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
            # Extract data for this unit
            if "unit_id" in forcing.dims:
                ds = forcing.isel(unit_id=i)
            else:
                ds = forcing

            times = pd.DatetimeIndex(ds.time.values)

            # Prepare data arrays
            data = {
                "temperature": ds["temperature"].values,
                "humidity_relative": ds["humidity_relative"].values,
                "wind_speed": ds["wind_speed"].values,
                "shortwave_total": (
                    ds["shortwave_direct"].values + ds["shortwave_diffuse"].values
                ),
                "longwave": ds["longwave"].values,
                "precipitation": ds["precipitation"].values * 1000.0,  # kg/m2 -> mm
            }

            if self.include_precip_phase:
                data["precip_phase"] = self._compute_precip_phase(
                    ds["rainfall"].values,
                    ds["snowfall"].values,
                )

            filepath = output_dir / f"{unit.id}.smet"
            self._write_smet_file(filepath, times, data, unit)
            written.append(filepath)

        return written
