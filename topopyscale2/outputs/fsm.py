"""FSM (Factorial Snow Model) text file output writer."""

from pathlib import Path

import pandas as pd
import xarray as xr

from topopyscale2.outputs.base import relative_humidity_percent
from topopyscale2.spatial.units import SpatialUnit


class FSMWriter:
    """Write forcing in FSM text file format.

    One file per spatial unit. Columns:
    year, month, day, hour, SW↓, LW↓, Sf, Rf, Ta, RH, Ua, Ps
    """

    model_name = "fsm"
    file_format = "txt"

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write one FSM text file per spatial unit.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data. If dims are (time,) it's for a single unit.
            If dims are (time, unit_id) it contains all units.
        units : list[SpatialUnit]
        output_dir : Path

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

            # SW = direct + diffuse
            sw_down = ds["shortwave_direct"].values + ds["shortwave_diffuse"].values
            lw_down = ds["longwave"].values

            # Snowfall/rainfall rate [kg/m2/s] from mm per timestep
            # Infer timestep from data (1 mm water = 1 kg/m2)
            if len(times) > 1:
                dt_s = (times[1] - times[0]).total_seconds()
            else:
                dt_s = 3600.0
            sf = ds["snowfall"].values / dt_s
            rf = ds["rainfall"].values / dt_s

            ta = ds["temperature"].values
            rh = relative_humidity_percent(ds["humidity_relative"])   # FSM reads RH in %
            ua = ds["wind_speed"].values
            ps = ds["pressure"].values

            filepath = output_dir / f"fsm_unit_{unit.id}.txt"

            with open(filepath, "w", encoding="utf-8") as f:
                for t in range(len(times)):
                    dt = times[t]
                    f.write(
                        f"{dt.year:4d} {dt.month:2d} {dt.day:2d} {dt.hour:2d} "
                        f"{sw_down[t]:10.4f} {lw_down[t]:10.4f} "
                        f"{sf[t]:12.6e} {rf[t]:12.6e} "
                        f"{ta[t]:8.3f} {rh[t]:6.4f} {ua[t]:8.3f} {ps[t]:10.2f}\n"
                    )

            written.append(filepath)

        return written
