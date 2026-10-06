"""FSM2 (Flexible Snow Model 2) multi-point text file output writer.

FSM2 differs from FSM1:
- Single forcing file for all points (Npnts rows per timestep)
- Input format: year month day hour SW LW Sf Rf Ta RH Ua Ps
- Multi-point simulation support
"""

from pathlib import Path

import pandas as pd
import xarray as xr

from topopyscale2.outputs.base import relative_humidity_percent
from topopyscale2.spatial.units import SpatialUnit


class FSM2Writer:
    """Write forcing in FSM2 multi-point text file format.

    Unlike FSM1 (one file per unit), FSM2 takes a single file with
    all points. For each timestep, Npnts rows are written, one per point.

    Input file format (same 12 columns as FSM1):
        year month day hour SW LW Sf Rf Ta RH Ua Ps

    Where:
        - SW: Incoming shortwave radiation [W/m2]
        - LW: Incoming longwave radiation [W/m2]
        - Sf: Snowfall rate [kg/m2/s]
        - Rf: Rainfall rate [kg/m2/s]
        - Ta: Air temperature [K]
        - RH: Relative humidity [%]
        - Ua: Wind speed [m/s]
        - Ps: Surface pressure [Pa]
    """

    model_name = "fsm2"
    file_format = "txt"

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
        filename: str = "met_input.txt",
    ) -> Path:
        """Write single FSM2 forcing file for all units.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data with dims (time,) for single unit or (time, unit_id).
        units : list[SpatialUnit]
            Spatial units for simulation.
        output_dir : Path
            Directory for output file.
        filename : str
            Output filename. Default: 'met_input.txt'.

        Returns
        -------
        Path
            Path to written file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename
        times = pd.DatetimeIndex(forcing.time.values)
        n_times = len(times)
        n_units = len(units)

        # Extract and convert data
        # SW = direct + diffuse
        # Ensure dimensions are (time, unit_id) for proper indexing
        def _get_array(da):
            """Get array with shape (time, unit_id)."""
            if "unit_id" not in da.dims:
                return da.values.reshape(-1, 1)
            # Transpose if needed to ensure (time, unit_id) order
            if da.dims[0] == "unit_id":
                return da.values.T
            return da.values

        sw_direct = _get_array(forcing["shortwave_direct"])
        sw_diffuse = _get_array(forcing["shortwave_diffuse"])
        sw_down = sw_direct + sw_diffuse
        lw_down = _get_array(forcing["longwave"])
        snowfall = _get_array(forcing["snowfall"])
        rainfall = _get_array(forcing["rainfall"])
        ta = _get_array(forcing["temperature"])
        rh_da = forcing["humidity_relative"]
        rh = _get_array(rh_da.copy(data=relative_humidity_percent(rh_da)))   # FSM2 reads RH in %
        ua = _get_array(forcing["wind_speed"])
        ps = _get_array(forcing["pressure"])

        # Convert snowfall/rainfall from mm/timestep to kg/m2/s
        # Infer timestep from data (1 mm water = 1 kg/m2)
        if n_times > 1:
            dt_s = (times[1] - times[0]).total_seconds()
        else:
            dt_s = 3600.0
        sf = snowfall / dt_s
        rf = rainfall / dt_s

        # Write file: for each timestep, write Npnts rows
        with open(filepath, "w", encoding="utf-8") as f:
            for t in range(n_times):
                dt = times[t]
                for i in range(n_units):
                    f.write(
                        f"{dt.year:4d} {dt.month:2d} {dt.day:2d} {dt.hour:2d} "
                        f"{sw_down[t, i]:10.4f} {lw_down[t, i]:10.4f} "
                        f"{sf[t, i]:12.6e} {rf[t, i]:12.6e} "
                        f"{ta[t, i]:8.3f} {rh[t, i]:6.4f} "
                        f"{ua[t, i]:8.3f} {ps[t, i]:10.2f}\n"
                    )

        return filepath

    def write_per_unit(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write separate forcing files for each unit (FSM1-style).

        Useful for running FSM2 in single-point mode or for debugging.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data.
        units : list[SpatialUnit]
            Spatial units.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Paths to written files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        written = []
        times = pd.DatetimeIndex(forcing.time.values)
        # Infer timestep from data
        if len(times) > 1:
            dt_s = (times[1] - times[0]).total_seconds()
        else:
            dt_s = 3600.0

        for i, unit in enumerate(units):
            if "unit_id" in forcing.dims:
                ds = forcing.isel(unit_id=i)
            else:
                ds = forcing

            sw_down = ds["shortwave_direct"].values + ds["shortwave_diffuse"].values
            lw_down = ds["longwave"].values
            sf = ds["snowfall"].values / dt_s
            rf = ds["rainfall"].values / dt_s
            ta = ds["temperature"].values
            rh = ds["humidity_relative"].values
            ua = ds["wind_speed"].values
            ps = ds["pressure"].values

            filepath = output_dir / f"met_unit_{unit.id}.txt"

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
