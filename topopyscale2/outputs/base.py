"""Output writer base definitions and CF attributes."""

from enum import Enum
from pathlib import Path
from typing import Protocol

import xarray as xr

from topopyscale2.spatial.units import SpatialUnit


class OutputVariable(Enum):
    """Standard output variables."""

    TEMPERATURE = "temperature"
    PRECIPITATION = "precipitation"
    RAINFALL = "rainfall"
    SNOWFALL = "snowfall"
    SHORTWAVE_DIRECT = "shortwave_direct"
    SHORTWAVE_DIFFUSE = "shortwave_diffuse"
    LONGWAVE = "longwave"
    HUMIDITY_SPECIFIC = "humidity_specific"
    HUMIDITY_RELATIVE = "humidity_relative"
    WIND_SPEED = "wind_speed"
    PRESSURE = "pressure"


# CF-compliant variable attributes
CF_ATTRIBUTES = {
    "temperature": {
        "standard_name": "air_temperature",
        "long_name": "Near-surface air temperature",
        "units": "K",
    },
    "precipitation": {
        "standard_name": "precipitation_amount",
        "long_name": "Total precipitation",
        "units": "kg m-2",
    },
    "rainfall": {
        "standard_name": "rainfall_amount",
        "long_name": "Rainfall",
        "units": "kg m-2",
    },
    "snowfall": {
        "standard_name": "snowfall_amount",
        "long_name": "Snowfall",
        "units": "kg m-2",
    },
    "shortwave_direct": {
        "standard_name": "surface_direct_downwelling_shortwave_flux_in_air",
        "long_name": "Direct shortwave radiation on slope",
        "units": "W m-2",
    },
    "shortwave_diffuse": {
        "standard_name": "surface_diffuse_downwelling_shortwave_flux_in_air",
        "long_name": "Diffuse shortwave radiation",
        "units": "W m-2",
    },
    "longwave": {
        "standard_name": "surface_downwelling_longwave_flux_in_air",
        "long_name": "Downwelling longwave radiation",
        "units": "W m-2",
    },
    "humidity_specific": {
        "standard_name": "specific_humidity",
        "long_name": "Specific humidity",
        "units": "kg kg-1",
    },
    "humidity_relative": {
        "standard_name": "relative_humidity",
        "long_name": "Relative humidity",
        "units": "1",
    },
    "wind_speed": {
        "standard_name": "wind_speed",
        "long_name": "Wind speed",
        "units": "m s-1",
    },
    "wind_direction": {
        "standard_name": "wind_from_direction",
        "long_name": "Wind direction (from)",
        "units": "degree",
    },
    "pressure": {
        "standard_name": "surface_air_pressure",
        "long_name": "Surface air pressure",
        "units": "Pa",
    },
}


class OutputWriter(Protocol):
    """Protocol for output file writers."""

    model_name: str
    file_format: str

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write forcing to disk."""
        ...


def installed_engine_ref() -> str:
    """Version of the running TPS2, plus its commit when run from a git checkout."""
    import subprocess
    from pathlib import Path

    import topopyscale2

    ref = getattr(topopyscale2, "__version__", "unknown")
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(topopyscale2.__file__).parent,
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        ref += f" @ {sha}"
    except (OSError, subprocess.SubprocessError):
        pass
    return ref


def with_cf_metadata(ds, *, engine_ref: str | None = None, run_date: str | None = None):
    """``ds`` with CF variable attributes (units, names) and TPS2 provenance attributes.

    Forcing written without these is ambiguous: relative humidity is a 0-1 fraction and
    precipitation an amount per time step, neither of which a reader can tell from a bare
    number. Variables not in ``CF_ATTRIBUTES`` are left as they are.
    """
    from datetime import datetime, timezone

    ds = ds.copy()
    for name in ds.data_vars:
        if name in CF_ATTRIBUTES:
            ds[name].attrs.update(CF_ATTRIBUTES[name])
    ds.attrs.update({
        "Conventions": "CF-1.8",
        "source": "TopoPyScale 2 downscaled forcing",
        "tps2_engine": engine_ref or ds.attrs.get("tps2_engine") or installed_engine_ref(),
        "tps2_run_date": run_date or ds.attrs.get("tps2_run_date")
        or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    })
    return ds
