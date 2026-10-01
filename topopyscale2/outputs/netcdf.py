"""CF-NetCDF output writer."""

import datetime
from pathlib import Path

import xarray as xr

from topopyscale2.outputs.base import CF_ATTRIBUTES
from topopyscale2.spatial.units import SpatialUnit


class CFNetCDFWriter:
    """Write downscaled forcing as CF-compliant NetCDF."""

    model_name = "generic"
    file_format = "netcdf"

    def __init__(self, compression: bool = True, complevel: int = 4):
        self.compression = compression
        self.complevel = complevel

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write forcing dataset to CF-NetCDF.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data with dims (time, unit_id).
        units : list[SpatialUnit]
            Spatial units with metadata.
        output_dir : Path
            Output directory.

        Returns
        -------
        list[Path]
            Written file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Add CF attributes to variables
        ds = forcing.copy()
        for var in ds.data_vars:
            if var in CF_ATTRIBUTES:
                ds[var].attrs.update(CF_ATTRIBUTES[var])

        # Add unit metadata as coordinates
        if units and "unit_id" in ds.dims:
            unit_ids = [u.id for u in units]
            ds = ds.assign_coords(
                unit_lat=("unit_id", [u.y for u in units]),
                unit_lon=("unit_id", [u.x for u in units]),
                unit_elevation=("unit_id", [u.elevation for u in units]),
                unit_surface_type=("unit_id", [u.surface_type for u in units]),
                unit_area=("unit_id", [u.area_m2 for u in units]),
            )

        # Global CF attributes
        ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": "TopoPyScale 2.0 downscaled meteorological forcing",
            "institution": "TopoPyScale",
            "source": "ERA5 reanalysis, topographically downscaled",
            "history": f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()} by TopoPyScale 2.0",
        })

        # Determine filename from time range
        if "time" in ds.dims and ds.sizes["time"] > 0:
            t_start = str(ds.time.values[0])[:10].replace("-", "")
            t_end = str(ds.time.values[-1])[:10].replace("-", "")
            filename = f"forcing_{t_start}_{t_end}.nc"
        else:
            filename = "forcing.nc"

        output_path = output_dir / filename

        # Encoding with compression
        encoding = {}
        if self.compression:
            for var in ds.data_vars:
                encoding[var] = {
                    "zlib": True,
                    "complevel": self.complevel,
                }

        ds.to_netcdf(output_path, encoding=encoding)
        return [output_path]
