"""Map cluster-based forcing data back to full DEM grid.

This module provides RasterMapper for converting downscaled forcing from
(time, unit_id) format to (time, y, x) format matching the DEM resolution.
Supports output to zarr, NetCDF, and GeoTIFF.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import xarray as xr

if TYPE_CHECKING:
    from topopyscale2.domain import Domain
    from topopyscale2.spatial.units import SpatialUnit


class RasterMapper:
    """Map cluster-based data back to DEM grid.

    Converts forcing or model output from (time, unit_id) dimensions to
    (time, y, x) dimensions for full-resolution visualization and GIS use.

    Parameters
    ----------
    membership : xr.DataArray
        2D raster (y, x) with integer cluster/unit IDs. -1 indicates
        unassigned/nodata pixels.
    units : list[SpatialUnit]
        List of spatial units for unit_id to index mapping.
    crs : str
        Coordinate reference system string for GeoTIFF output.

    Examples
    --------
    >>> from topopyscale2.outputs.raster_mapper import RasterMapper
    >>> mapper = RasterMapper.from_domain(domain)
    >>> raster_ds = mapper.map_to_raster(forcing)
    >>> mapper.write_geotiff_series(forcing, output_dir, "temperature")
    """

    def __init__(
        self,
        membership: xr.DataArray,
        units: list["SpatialUnit"],
        crs: str = "EPSG:4326",
    ):
        self.membership = membership
        self.units = units
        self.crs = crs
        self._unit_id_to_idx: dict[str, int] = {}
        self._build_unit_index()

    def _build_unit_index(self) -> None:
        """Build mapping from unit_id string to array index."""
        self._unit_id_to_idx = {unit.id: i for i, unit in enumerate(self.units)}

    @classmethod
    def from_domain(cls, domain: "Domain") -> "RasterMapper":
        """Create mapper from a Domain object.

        Parameters
        ----------
        domain : Domain
            Domain object with DEM data, units, and membership set up.

        Returns
        -------
        RasterMapper
            Initialized mapper.

        Raises
        ------
        ValueError
            If domain doesn't have membership or units.
        """
        membership = domain.get_membership()
        if membership is None:
            raise ValueError("Domain has no membership. Run setup() first.")

        if domain.units is None:
            raise ValueError("Domain has no units. Run setup() first.")

        # Get CRS from DEM data
        crs = "EPSG:4326"
        if domain.dem_data is not None:
            crs = domain.dem_data.attrs.get("crs", "EPSG:4326")

        return cls(membership, list(domain.units), crs)

    def map_to_raster(
        self,
        forcing: xr.Dataset,
        variables: Optional[list[str]] = None,
    ) -> xr.Dataset:
        """Map forcing from (time, unit_id) to (time, y, x).

        Uses vectorized indexing for efficiency. Unassigned pixels
        (membership == -1) become NaN in the output.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing dataset with dimensions (time, unit_id).
        variables : list[str], optional
            Variables to map. If None, maps all data variables.
            If empty list, no variables are mapped.

        Returns
        -------
        xr.Dataset
            Dataset with dimensions (time, y, x) and same variables.
        """
        vars_to_map = variables if variables is not None else list(forcing.data_vars)
        membership = self.membership.values  # (ny, nx)
        ny, nx = membership.shape

        # Get unit dimension from forcing (could be "unit_id" or "unit")
        if "unit_id" in forcing.dims:
            unit_dim = "unit_id"
            unit_ids = forcing.unit_id.values
        elif "unit" in forcing.dims:
            unit_dim = "unit"
            unit_ids = forcing.unit.values
        else:
            # Fallback: assume sorted by index
            unit_dim = None
            unit_ids = [f"unit_{i:04d}" for i in range(len(self.units))]

        # Build index mapping: membership value -> forcing index
        # membership contains integer IDs (0, 1, 2, ...) that map to unit_ids
        n_units = len(unit_ids)

        result_vars = {}
        for var in vars_to_map:
            if var not in forcing:
                continue

            var_data = forcing[var]
            var_dims = var_data.dims
            has_time = "time" in var_dims

            # Determine dimension order - could be (time, unit) or (unit, time)
            if unit_dim and unit_dim in var_dims:
                unit_axis = var_dims.index(unit_dim)
            else:
                unit_axis = -1

            data = var_data.values

            if has_time:
                time_axis = var_dims.index("time")

                # Transpose to (time, unit) if needed
                if time_axis > unit_axis:
                    # Data is (unit, time) - transpose to (time, unit)
                    data = data.T

                # Now data is (time, n_units) - map to (time, ny, nx)
                safe_membership = np.clip(membership, 0, n_units - 1)
                mapped = data[:, safe_membership]  # (time, ny, nx)
                # Mask unassigned pixels with NaN
                mapped = np.where(
                    membership[None, :, :] >= 0,
                    mapped,
                    np.nan,
                )
                result_vars[var] = (["time", "y", "x"], mapped)
            else:
                safe_membership = np.clip(membership, 0, n_units - 1)
                mapped = data[safe_membership]  # (ny, nx)
                mapped = np.where(membership >= 0, mapped, np.nan)
                result_vars[var] = (["y", "x"], mapped)

        # Build coordinates
        coords = {
            "y": self.membership.y,
            "x": self.membership.x,
        }
        if "time" in forcing.dims:
            coords["time"] = forcing.time

        result = xr.Dataset(result_vars, coords=coords)

        # Copy variable attributes
        for var in vars_to_map:
            if var in forcing and var in result:
                result[var].attrs = forcing[var].attrs.copy()

        # Add CRS info
        result.attrs["crs"] = self.crs

        return result

    def write_geotiff(
        self,
        data: xr.DataArray,
        output_path: Path,
        variable: Optional[str] = None,
        time_idx: Optional[int] = None,
    ) -> Path:
        """Write a single-band GeoTIFF.

        Parameters
        ----------
        data : xr.DataArray
            Data to write. Can be 2D (y, x) or 3D (time, y, x).
        output_path : Path
            Output file path.
        variable : str, optional
            Variable name for metadata tag.
        time_idx : int, optional
            Time index to extract if data is 3D. Required if data is 3D.

        Returns
        -------
        Path
            Path to the written GeoTIFF.

        Raises
        ------
        ImportError
            If rasterio is not available.
        ValueError
            If data is 3D and time_idx is not provided.
        """
        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.transform import from_bounds
        except ImportError:
            raise ImportError("rasterio is required for GeoTIFF output")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Extract 2D slice if needed
        if data.ndim == 3:
            if time_idx is None:
                raise ValueError("time_idx required for 3D data")
            data = data.isel(time=time_idx)

        values = data.values
        ny, nx = values.shape

        # Compute transform from bounds
        x_min = float(data.x.min())
        x_max = float(data.x.max())
        y_min = float(data.y.min())
        y_max = float(data.y.max())

        # Add half-pixel to get cell edges
        if nx > 1:
            dx = (x_max - x_min) / (nx - 1)
            x_min -= dx / 2
            x_max += dx / 2
        if ny > 1:
            dy = (y_max - y_min) / (ny - 1)
            y_min -= dy / 2
            y_max += dy / 2

        transform = from_bounds(x_min, y_min, x_max, y_max, nx, ny)

        # Determine dtype
        dtype = values.dtype
        if np.issubdtype(dtype, np.floating):
            nodata = np.nan
        else:
            nodata = -9999

        profile = {
            "driver": "GTiff",
            "dtype": dtype,
            "count": 1,
            "width": nx,
            "height": ny,
            "crs": CRS.from_string(self.crs),
            "transform": transform,
            "compress": "deflate",
            "tiled": True,
        }

        with rasterio.open(output_path, "w", **profile) as dst:
            # Flip y-axis if needed (GeoTIFF expects top-down)
            if data.y[0] < data.y[-1]:
                values = np.flipud(values)
            dst.write(values, 1)
            dst.set_band_description(1, variable or "data")
            if variable:
                dst.update_tags(variable=variable)

        return output_path

    def write_raster_dataset(
        self,
        forcing: xr.Dataset,
        output_path: Path,
        format: str = "zarr",
        variables: Optional[list[str]] = None,
    ) -> Path:
        """Write full mapped dataset to zarr or NetCDF.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing dataset with (time, unit_id) dimensions.
        output_path : Path
            Output directory (zarr) or file path (NetCDF).
        format : str
            Output format: "zarr" or "netcdf".
        variables : list[str], optional
            Variables to include. If None, includes all.

        Returns
        -------
        Path
            Path to the written output.
        """
        output_path = Path(output_path)

        # Map to raster
        raster_ds = self.map_to_raster(forcing, variables)

        if format == "zarr":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raster_ds.to_zarr(str(output_path), mode="w")
        elif format == "netcdf":
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raster_ds.to_netcdf(str(output_path))
        else:
            raise ValueError(f"Unknown format: {format}. Use 'zarr' or 'netcdf'.")

        return output_path

    def write_geotiff_series(
        self,
        forcing: xr.Dataset,
        output_dir: Path,
        variable: str,
        times: Optional[list] = None,
    ) -> list[Path]:
        """Write one GeoTIFF per timestep.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing dataset with (time, unit_id) dimensions.
        output_dir : Path
            Output directory for GeoTIFF files.
        variable : str
            Variable to export.
        times : list, optional
            Specific times to export. If None, exports all.

        Returns
        -------
        list[Path]
            List of paths to written GeoTIFF files.
        """
        import pandas as pd

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Map the variable to raster
        raster_ds = self.map_to_raster(forcing, [variable])
        data = raster_ds[variable]

        # Get time values
        time_vals = data.time.values if "time" in data.dims else [None]
        if times is not None:
            # Filter to requested times
            time_mask = np.isin(time_vals, times)
            time_indices = np.where(time_mask)[0]
        else:
            time_indices = range(len(time_vals))

        paths = []
        for i in time_indices:
            if "time" in data.dims:
                time_val = pd.Timestamp(data.time.values[i])
                filename = f"{variable}_{time_val.strftime('%Y%m%d_%H%M')}.tif"
                slice_data = data.isel(time=i)
            else:
                filename = f"{variable}.tif"
                slice_data = data

            out_path = output_dir / filename
            self.write_geotiff(slice_data, out_path, variable=variable)
            paths.append(out_path)

        return paths
