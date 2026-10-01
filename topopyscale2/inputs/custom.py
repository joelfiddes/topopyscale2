"""Custom NWP data source loader.

Loads user-provided NetCDF/Zarr files with configurable variable name mapping.
Useful for incorporating regional/national NWP products, station interpolations,
or custom forcing datasets.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

from topopyscale2.config.schema import InputConfig, NWPSourceConfig
from topopyscale2.inputs.base import STANDARD_VARIABLES


class CustomSource:
    """Custom NWP data source from user-provided files.

    Loads NetCDF or Zarr files and maps variable names to TopoPyScale 2.0
    standard names using a configurable mapping dictionary.

    Parameters
    ----------
    config : InputConfig
        General input configuration.
    source_config : NWPSourceConfig
        Source-specific configuration with path and variable_mapping.
        The path can point to a single file, a directory with multiple files,
        or a Zarr store.

    Examples
    --------
    >>> from topopyscale2.config.schema import InputConfig, NWPSourceConfig
    >>> source_config = NWPSourceConfig(
    ...     name="my_nwp",
    ...     type="custom",
    ...     path=Path("/data/my_model/"),
    ...     resolution_m=2000.0,
    ...     variable_mapping={
    ...         "temperature_2m": "t2m",
    ...         "dewpoint_2m": "d2m",
    ...         "surface_pressure": "sp",
    ...     }
    ... )
    >>> source = CustomSource(InputConfig(), source_config)
    """

    def __init__(
        self,
        config: InputConfig,
        source_config: NWPSourceConfig,
    ):
        if source_config is None:
            raise ValueError("CustomSource requires source_config with path.")
        if source_config.path is None:
            raise ValueError("CustomSource requires source_config.path to be set.")

        self.config = config
        self.source_config = source_config
        self._path = Path(source_config.path)
        self._variable_mapping = source_config.variable_mapping or {}

    @property
    def name(self) -> str:
        """Return data source identifier."""
        return self.source_config.name

    @property
    def resolution_m(self) -> float:
        """Return native horizontal resolution in meters."""
        if self.source_config.resolution_m:
            return self.source_config.resolution_m
        # Default to 10km if not specified
        return 10000.0

    @property
    def path(self) -> Path:
        """Return the data path."""
        return self._path

    @property
    def variable_mapping(self) -> dict[str, str]:
        """Return the variable name mapping (source -> standard)."""
        return self._variable_mapping

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        time_range: list[str],
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Load custom data files and return as datasets.

        Parameters
        ----------
        bbox : tuple[float, float, float, float]
            Bounding box as (west, south, east, north) in degrees.
            Used to subset the data if it covers a larger area.
        time_range : list[str]
            Time range as [start_date, end_date] in ISO format.
            Used to subset the data.

        Returns
        -------
        tuple[xr.Dataset, xr.Dataset]
            Surface and pressure level datasets with standardized variable names.

        Raises
        ------
        FileNotFoundError
            If the specified path doesn't exist.
        ValueError
            If required variables are missing.
        """
        if not self._path.exists():
            raise FileNotFoundError(f"Custom data path not found: {self._path}")

        # Load the dataset(s)
        ds = self._load_data()

        # Apply variable mapping
        ds = self._apply_mapping(ds)

        # Subset by bbox and time
        ds = self._subset(ds, bbox, time_range)

        # Split into surface and pressure level datasets
        ds_surface, ds_pressure = self._split_surface_pressure(ds)

        return ds_surface, ds_pressure

    def _load_data(self) -> xr.Dataset:
        """Load data from the configured path."""
        if self._path.is_dir():
            # Check if it's a Zarr store (v2 or v3)
            # Zarr v2: .zarray, .zgroup files
            # Zarr v3: zarr.json file
            is_zarr = (
                (self._path / ".zarray").exists()
                or (self._path / ".zgroup").exists()
                or (self._path / "zarr.json").exists()
            )
            if is_zarr:
                return xr.open_zarr(self._path)
            # Check if it looks like a Zarr store by name
            if self._path.suffix == ".zarr" or str(self._path).endswith(".zarr"):
                return xr.open_zarr(self._path)
            # Otherwise, try to open as multi-file NetCDF
            nc_files = list(self._path.glob("*.nc"))
            if not nc_files:
                raise FileNotFoundError(f"No NetCDF files found in {self._path}")
            return xr.open_mfdataset(nc_files)
        else:
            # Single file
            if self._path.suffix == ".zarr":
                return xr.open_zarr(self._path)
            else:
                return xr.open_dataset(self._path)

    def _apply_mapping(self, ds: xr.Dataset) -> xr.Dataset:
        """Apply variable name mapping to standardize names."""
        if not self._variable_mapping:
            return ds

        # Create reverse mapping for efficiency
        rename_dict = {}
        for source_name, standard_name in self._variable_mapping.items():
            if source_name in ds.data_vars:
                rename_dict[source_name] = standard_name

        if rename_dict:
            ds = ds.rename(rename_dict)

        return ds

    def _subset(
        self,
        ds: xr.Dataset,
        bbox: tuple[float, float, float, float],
        time_range: list[str],
    ) -> xr.Dataset:
        """Subset dataset by bbox and time range."""
        west, south, east, north = bbox

        # Find the coordinate names for latitude/longitude
        lat_coord = self._find_coord(ds, ["latitude", "lat", "y"])
        lon_coord = self._find_coord(ds, ["longitude", "lon", "x"])

        # Subset spatially if coordinates exist
        if lat_coord and lon_coord:
            lat_vals = ds[lat_coord].values
            lon_vals = ds[lon_coord].values

            # Handle descending latitude
            if lat_vals[0] > lat_vals[-1]:
                lat_slice = slice(north, south)
            else:
                lat_slice = slice(south, north)

            lon_slice = slice(west, east)

            ds = ds.sel({lat_coord: lat_slice, lon_coord: lon_slice})

        # Subset temporally
        if "time" in ds.dims and time_range:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))

        return ds

    def _find_coord(self, ds: xr.Dataset, names: list[str]) -> Optional[str]:
        """Find coordinate by checking multiple possible names."""
        for name in names:
            if name in ds.coords or name in ds.dims:
                return name
        return None

    def _split_surface_pressure(
        self, ds: xr.Dataset
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Split dataset into surface and pressure level variables."""
        surf_vars = {"t2m", "d2m", "sp", "ssrd", "strd", "tp", "u10", "v10", "z_surf"}
        plev_vars = {"t", "z", "u", "v", "q", "r"}

        # Find which standard variables are present
        surf_present = [v for v in surf_vars if v in ds.data_vars]
        plev_present = [v for v in plev_vars if v in ds.data_vars]

        ds_surface = ds[surf_present] if surf_present else xr.Dataset()
        ds_pressure = ds[plev_present] if plev_present else xr.Dataset()

        return ds_surface, ds_pressure

    def compute_lapse_rate(
        self, ds_plev: xr.Dataset, z_surface: xr.DataArray
    ) -> xr.DataArray:
        """Compute environmental lapse rate from pressure level data.

        Delegates to the vectorized implementation in
        ``topopyscale2.inputs.derived``.
        """
        from topopyscale2.inputs.derived import compute_lapse_rate as _compute_lr

        return _compute_lr(ds_plev)

    def compute_solar_geometry(
        self,
        time: np.ndarray,
        lat: np.ndarray,
        lon: np.ndarray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute solar elevation and azimuth angles.

        Delegates to the vectorized implementation in
        ``topopyscale2.inputs.derived``.
        """
        from topopyscale2.inputs.derived import compute_solar_geometry as _compute_sg

        solar_elev, solar_az = _compute_sg(time, lat, lon)

        coords = {"time": time, "latitude": lat, "longitude": lon}
        dims = ["time", "latitude", "longitude"]

        elev_da = xr.DataArray(solar_elev, dims=dims, coords=coords, attrs={"units": "degrees"})
        az_da = xr.DataArray(solar_az, dims=dims, coords=coords, attrs={"units": "degrees"})

        return elev_da, az_da

    def compute_clearness_index(
        self,
        ds_surf: xr.Dataset,
        solar_elevation: xr.DataArray,
    ) -> xr.DataArray:
        """Compute clearness index kt = ssrd / sw_toa."""
        SOLAR_CONSTANT = 1361.0  # W/m2

        sin_elev = np.sin(np.radians(solar_elevation.values))
        sw_toa = SOLAR_CONSTANT * np.maximum(sin_elev, 0.0)

        ssrd = ds_surf["ssrd"].values

        kt = np.where(sw_toa > 1.0, ssrd / sw_toa, 0.0)
        kt = np.clip(kt, 0.0, 1.0)

        return xr.DataArray(
            kt,
            dims=solar_elevation.dims,
            coords=solar_elevation.coords,
            attrs={"units": "-", "long_name": "Clearness index"},
        )


def validate_variable_mapping(
    mapping: dict[str, str], required_vars: Optional[list[str]] = None
) -> list[str]:
    """Validate a variable mapping against standard names.

    Parameters
    ----------
    mapping : dict[str, str]
        Variable mapping from source names to standard names.
    required_vars : list[str], optional
        List of standard variable names that must be present.

    Returns
    -------
    list[str]
        List of validation warnings (empty if all OK).

    Raises
    ------
    ValueError
        If required variables are missing from the mapping.
    """
    warnings = []

    # Check that target names are valid standard names
    for source_name, standard_name in mapping.items():
        if standard_name not in STANDARD_VARIABLES:
            warnings.append(
                f"'{standard_name}' (mapped from '{source_name}') is not a "
                f"recognized standard variable name."
            )

    # Check required variables
    if required_vars:
        mapped_standard_names = set(mapping.values())
        missing = set(required_vars) - mapped_standard_names
        if missing:
            raise ValueError(
                f"Required variables missing from mapping: {missing}. "
                f"These are needed for downscaling."
            )

    return warnings
