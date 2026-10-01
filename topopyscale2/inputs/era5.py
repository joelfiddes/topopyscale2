"""ERA5 data source — thin wrapper around ecmwf-downloader."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from topopyscale2.config.schema import InputConfig, StorageConfig

log = logging.getLogger(__name__)

# ERA5 native horizontal resolution: 0.25 degrees ~= 31 km at equator
ERA5_RESOLUTION_M = 31000.0

# ── Short name → CDS/Google long name mappings ──────────────────────────
# Used to translate the minimal variable sets from required_era5_variables()
# into the long names expected by download backends.

_SURF_SHORT_TO_LONG = {
    "t2m": "2m_temperature",
    "d2m": "2m_dewpoint_temperature",
    "sp": "surface_pressure",
    "ssrd": "surface_solar_radiation_downwards",
    "strd": "surface_thermal_radiation_downwards",
    "tp": "total_precipitation",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "z_surf": "geopotential",
    "tisr": "toa_incident_solar_radiation",
}

_PLEV_SHORT_TO_LONG = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "r": "relative_humidity",
}


def merge_era5_expver(ds: xr.Dataset) -> xr.Dataset:
    """Merge ERA5 experiment versions (expver) into a single dataset.

    ERA5 data can contain two experiment versions when requests span both
    final (ERA5) and preliminary (ERA5T) data periods:

    - **expver=1 (0001)**: Final ERA5 reanalysis, released ~2-3 months after
      the month in question. Quality-controlled and validated.

    - **expver=5 (0005)**: Preliminary ERA5T ("T" = Temporary), available
      ~5 days behind real-time. May be corrected later.

    When both versions are present, they have **complementary NaN patterns**:
    expver=1 has valid data for older timesteps (NaN for recent), while
    expver=5 has NaN for older timesteps (valid for recent). Together they
    cover the full time range.

    This function merges them by taking the first non-NaN value from either
    version, preferring expver=1 (final) when both are valid.

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 dataset, potentially with 'expver' dimension.

    Returns
    -------
    xr.Dataset
        Dataset with expver dimension removed and data merged.

    Notes
    -----
    The expver dimension appears when:
    1. Requesting data within ~3 months of present (mixed ERA5/ERA5T)
    2. After ECMWF reprocessing campaigns (e.g., snow observation corrections
       in Sept-Dec 2021 and July-Nov 2024)

    References
    ----------
    - https://confluence.ecmwf.int/x/wv2NB (ERA5 data documentation)
    - https://confluence.ecmwf.int/pages/viewpage.action?pageId=173385064

    Examples
    --------
    >>> ds = xr.open_mfdataset("era5_*.nc", preprocess=merge_era5_expver)
    """
    if "expver" not in ds.dims:
        # No expver dimension, just drop coord if present
        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")
        return ds

    if ds.sizes["expver"] == 1:
        # Single expver, just squeeze it out
        ds = ds.isel(expver=0)
        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")
        return ds

    # Multiple expvers: merge by taking first non-NaN value
    # Prefer expver=1 (final ERA5) over expver=5 (preliminary ERA5T)
    ds_merged = ds.isel(expver=0).copy()

    for var in ds.data_vars:
        if "expver" in ds[var].dims:
            v0 = ds[var].isel(expver=0).values  # expver=1 (final)
            v1 = ds[var].isel(expver=1).values  # expver=5 (preliminary)
            # Take v0 where valid, else v1
            merged = np.where(np.isnan(v0), v1, v0)
            ds_merged[var].values = merged

    if "expver" in ds_merged.coords:
        ds_merged = ds_merged.drop_vars("expver")

    return ds_merged


class ERA5Source:
    """ERA5 meteorological data source wrapping ecmwf-downloader."""

    def __init__(self, config: InputConfig, storage_config: StorageConfig | None = None):
        self.config = config
        self.storage_config = storage_config or StorageConfig()

    @property
    def name(self) -> str:
        """Return data source identifier."""
        return "era5"

    @property
    def resolution_m(self) -> float:
        """Return native horizontal resolution in meters."""
        return ERA5_RESOLUTION_M

    def fetch(
        self,
        bbox: tuple[float, float, float, float],
        time_range: list[str],
        required_vars: tuple[set[str], set[str]] | None = None,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Download ERA5 data via ecmwf-downloader and return as datasets.

        Parameters
        ----------
        required_vars : tuple of (surface_short_names, plev_short_names), optional
            If provided, only fetch these ERA5 variables (short names).
            None means fetch all variables (backward compatible default).

        Returns (ds_surface, ds_pressure).
        """
        from topopyscale2.inputs.nwp_downloader import ERA5Loader

        # Google ARCO-ERA5 doesn't have relative_humidity, so exclude it
        # and let compute_rh=True compute it from specific_humidity + temperature
        backend_kwargs = {}
        if self.config.backend == "google":
            # Default: all variables
            all_surf = [
                "geopotential",
                "2m_dewpoint_temperature",
                "surface_thermal_radiation_downwards",
                "surface_solar_radiation_downwards",
                "surface_pressure",
                "total_precipitation",
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "toa_incident_solar_radiation",
            ]
            all_plev = [
                "geopotential",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                # Note: relative_humidity excluded - will be computed
            ]

            # Filter to minimal set if required_vars is specified
            if required_vars is not None:
                surf_short, plev_short = required_vars
                all_surf = self._filter_long_names(surf_short, _SURF_SHORT_TO_LONG)
                all_plev = self._filter_long_names(plev_short, _PLEV_SHORT_TO_LONG)
                log.info(
                    "Variable filtering: surface=%s, pressure=%s",
                    sorted(surf_short), sorted(plev_short),
                )

            backend_kwargs = {
                "plev_vars": all_plev,
                "surf_vars": all_surf,
            }
        elif self.config.backend == "hybrid":
            # Hybrid backend needs start/end dates for regional s3zarr auto-detection
            backend_kwargs = {
                "start_date": time_range[0],
                "end_date": time_range[1],
                "regional_zarr_url": self.config.s3_zarr_url,
                "precip_model": getattr(self.config, "hybrid_precip_model", "era5"),
            }
        elif self.config.backend == "s3zarr":
            # S3 Zarr backend - needs inputs.s3_zarr_url (no built-in store)
            backend_kwargs = {}
            if self.config.s3_zarr_url:
                backend_kwargs["zarr_url"] = self.config.s3_zarr_url
        elif self.config.backend == "local":
            # Local Zarr cache backend (built by `tps2 build-cache`)
            if not self.config.cache_path:
                raise ValueError(
                    "backend='local' requires cache_path in config "
                    "(path to Zarr store built by `tps2 build-cache`)"
                )
            backend_kwargs = {"cache_path": str(self.config.cache_path)}

        # For non-Google backends, add surf_vars/plev_vars if filtering requested
        if required_vars is not None and self.config.backend not in ("google",):
            surf_short, plev_short = required_vars
            if self.config.backend == "local":
                # Local Zarr cache uses short names (t2m, sp, t, z, ...)
                filtered_surf = list(surf_short)
                filtered_plev = list(plev_short)
            else:
                filtered_surf = self._filter_long_names(surf_short, _SURF_SHORT_TO_LONG)
                filtered_plev = self._filter_long_names(plev_short, _PLEV_SHORT_TO_LONG)
            if filtered_surf:
                backend_kwargs.setdefault("surf_vars", filtered_surf)
            if filtered_plev:
                backend_kwargs.setdefault("plev_vars", filtered_plev)
            log.info(
                "Variable filtering: surface=%s, pressure=%s",
                sorted(surf_short), sorted(plev_short),
            )

        # Use storage config for cache directory if specified
        cache_dir = str(self.storage_config.forcing_cache)

        loader = ERA5Loader(
            backend=self.config.backend,
            bbox=bbox,
            start_date=time_range[0],
            end_date=time_range[1],
            pressure_levels=self.config.pressure_levels,
            output_format=self.config.output_format,
            output_dir=cache_dir,
            time_resolution=self.config.time_resolution,
            max_workers=self.config.max_workers,
            compute_rh=True,  # Compute RH from specific humidity and temperature
            backend_kwargs=backend_kwargs,
        )
        loader.download()

        # Open the downloaded files
        ds_surface, ds_pressure = self._open_downloaded(time_range)
        return ds_surface, ds_pressure

    @staticmethod
    def _filter_long_names(
        short_names: set[str],
        mapping: dict[str, str],
    ) -> list[str]:
        """Convert ERA5 short names to CDS/Google long names, deduplicating."""
        long_names = []
        seen = set()
        for short in sorted(short_names):
            long = mapping.get(short)
            if long and long not in seen:
                long_names.append(long)
                seen.add(long)
        return long_names

    def _open_downloaded(
        self, time_range: list[str]
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Open downloaded NetCDF/Zarr files from cache directory."""
        cache = Path(self.storage_config.forcing_cache)

        if self.config.output_format == "zarr":
            ds = self._open_zarr_cache(cache)
            # Slice to requested time range to avoid loading entire cache
            if time_range and "time" in ds.dims:
                ds = ds.sel(time=slice(time_range[0], time_range[1]))
            # Split surface and pressure variables
            surf_vars = {"t2m", "d2m", "sp", "ssrd", "strd", "tp", "z_surf", "u10", "v10", "tisr"}
            plev_vars = {"t", "z", "u", "v", "q", "r"}

            surf_available = [v for v in surf_vars if v in ds]
            plev_available = [v for v in plev_vars if v in ds]

            ds_surface = merge_era5_expver(ds[surf_available])
            ds_pressure = merge_era5_expver(ds[plev_available])
        else:
            # NetCDF: look for yearly files
            start_year = int(time_range[0][:4])
            end_year = int(time_range[1][:4])

            surf_files = []
            plev_files = []
            for year in range(start_year, end_year + 1):
                sf = cache / "yearly" / f"SURF_{year}.nc"
                pf = cache / "yearly" / f"PLEV_{year}.nc"
                if sf.exists():
                    surf_files.append(sf)
                if pf.exists():
                    plev_files.append(pf)

            if not surf_files:
                # Try daily files
                from glob import glob

                surf_files = sorted(glob(str(cache / "daily" / "dSURF_*.nc")))
                plev_files = sorted(glob(str(cache / "daily" / "dPLEV_*.nc")))

            ds_surface = merge_era5_expver(xr.open_mfdataset(surf_files)) if surf_files else xr.Dataset()
            ds_pressure = merge_era5_expver(xr.open_mfdataset(plev_files)) if plev_files else xr.Dataset()

        return ds_surface, ds_pressure

    def _open_zarr_cache(self, cache: Path) -> xr.Dataset:
        """Open Zarr cache, handling both merged and per-day stores.

        Prefers merged ERA5.zarr if it exists. Falls back to opening
        per-day stores via open_mfdataset.
        """
        # Try merged store first
        zarr_path = cache / "ERA5.zarr"
        if zarr_path.exists():
            return xr.open_zarr(zarr_path)

        # Try per-day stores
        daily_dir = cache / "daily"
        if daily_dir.exists():
            day_stores = sorted(daily_dir.glob("day_*.zarr"))
            if day_stores:
                return xr.open_mfdataset(
                    [str(p) for p in day_stores],
                    engine="zarr",
                    combine="by_coords",
                    parallel=True,
                )

        raise FileNotFoundError(
            f"No Zarr data found in {cache}. "
            f"Expected ERA5.zarr or daily/day_*.zarr stores."
        )

    def compute_lapse_rate(
        self, ds_plev: xr.Dataset, z_surface: xr.DataArray
    ) -> xr.DataArray:
        """Compute environmental lapse rate from pressure level data.

        Delegates to the vectorized implementation in
        ``topopyscale2.inputs.derived``.

        Returns gamma [K/m] as DataArray(time, latitude, longitude).
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

        Parameters
        ----------
        time : array of datetime64
        lat : array of latitudes [degrees]
        lon : array of longitudes [degrees]

        Returns
        -------
        tuple of (solar_elevation, solar_azimuth) DataArrays in degrees.
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
        """Compute clearness index kt = ssrd / sw_toa.

        Parameters
        ----------
        ds_surf : xr.Dataset
            Surface dataset with 'ssrd' variable.
        solar_elevation : xr.DataArray
            Solar elevation angles [degrees].

        Returns
        -------
        xr.DataArray
            Clearness index kt, clamped to [0, 1].
        """
        SOLAR_CONSTANT = 1361.0  # W/m2

        sin_elev = np.sin(np.radians(solar_elevation.values))
        sw_toa = SOLAR_CONSTANT * np.maximum(sin_elev, 0.0)

        ssrd = ds_surf["ssrd"].values

        # Avoid division by zero
        kt = np.where(sw_toa > 1.0, ssrd / sw_toa, 0.0)
        kt = np.clip(kt, 0.0, 1.0)

        return xr.DataArray(
            kt,
            dims=solar_elevation.dims,
            coords=solar_elevation.coords,
            attrs={"units": "-", "long_name": "Clearness index"},
        )
