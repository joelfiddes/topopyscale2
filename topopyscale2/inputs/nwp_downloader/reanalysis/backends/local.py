"""Local Zarr cache backend for ERA5 data.

Reads from a pre-built regional ERA5 Zarr cache on local or mounted disk
(SSHFS, NFS, USB). Built by ``tps2 build-cache``.

Modelled on S3ZarrBackend but without S3/auth — just ``xr.open_zarr(path)``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox

logger = logging.getLogger(__name__)

# Default variable groupings (matching ZarrWriter convention)
DEFAULT_SURF_VARS = ["d2m", "sp", "ssrd", "strd", "t2m", "tp", "z_surf", "u10", "v10", "tisr"]
DEFAULT_PLEV_VARS = ["q", "t", "u", "v", "z"]


class LocalCacheBackend(ERA5Backend):
    """ERA5 backend reading from a local Zarr cache store.

    The store is expected to contain both surface and pressure-level
    variables with standard short names and dimensions
    (time, latitude, longitude, level).

    Cache stores are built by ``tps2 build-cache`` and contain metadata
    attrs (``tps2_surf_vars``, ``tps2_plev_vars``) describing the
    variable layout.

    Args:
        bbox: Bounding box (west, south, east, north).
        pressure_levels: Pressure levels in hPa.
        time_resolution: '1H', '2H', '3H', or '6H'.
        cache_path: Path to the Zarr store directory.
        surf_vars: Surface variable names (overrides store attrs).
        plev_vars: Pressure-level variable names (overrides store attrs).
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int],
        time_resolution: str = "1H",
        cache_path: str = "",
        surf_vars: list[str] | None = None,
        plev_vars: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(bbox, pressure_levels, time_resolution, **kwargs)

        if not cache_path:
            raise ValueError("cache_path is required for the local backend")

        self.cache_path = cache_path

        # Open the Zarr store once and keep it open
        self._ds = xr.open_zarr(self.cache_path, chunks="auto")

        # Read variable lists from store attrs, falling back to defaults
        if surf_vars is not None:
            self.surf_vars = surf_vars
        elif "tps2_surf_vars" in self._ds.attrs:
            self.surf_vars = list(self._ds.attrs["tps2_surf_vars"])
        else:
            self.surf_vars = DEFAULT_SURF_VARS

        if plev_vars is not None:
            self.plev_vars = plev_vars
        elif "tps2_plev_vars" in self._ds.attrs:
            self.plev_vars = list(self._ds.attrs["tps2_plev_vars"])
        else:
            self.plev_vars = DEFAULT_PLEV_VARS

        # Pre-select spatial subset
        self._ds = self._ds.sel(
            longitude=slice(self.bbox.west, self.bbox.east),
            latitude=slice(self.bbox.north, self.bbox.south),
        )

        # Validate requested pressure levels exist
        if "level" in self._ds.dims:
            available = sorted(self._ds.level.values.tolist())
            missing = [lv for lv in self.pressure_levels if lv not in available]
            if missing:
                logger.warning(
                    "Pressure levels %s not in cache (available: %s)",
                    missing, available,
                )

        logger.info(
            "Opened local cache: %s (vars=%s)",
            self.cache_path,
            list(self._ds.data_vars),
        )

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data from the local cache.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev) with standardised names and dims.
        """
        date = pd.Timestamp(date)
        start = date.normalize()
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        logger.info("Fetching local cache data for %s", date.strftime("%Y-%m-%d"))

        ds_day = self._ds.sel(time=slice(start, end))

        if ds_day.sizes["time"] == 0:
            raise FileNotFoundError(
                f"No data in cache for {date.strftime('%Y-%m-%d')}"
            )

        # Split into surface and pressure-level datasets (lazy)
        available_surf = [v for v in self.surf_vars if v in ds_day]
        available_plev = [v for v in self.plev_vars if v in ds_day]

        ds_surf_lazy = ds_day[available_surf]
        ds_plev_lazy = ds_day[available_plev]

        # Select requested pressure levels
        if "level" in ds_plev_lazy.dims and self.pressure_levels:
            store_levels = ds_plev_lazy.level.values.tolist()
            requested = [lv for lv in self.pressure_levels if lv in store_levels]
            if not requested:
                logger.warning(
                    "No requested levels %s found in cache levels %s",
                    self.pressure_levels, store_levels,
                )
            else:
                if len(requested) < len(self.pressure_levels):
                    missing = [lv for lv in self.pressure_levels if lv not in store_levels]
                    logger.warning("Levels %s not in cache, using %s", missing, requested)
                ds_plev_lazy = ds_plev_lazy.sel(level=requested)

        # Load surface and pressure-level data concurrently
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_surf = executor.submit(ds_surf_lazy.load)
            fut_plev = executor.submit(ds_plev_lazy.load)
            ds_surf = fut_surf.result()
            ds_plev = fut_plev.result()

        # Rename z_surf back to z for surface dataset (matching convention)
        if "z_surf" in ds_surf:
            ds_surf = ds_surf.rename({"z_surf": "z"})

        # Handle time resolution subsampling: if cache is finer than requested,
        # subsample and scale accumulated precipitation accordingly
        if "tp" in ds_surf and ds_surf.sizes["time"] >= 2:
            deltas = np.diff(ds_surf.time.values[:min(8, ds_surf.sizes["time"])])
            step_hours = float(np.median(deltas) / np.timedelta64(1, "h"))
            if step_hours > 1.0:
                ds_surf["tp"] = ds_surf["tp"] * step_hours
                logger.info(
                    "Scaled tp by %.0fx for %.0fH data",
                    step_hours, step_hours,
                )

        # Ensure consistent coordinate ordering for downstream combine
        if "latitude" in ds_surf.dims:
            ds_surf = ds_surf.sortby("latitude")
            ds_plev = ds_plev.sortby("latitude")

        # Sort pressure levels ascending
        if "level" in ds_plev.dims:
            ds_plev = ds_plev.sortby("level", ascending=True)

        logger.info(
            "Fetched local cache: SURF vars=%s, PLEV vars=%s",
            list(ds_surf.data_vars),
            list(ds_plev.data_vars),
        )

        return ds_surf, ds_plev

    def probe_date(self, date: pd.Timestamp) -> bool:
        """Check if a date exists in the cache time coordinate."""
        date = pd.Timestamp(date).normalize()
        end = date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        return self._ds.sel(time=slice(date, end)).sizes["time"] > 0

    def close(self):
        """Close the Zarr store."""
        if hasattr(self, "_ds") and self._ds is not None:
            self._ds.close()
            self._ds = None
