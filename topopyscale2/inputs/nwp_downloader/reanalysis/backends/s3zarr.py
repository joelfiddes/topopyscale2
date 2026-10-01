"""S3 Zarr backend for ERA5 data.

Opens pre-aggregated regional ERA5 zarr stores on S3-compatible storage.
Auth via environment variables: AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY.

Ported from: era5google/makedatasets2.py + TopoPyScale/fetch_era5_zarr.py
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import xarray as xr

try:
    import dotenv
    _HAS_DOTENV = True
except ImportError:
    _HAS_DOTENV = False

import numpy as np

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox

logger = logging.getLogger(__name__)


# Default variable groupings expected in the zarr store
DEFAULT_SURF_VARS = ["d2m", "sp", "ssrd", "strd", "t2m", "tp", "z_surf"]
DEFAULT_PLEV_VARS = ["q", "t", "u", "v", "z"]


class S3ZarrBackend(ERA5Backend):
    """ERA5 backend reading from a pre-built zarr store on S3.

    The zarr store is expected to contain both surface and pressure-level
    variables with standard short names and dimensions
    (time, latitude, longitude, level).

    Auth is provided via environment variables:
        - AWS_ENDPOINT_URL
        - AWS_ACCESS_KEY_ID
        - AWS_SECRET_ACCESS_KEY

    Args:
        bbox: Bounding box (west, south, east, north).
        pressure_levels: Pressure levels in hPa.
        time_resolution: '1H', '2H', '3H', or '6H'.
        zarr_url: URL to the zarr store (e.g. 's3://bucket/era5.zarr/').
        surf_vars: Surface variable names in the store.
        plev_vars: Pressure-level variable names in the store.
        storage_options: Additional kwargs for s3fs (overrides env vars).
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int],
        time_resolution: str = "1H",
        zarr_url: str = "",
        surf_vars: list[str] | None = None,
        plev_vars: list[str] | None = None,
        storage_options: dict | None = None,
        **kwargs,
    ):
        super().__init__(bbox, pressure_levels, time_resolution, **kwargs)

        if not zarr_url:
            raise ValueError(
                "The s3zarr backend needs the URL of a pre-built ERA5 Zarr store: set "
                "inputs.s3_zarr_url in the config (e.g. 's3://bucket/era5.zarr/'). "
                "There is no built-in store; `tps2 build-cache` makes a local one."
            )

        self.zarr_url = zarr_url
        self.surf_vars = surf_vars or DEFAULT_SURF_VARS
        self.plev_vars = plev_vars or DEFAULT_PLEV_VARS

        # Build storage options from environment variables if not provided
        self._storage_options = storage_options or {}
        if not self._storage_options:
            # Load .env file from current directory if dotenv is available
            if _HAS_DOTENV:
                dotenv.load_dotenv()

            # Non-AWS object stores (e.g. an OpenStack/Ceph S3 endpoint) are set
            # through AWS_ENDPOINT_URL, alongside the usual AWS_* credentials.
            endpoint_url = os.environ.get("AWS_ENDPOINT_URL")

            if endpoint_url:
                self._storage_options["client_kwargs"] = {"endpoint_url": endpoint_url}

            logger.debug(
                "Using S3 storage options: endpoint=%s",
                endpoint_url,
            )

        # Open the zarr store once and keep it open
        self._ds = xr.open_zarr(
            self.zarr_url,
            storage_options=self._storage_options,
            chunks="auto",
        )

        # Pre-select spatial subset
        self._ds = self._ds.sel(
            longitude=slice(self.bbox.west, self.bbox.east),
            latitude=slice(self.bbox.north, self.bbox.south),
        )

        logger.info(
            "Opened S3 zarr store: %s (vars=%s)",
            self.zarr_url,
            list(self._ds.data_vars),
        )

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data from the zarr store.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev) with standardised names and dims.
        """
        date = pd.Timestamp(date)
        start = date.normalize()
        end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        logger.info("Fetching S3 zarr data for %s", date.strftime("%Y-%m-%d"))

        ds_day = self._ds.sel(time=slice(start, end))

        if ds_day.sizes["time"] == 0:
            raise ValueError(f"No data available for {date.strftime('%Y-%m-%d')}")

        # Split into surface and pressure-level datasets (lazy)
        available_surf = [v for v in self.surf_vars if v in ds_day]
        available_plev = [v for v in self.plev_vars if v in ds_day]

        ds_surf_lazy = ds_day[available_surf]
        ds_plev_lazy = ds_day[available_plev]

        # Select pressure levels (only those available in store)
        if "level" in ds_plev_lazy.dims and self.pressure_levels:
            available_levels = ds_plev_lazy.level.values.tolist()
            requested = [lv for lv in self.pressure_levels if lv in available_levels]
            if not requested:
                logger.warning(
                    "No requested levels %s found in store levels %s",
                    self.pressure_levels, available_levels
                )
            else:
                if len(requested) < len(self.pressure_levels):
                    missing = [lv for lv in self.pressure_levels if lv not in available_levels]
                    logger.warning("Levels %s not in store, using %s", missing, requested)
                ds_plev_lazy = ds_plev_lazy.sel(level=requested)

        # Load surface and pressure-level data concurrently from S3
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_surf = executor.submit(ds_surf_lazy.load)
            fut_plev = executor.submit(ds_plev_lazy.load)
            ds_surf = fut_surf.result()
            ds_plev = fut_plev.result()

        # Rename z_surf back to z for surface dataset (matching convention)
        if "z_surf" in ds_surf:
            ds_surf = ds_surf.rename({"z_surf": "z"})

        # Scale precipitation for subsampled data.
        # The zarr store contains hourly ERA5 data subsampled to N-hourly,
        # so each tp value is a 1-hour accumulation.  Multiply by the step
        # ratio so tp represents the total for the timestep interval.
        if "tp" in ds_surf and ds_surf.sizes["time"] >= 2:
            deltas = np.diff(ds_surf.time.values[:min(8, ds_surf.sizes["time"])])
            step_hours = float(np.median(deltas) / np.timedelta64(1, "h"))
            if step_hours > 1.0:
                ds_surf["tp"] = ds_surf["tp"] * step_hours
                logger.info(
                    "Scaled tp by %.0fx for %.0fH subsampled data (approximation)",
                    step_hours, step_hours,
                )

        # Sort pressure levels ascending
        if "level" in ds_plev.dims:
            ds_plev = ds_plev.sortby("level", ascending=True)

        logger.info(
            "Fetched S3 zarr: SURF vars=%s, PLEV vars=%s",
            list(ds_surf.data_vars),
            list(ds_plev.data_vars),
        )

        return ds_surf, ds_plev

    def close(self):
        """Close the zarr store."""
        if hasattr(self, "_ds") and self._ds is not None:
            self._ds.close()
            self._ds = None
