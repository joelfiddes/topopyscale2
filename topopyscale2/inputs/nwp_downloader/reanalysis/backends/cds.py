"""CDS API backend for ERA5 reanalysis data.

Downloads ERA5 (and ERA5T) data via the Copernicus Climate Data Store API.
Requires ``cdsapi`` and a valid ``~/.cdsapirc`` configuration.

Ported from: TopoPyScale/fetch_era5.py
"""

from __future__ import annotations

import logging
import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import xarray as xr
from tqdm import tqdm

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.variables import (
    CDS_DROP_VARS,
    CDS_RENAME_PLEV,
    CDS_RENAME_SURF,
    DEFAULT_PLEV_VARS,
    DEFAULT_SURF_VARS,
    TIME_RESOLUTION_STRINGS,
)

logger = logging.getLogger(__name__)


def _unzip_cds_file(file_path: str) -> None:
    """Handle CDS quirk: zip file disguised as netCDF.

    The new CDS API sometimes sends zip files with a .nc extension.
    This detects that case and re-packs as proper netCDF.
    """
    # Try opening as netCDF first
    try:
        with xr.open_dataset(file_path) as ds:
            _ = ds.sizes
        return  # Valid netCDF, nothing to do
    except Exception:
        pass

    # Check if it's a zip
    if not zipfile.is_zipfile(file_path):
        return

    if not file_path.endswith(".nc"):
        return

    logger.debug("CDS sent zip disguised as netCDF: %s", file_path)

    zip_path = file_path.replace(".nc", ".zip")
    os.rename(file_path, zip_path)

    unzip_dir = file_path.replace(".nc", "")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(unzip_dir)

    # Merge all .nc files inside the zip
    nc_files = [
        os.path.join(unzip_dir, f)
        for f in os.listdir(unzip_dir)
        if f.endswith(".nc")
    ]
    if nc_files:
        datasets = [xr.open_dataset(f, engine="netcdf4") for f in nc_files]
        merged = xr.merge(datasets, compat="override")
        merged.to_netcdf(file_path)
        for ds in datasets:
            ds.close()

    # Clean up
    os.remove(zip_path)
    shutil.rmtree(unzip_dir)


def _remap_cds_dataset(ds: xr.Dataset, file_type: str) -> xr.Dataset:
    """Remap variable/dimension names from new CDS API to legacy standard.

    The CDS API (post mid-2024) uses 'valid_time' instead of 'time' and
    'pressure_level' instead of 'level'.
    """
    rename_map = CDS_RENAME_PLEV if file_type == "plev" else CDS_RENAME_SURF

    # Only rename dimensions that actually exist
    actual_renames = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    if actual_renames:
        ds = ds.rename(actual_renames)

    # Drop unwanted coordinates
    for var in CDS_DROP_VARS:
        if var in ds.coords or var in ds.data_vars:
            ds = ds.drop_vars(var)

    # Sort levels ascending for pressure level data
    if file_type == "plev" and "level" in ds.dims:
        ds = ds.sortby("level", ascending=True)

    return ds


class CDSBackend(ERA5Backend):
    """ERA5 backend using the Copernicus CDS API.

    Requires ``~/.cdsapirc`` with valid CDS credentials.
    Downloads ERA5 or ERA5T (data within ~5 days is ERA5T automatically).

    Args:
        bbox: Bounding box (west, south, east, north).
        pressure_levels: Pressure levels in hPa.
        time_resolution: '1H', '2H', '3H', or '6H'.
        product_type: CDS product type, default 'reanalysis'.
        output_format: CDS output format, default 'netcdf'.
        download_format: CDS download format, default 'unarchived'.
        surf_vars: Surface variable long names to download.
        plev_vars: Pressure level variable long names to download.
        tmp_dir: Temporary directory for downloaded files.
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int],
        time_resolution: str = "1H",
        product_type: str = "reanalysis",
        output_format: str = "netcdf",
        download_format: str = "unarchived",
        surf_vars: list[str] | None = None,
        plev_vars: list[str] | None = None,
        tmp_dir: str | None = None,
        show_progress: bool = True,
        **kwargs,
    ):
        super().__init__(bbox, pressure_levels, time_resolution, **kwargs)
        self.product_type = product_type
        self.output_format = output_format
        self.download_format = download_format
        self.surf_vars = surf_vars or DEFAULT_SURF_VARS
        self.plev_vars = plev_vars or DEFAULT_PLEV_VARS
        self._tmp_dir = Path(tmp_dir) if tmp_dir else Path("./cds_tmp/")
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self.show_progress = show_progress

        self._time_strings = TIME_RESOLUTION_STRINGS.get(time_resolution)
        if self._time_strings is None:
            raise ValueError(
                f"Invalid time_resolution '{time_resolution}'. "
                f"Must be one of: {list(TIME_RESOLUTION_STRINGS.keys())}"
            )

        # Pad bbox slightly to ensure edge cells are included
        self._cds_bbox = self.bbox.pad(0.0)

    def _request_surf(self, date: pd.Timestamp) -> Path:
        """Download one day of surface data via CDS API."""
        import cdsapi

        target = self._tmp_dir / f"dSURF_{date:%Y%m%d}.nc"
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "variable": self.surf_vars,
                "product_type": [self.product_type],
                "area": self._cds_bbox.to_cds_area(),
                "year": date.year,
                "month": f"{date.month:02d}",
                "day": f"{date.day:02d}",
                "time": self._time_strings,
                "grid": "0.25/0.25",
                "data_format": self.output_format,
                "download_format": self.download_format,
            },
            str(target),
        )
        _unzip_cds_file(str(target))
        return target

    def _request_plev(self, date: pd.Timestamp) -> Path:
        """Download one day of pressure-level data via CDS API."""
        import cdsapi

        target = self._tmp_dir / f"dPLEV_{date:%Y%m%d}.nc"
        c = cdsapi.Client()
        c.retrieve(
            "reanalysis-era5-pressure-levels",
            {
                "variable": self.plev_vars,
                "product_type": [self.product_type],
                "area": self._cds_bbox.to_cds_area(),
                "pressure_level": self.pressure_levels,
                "year": date.year,
                "month": f"{date.month:02d}",
                "day": f"{date.day:02d}",
                "time": self._time_strings,
                "grid": "0.25/0.25",
                "data_format": self.output_format,
                "download_format": self.download_format,
            },
            str(target),
        )
        _unzip_cds_file(str(target))
        return target

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data from CDS.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev) with standardised names and dims.
        """
        date = pd.Timestamp(date)
        logger.info("Fetching CDS ERA5 data for %s", date.strftime("%Y-%m-%d"))

        # Download surface and pressure-level data concurrently
        # (CDS allows 2 concurrent requests per user)
        with ThreadPoolExecutor(max_workers=2) as executor:
            fut_surf = executor.submit(self._request_surf, date)
            fut_plev = executor.submit(self._request_plev, date)

            completed = as_completed([fut_surf, fut_plev])
            if self.show_progress:
                completed = tqdm(
                    completed,
                    total=2,
                    desc=f"CDS {date.strftime('%Y-%m-%d')}",
                    unit="req",
                    leave=False,
                )
            for future in completed:
                future.result()  # Raise any exceptions

        surf_file = fut_surf.result()
        plev_file = fut_plev.result()

        ds_surf = xr.open_dataset(surf_file).load()
        ds_plev = xr.open_dataset(plev_file).load()

        # Remap CDS names to standard
        ds_surf = _remap_cds_dataset(ds_surf, "surf")
        ds_plev = _remap_cds_dataset(ds_plev, "plev")

        # Clean up temp files
        surf_file.unlink(missing_ok=True)
        plev_file.unlink(missing_ok=True)

        # Scale precipitation for subsampled data.
        # CDS returns only the requested hours, so each tp value is a 1-hour
        # accumulation.  Multiply by the step ratio so tp represents the total
        # for the timestep interval.
        res_hours = 24 // len(self._time_strings)
        if res_hours > 1 and "tp" in ds_surf:
            ds_surf["tp"] = ds_surf["tp"] * res_hours
            logger.info(
                "Scaled tp by %dx for %dH resolution (approximation)",
                res_hours, res_hours,
            )

        logger.info(
            "Fetched CDS: SURF vars=%s, PLEV vars=%s",
            list(ds_surf.data_vars),
            list(ds_plev.data_vars),
        )

        return ds_surf, ds_plev

    def close(self):
        """Clean up temp directory."""
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)
