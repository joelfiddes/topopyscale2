"""File validation and resume logic for downloads."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def daily_netcdf_exists(output_dir: str | Path, prefix: str, date: pd.Timestamp) -> bool:
    """Check if a daily netCDF file already exists and is valid.

    Args:
        output_dir: Base output directory.
        prefix: 'dSURF' or 'dPLEV'.
        date: The date to check.

    Returns:
        True if the file exists and can be opened.
    """
    path = Path(output_dir) / "daily" / f"{prefix}_{date:%Y%m%d}.nc"
    if not path.exists():
        return False
    try:
        with xr.open_dataset(path) as ds:
            return ds.sizes["time"] > 0
    except Exception:
        logger.warning("Corrupt file detected, will re-download: %s", path)
        return False


def zarr_day_exists(output_dir: str | Path, date: pd.Timestamp) -> bool:
    """Check if a day already exists as a per-day zarr store.

    Checks for daily/day_YYYYMMDD.zarr stores (new format) or
    dates in ERA5.zarr (legacy format).

    Args:
        output_dir: Base output directory.
        date: The date to check.

    Returns:
        True if the date's data is present.
    """
    output_dir = Path(output_dir)

    # Check per-day store (new format)
    day_path = output_dir / "daily" / f"day_{date:%Y%m%d}.zarr"
    if day_path.exists():
        try:
            ds = xr.open_zarr(day_path)
            has_data = ds.sizes.get("time", 0) > 0
            ds.close()
            return has_data
        except Exception:
            logger.warning("Corrupt zarr store detected: %s", day_path)
            return False

    # Check merged store (legacy format)
    zarr_path = output_dir / "ERA5.zarr"
    if zarr_path.exists():
        try:
            ds = xr.open_zarr(zarr_path)
            times = pd.DatetimeIndex(ds.time.values)
            ds.close()
            return date.normalize() in times.normalize()
        except Exception:
            return False

    return False


def get_existing_dates(
    output_dir: str | Path,
    output_format: str,
    prefix: str = "dSURF",
) -> set[pd.Timestamp]:
    """Scan output directory for already-downloaded dates.

    Args:
        output_dir: Base output directory.
        output_format: 'netcdf' or 'zarr'.
        prefix: File prefix to check ('dSURF' or 'dPLEV'). Ignored for zarr.

    Returns:
        Set of dates already present.
    """
    output_dir = Path(output_dir)
    existing = set()

    if output_format == "netcdf":
        daily_dir = output_dir / "daily"
        if daily_dir.exists():
            for f in daily_dir.glob(f"{prefix}_*.nc"):
                try:
                    date_str = f.stem.split("_")[1]
                    existing.add(pd.Timestamp(date_str))
                except (IndexError, ValueError):
                    continue

    elif output_format == "zarr":
        # Check per-day stores (new format)
        daily_dir = output_dir / "daily"
        if daily_dir.exists():
            for f in daily_dir.glob("day_*.zarr"):
                try:
                    date_str = f.stem.split("_")[1]
                    existing.add(pd.Timestamp(date_str))
                except (IndexError, ValueError):
                    continue

        # Also check merged store (legacy format)
        zarr_path = output_dir / "ERA5.zarr"
        if zarr_path.exists():
            try:
                ds = xr.open_zarr(zarr_path)
                times = pd.DatetimeIndex(ds.time.values)
                ds.close()
                existing.update(times.normalize().unique())
            except Exception:
                pass

    return existing
