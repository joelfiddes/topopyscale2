"""Open-Meteo S3 backend — direct access to .om files on AWS.

Reads ERA5 data directly from s3://openmeteo/data/copernicus_era5/ using
the omfiles library. Provides standard 0.25° grid matching Google/CDS.

Advantages over API:
- No rate limits (API: 500 req/min)
- Standard 0.25° grid (consistent with Google/CDS)
- Efficient partial reads for large regions

Limitations:
- Surface variables only (no pressure levels)
- ~4 years of recent data in chunks (late 2021+), year files 1950+
- Missing: strd (longwave radiation), q (specific humidity), z (nodata issues)

Use cases:
- Large regions (>1000 grid points) where API rate limits apply
- When grid consistency with Google/CDS is required
- Historical surface-only data (1950+)

Requires: omfiles, fsspec, s3fs
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that uses tqdm.write() to avoid breaking progress bars."""

    def emit(self, record):
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)

# ── Grid constants ────────────────────────────────────────────────────────

# Global 0.25° grid
N_LAT = 721  # 90 to -90
N_LON = 1440  # -180 to 179.75
STEP = 0.25
LATS = np.linspace(90.0, -90.0, N_LAT)  # Descending
LONS = np.linspace(-180.0, 179.75, N_LON)

# Chunk parameters (from S3 meta.json)
CHUNK_HOURS = 504  # 21 days per chunk

# ── Variable mappings ─────────────────────────────────────────────────────

# S3 variable name → (ERA5 short name, unit conversion)
# Conversion: callable(array) → array in ERA5 units
_C_TO_K = 273.15
_G = 9.80665

S3_SURFACE_VARS = {
    "temperature_2m": ("t2m", lambda x: x + _C_TO_K),       # °C → K
    "dew_point_2m": ("d2m", lambda x: x + _C_TO_K),         # °C → K
    "pressure_msl": ("msl", lambda x: x),                   # Pa (note: MSL not surface pressure)
    "shortwave_radiation": ("ssrd", lambda x: x * 3600.0),  # W/m² → J/m² per hour
    "precipitation": ("tp", lambda x: x / 1000.0),          # mm → m per hour
}

# Accumulated variables needing rolling sum
ACCUM_VARS = {"ssrd", "tp"}

# Variables NOT available in S3 (for documentation)
MISSING_VARS = {"strd", "q", "z", "sp"}  # longwave, specific humidity, geopotential, surface pressure


class OpenMeteoS3Backend(ERA5Backend):
    """Backend reading ERA5 from Open-Meteo S3 bucket.

    Reads .om files directly from s3://openmeteo using the omfiles library.
    Provides exact 0.25° grid alignment and no rate limits.

    Args:
        bbox: Bounding box (west, south, east, north).
        pressure_levels: Ignored (S3 has no pressure level data).
        time_resolution: '1H', '2H', '3H', or '6H'.
        cache_dir: Local cache directory for fsspec blockcache.
        max_workers: Number of parallel variable fetches.
        dataset: 'era5' or 'era5_land'.

    Note:
        Missing variables: strd (longwave), q (specific humidity), z (geopotential).
        Use Google or CDS backend if these are required.
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int] | None = None,
        time_resolution: str = "1H",
        cache_dir: str = "/tmp/openmeteo_s3_cache",
        max_workers: int = 4,
        dataset: str = "era5",  # "era5" or "era5_land"
        show_progress: bool = True,
        **kwargs,
    ):
        # Note: pressure_levels passed to parent but not used
        super().__init__(bbox, pressure_levels or [], time_resolution, **kwargs)

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers
        self.dataset = dataset
        self.show_progress = show_progress

        # Dataset path prefix
        self._dataset_prefix = f"copernicus_{dataset}"

        # Validate time resolution
        self._time_hours = {
            "1H": list(range(24)),
            "2H": list(range(0, 24, 2)),
            "3H": list(range(0, 24, 3)),
            "6H": list(range(0, 24, 6)),
        }.get(time_resolution)
        if self._time_hours is None:
            raise ValueError(
                f"Invalid time_resolution '{time_resolution}'. "
                f"Must be one of: 1H, 2H, 3H, 6H"
            )

        self._res_hours = int(time_resolution[0])

        # Calculate grid indices for bbox
        self._i_lat_start, self._i_lat_end, self._i_lon_start, self._i_lon_end = (
            self._bbox_to_indices(bbox)
        )

        # Extract coordinate arrays for our subset
        self._lats = LATS[self._i_lat_start : self._i_lat_end]
        self._lons = LONS[self._i_lon_start : self._i_lon_end]

        logger.info(
            "OpenMeteoS3Backend: %s bbox=%s → %d×%d grid points, %d workers",
            dataset,
            (bbox.west, bbox.south, bbox.east, bbox.north),
            len(self._lats),
            len(self._lons),
            max_workers,
        )

        if pressure_levels:
            logger.warning(
                "OpenMeteoS3Backend: S3 bucket has no pressure level data. "
                "Pressure level dataset will be empty."
            )

    def _bbox_to_indices(self, bbox: BBox) -> tuple[int, int, int, int]:
        """Convert bbox to array indices (inclusive end)."""
        west, south, east, north = bbox.west, bbox.south, bbox.east, bbox.north

        # Latitude is descending (90 at idx 0, -90 at idx 720)
        i_lat_start = int((90.0 - north) / STEP)
        i_lat_end = int((90.0 - south) / STEP) + 1

        # Longitude: -180 at idx 0, 179.75 at idx 1439
        i_lon_start = int((west + 180.0) / STEP)
        i_lon_end = int((east + 180.0) / STEP) + 1

        # Clamp to valid range
        i_lat_start = max(0, min(i_lat_start, N_LAT - 1))
        i_lat_end = max(1, min(i_lat_end, N_LAT))
        i_lon_start = max(0, min(i_lon_start, N_LON - 1))
        i_lon_end = max(1, min(i_lon_end, N_LON))

        return i_lat_start, i_lat_end, i_lon_start, i_lon_end

    def _date_to_chunk_id(self, date: pd.Timestamp) -> int:
        """Calculate chunk ID for a given date.

        Chunks are numbered sequentially from a reference point.
        Each chunk covers CHUNK_HOURS (504) hours = 21 days.
        """
        # Reference: chunk 975 starts at 2026-01-11 00:00 UTC
        ref_chunk = 975
        ref_start = pd.Timestamp("2026-01-11", tz="UTC")

        # Normalize date
        if date.tzinfo is None:
            date = date.tz_localize("UTC")

        # Calculate hours from reference start
        hours_diff = (date - ref_start).total_seconds() / 3600

        # Chunk offset (negative = earlier chunks)
        chunk_offset = math.floor(hours_diff / CHUNK_HOURS)

        return ref_chunk + chunk_offset

    def _chunk_time_range(self, chunk_id: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        """Get the time range covered by a chunk."""
        ref_chunk = 975
        ref_start = pd.Timestamp("2026-01-11", tz="UTC")

        chunk_offset = chunk_id - ref_chunk
        start = ref_start + pd.Timedelta(hours=chunk_offset * CHUNK_HOURS)
        end = start + pd.Timedelta(hours=CHUNK_HOURS)

        return start.tz_localize(None), end.tz_localize(None)

    def _open_s3_file(self, path: str):
        """Open an S3 file with caching."""
        try:
            import fsspec
            from omfiles import OmFileReader
        except ImportError as e:
            raise ImportError(
                "OpenMeteoS3Backend requires 'omfiles' and 'fsspec' packages. "
                "Install with: pip install omfiles fsspec s3fs"
            ) from e

        backend = fsspec.open(
            f"blockcache::s3://{path}",
            mode="rb",
            s3={"anon": True},
            blockcache={"cache_storage": str(self.cache_dir)},
        )
        return OmFileReader(backend.open())

    def _get_file_path(self, var_name: str, date: pd.Timestamp) -> tuple[str, slice]:
        """Determine file path and time slice for a given date.

        Returns:
            (s3_path, time_slice) tuple.
        """
        # Check if date falls in recent chunk range (late 2021+) or needs year file
        # Chunk 904 starts ~2021-12-12, so use year files for anything before 2022
        if date.year < 2022:
            # Year file: year_YYYY.om contains full year hourly data
            year = date.year
            path = f"openmeteo/data/{self._dataset_prefix}/{var_name}/year_{year}.om"

            # Time slice within year file (hour of year)
            start_of_year = pd.Timestamp(f"{year}-01-01")
            hour_offset = int((date - start_of_year).total_seconds() / 3600)
            time_slice = slice(hour_offset, hour_offset + 24 + self._res_hours)
        else:
            # Chunk file: chunk_XXX.om for recent data
            chunk_id = self._date_to_chunk_id(date)
            path = f"openmeteo/data/{self._dataset_prefix}/{var_name}/chunk_{chunk_id}.om"

            chunk_start, _ = self._chunk_time_range(chunk_id)
            hour_offset = int((date - chunk_start).total_seconds() / 3600)
            wider_offset = max(0, hour_offset - (self._res_hours - 1))
            time_slice = slice(wider_offset, hour_offset + 24)

        return path, time_slice

    def _read_variable(
        self,
        var_name: str,
        date: pd.Timestamp,
    ) -> np.ndarray:
        """Read a variable from S3 for the configured bbox.

        Args:
            var_name: S3 variable name (e.g., 'temperature_2m').
            date: Date to fetch.

        Returns:
            Array of shape (n_times, n_lats, n_lons).
        """
        path, time_slice = self._get_file_path(var_name, date)

        reader = self._open_s3_file(path)
        try:
            # Shape is (lat, lon, time) in .om files
            data = reader[
                self._i_lat_start : self._i_lat_end,
                self._i_lon_start : self._i_lon_end,
                time_slice,
            ]
        finally:
            reader.close()

        # Transpose to (time, lat, lon) for xarray convention
        return np.transpose(data, (2, 0, 1))

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data from S3.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev). ds_plev is empty (no plev data in S3).
        """
        date = pd.Timestamp(date).normalize()
        day_start = date
        day_end = date + pd.Timedelta(days=1)

        logger.debug(
            "fetch_day %s: parallel=%d workers",
            date.strftime("%Y-%m-%d"),
            self.max_workers,
        )

        # Parallel fetch of all variables
        def fetch_var(s3_name: str) -> tuple[str, np.ndarray | None]:
            era5_name, convert = S3_SURFACE_VARS[s3_name]
            try:
                raw = self._read_variable(s3_name, date)
                return era5_name, convert(raw).astype("float32")
            except Exception as e:
                logger.warning("Variable %s not found in S3: %s", s3_name, e)
                return era5_name, None

        surf_arrays: dict[str, np.ndarray] = {}
        var_items = list(S3_SURFACE_VARS.keys())

        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(fetch_var, name): name for name in var_items}
                iterator = as_completed(futures)
                if self.show_progress:
                    iterator = tqdm(
                        iterator,
                        total=len(futures),
                        desc=f"Fetching {date.strftime('%Y-%m-%d')}",
                        unit="var",
                        leave=False,
                    )
                for future in iterator:
                    era5_name, data = future.result()
                    if data is not None:
                        surf_arrays[era5_name] = data
        else:
            iterator = var_items
            if self.show_progress:
                iterator = tqdm(
                    iterator,
                    desc=f"Fetching {date.strftime('%Y-%m-%d')}",
                    unit="var",
                    leave=False,
                )
            for s3_name in iterator:
                era5_name, data = fetch_var(s3_name)
                if data is not None:
                    surf_arrays[era5_name] = data

        if not surf_arrays:
            raise RuntimeError(f"No variables found for {date}")

        # Build time coordinate from first variable's shape
        n_times = list(surf_arrays.values())[0].shape[0]
        if date.year < 2022:
            # Year file: hours from start of day
            times_full = pd.date_range(date, periods=n_times, freq="1h")
        else:
            # Chunk file
            chunk_id = self._date_to_chunk_id(date)
            chunk_start, _ = self._chunk_time_range(chunk_id)
            hour_offset = int((date - chunk_start).total_seconds() / 3600)
            wider_offset = max(0, hour_offset - (self._res_hours - 1))
            times_full = pd.date_range(
                chunk_start + pd.Timedelta(hours=wider_offset),
                periods=n_times,
                freq="1h",
            )

        # Handle accumulated variables: rolling sum over time resolution
        if self._res_hours > 1:
            for var in ACCUM_VARS:
                if var in surf_arrays:
                    arr = surf_arrays[var]
                    rolled = np.zeros_like(arr)
                    for t in range(len(arr)):
                        t_start = max(0, t - self._res_hours + 1)
                        rolled[t] = arr[t_start : t + 1].sum(axis=0)
                    surf_arrays[var] = rolled

        # Build dataset
        ds_surf = xr.Dataset(
            {
                name: (["time", "latitude", "longitude"], data)
                for name, data in surf_arrays.items()
            },
            coords={
                "time": times_full,
                "latitude": self._lats,
                "longitude": self._lons,
            },
        )

        # Filter to requested time resolution and trim to day
        ds_surf = ds_surf.sel(time=ds_surf.time.dt.hour.isin(self._time_hours))
        ds_surf = ds_surf.sel(time=slice(day_start, day_end - pd.Timedelta(seconds=1)))

        # Empty plev dataset (S3 has no pressure level data)
        ds_plev = xr.Dataset()

        logger.info(
            "OpenMeteoS3: fetched %s — SURF vars=%s",
            date.strftime("%Y-%m-%d"),
            list(ds_surf.data_vars),
        )

        return ds_surf, ds_plev

    def close(self):
        """Clean up resources."""
        pass
