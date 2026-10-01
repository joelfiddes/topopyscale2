"""Google ARCO-ERA5 backend.

Downloads ERA5 data from Google Cloud Storage's public ARCO-ERA5 dataset.
Anonymous access, no credentials required. Requires ``gcsfs`` and ``h5netcdf``.

URI pattern:
    gs://gcp-public-data-arco-era5/raw/date-variable-{single_level|pressure_level}/
        {YYYY}/{MM}/{DD}/{variable}/{level|surface}.nc

Ported from: era5google/era5_downloader/core.py (Luke Gregor)
"""

from __future__ import annotations

import logging
import pathlib
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

import fsspec
import pandas as pd
import xarray as xr
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.variables import (
    DEFAULT_PLEV_VARS,
    DEFAULT_SURF_VARS,
    TIME_RESOLUTION_HOURS,
    VARS_BOTH_SURFACE_AND_LEVEL,
)

logger = logging.getLogger(__name__)

# libhdf5 is NOT thread-safe. The loader fetches days in parallel (loader.py
# ThreadPoolExecutor over fetch_day), so HDF5 opens/reads can otherwise happen
# concurrently across threads -> SIGSEGV on the NetCDF-4 files ARCO now serves.
# Serialise every HDF5 open+read through one process-wide lock; the download
# phase (network I/O) stays parallel.
_HDF5_LOCK = threading.Lock()


class AdaptiveThrottle:
    """Global concurrency limiter with gradual ramp-up for GCS downloads.

    Google Cloud Storage recommends ramping up request rates gradually:
    "no faster than doubling the rate over a period of 20 minutes."

    This throttle limits the total number of concurrent file downloads
    across all day-workers and file-workers, starting low and doubling
    at a configurable interval.

    Args:
        initial: Starting number of concurrent downloads allowed.
        maximum: Maximum concurrent downloads after full ramp-up.
        ramp_interval: Seconds between each doubling (default 1200 = 20 min).
    """

    def __init__(
        self,
        initial: int = 16,
        maximum: int = 64,
        ramp_interval: float = 1200.0,
    ):
        self._semaphore = threading.Semaphore(initial)
        self._current = initial
        self._maximum = maximum
        self._ramp_interval = ramp_interval
        self._next_ramp = time.monotonic() + ramp_interval
        self._lock = threading.Lock()
        self._request_count = 0
        self._start_time = time.monotonic()

    @property
    def current_limit(self) -> int:
        """Current concurrency limit."""
        return self._current

    def acquire(self):
        """Acquire a download slot. Blocks if at capacity."""
        self._check_ramp()
        self._semaphore.acquire()

    def release(self):
        """Release a download slot."""
        self._semaphore.release()
        with self._lock:
            self._request_count += 1

    def _check_ramp(self):
        """Double concurrency limit if ramp interval has elapsed."""
        now = time.monotonic()
        with self._lock:
            if now >= self._next_ramp and self._current < self._maximum:
                increase = min(self._current, self._maximum - self._current)
                for _ in range(increase):
                    self._semaphore.release()  # Add permits
                old = self._current
                self._current += increase
                self._next_ramp = now + self._ramp_interval
                elapsed_min = (now - self._start_time) / 60
                logger.info(
                    "Throttle ramp-up: %d → %d concurrent downloads "
                    "(%.0f min elapsed, %d requests completed)",
                    old, self._current, elapsed_min, self._request_count,
                )

URI_LEVELS = (
    "filecache::gs://gcp-public-data-arco-era5/raw/"
    "date-variable-pressure_level/{t:%Y}/{t:%m}/{t:%d}/{variable}/{level}.nc"
)
URI_SURFACE = (
    "filecache::gs://gcp-public-data-arco-era5/raw/"
    "date-variable-single_level/{t:%Y}/{t:%m}/{t:%d}/{variable}/surface.nc"
)


class GoogleCloudBackend(ERA5Backend):
    """ERA5 backend using Google ARCO-ERA5 public netCDF files.

    Args:
        bbox: Bounding box (west, south, east, north) in -180:180 coords.
        pressure_levels: List of pressure levels in hPa.
        time_resolution: '1H', '2H', '3H', or '6H'.
        cache_dir: Directory for fsspec file cache. Cleaned after each day.
        surf_vars: Surface variable long names to download.
        plev_vars: Pressure level variable long names to download.
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int],
        time_resolution: str = "1H",
        cache_dir: str = "./era5_cache/",
        surf_vars: list[str] | None = None,
        plev_vars: list[str] | None = None,
        show_progress: bool = True,
        max_workers: int = 8,
        throttle: bool = True,
        throttle_initial: int = 16,
        throttle_ramp_minutes: float = 20.0,
        **kwargs,
    ):
        super().__init__(bbox, pressure_levels, time_resolution, **kwargs)

        # Convert bbox to 0:360 for Google's data
        self._bbox_360 = self.bbox.to_0_360()

        self.surf_vars = surf_vars or DEFAULT_SURF_VARS
        self.plev_vars = plev_vars or DEFAULT_PLEV_VARS
        self.show_progress = show_progress
        self._max_workers = max_workers

        self._time_steps = TIME_RESOLUTION_HOURS.get(time_resolution)
        if self._time_steps is None:
            raise ValueError(
                f"Invalid time_resolution '{time_resolution}'. "
                f"Must be one of: {list(TIME_RESOLUTION_HOURS.keys())}"
            )

        # Set up cache directory
        self._cache_dir = pathlib.Path(cache_dir).expanduser().resolve()
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Adaptive throttle for sustained bulk downloads.
        # GCS throttles above ~32 concurrent connections from a single IP.
        # Cap maximum at 32 regardless of worker count.
        if throttle:
            throttle_max = min(32, max_workers * 8)
            self._throttle: AdaptiveThrottle | None = AdaptiveThrottle(
                initial=throttle_initial,
                maximum=throttle_max,
                ramp_interval=throttle_ramp_minutes * 60,
            )
            logger.info(
                "Adaptive throttle enabled: %d initial → %d max concurrent "
                "(ramp every %.0f min)",
                throttle_initial, throttle_max, throttle_ramp_minutes,
            )
        else:
            self._throttle = None

        # Discover valid variables from the store
        self._valid_surface_vars: list[str] | None = None
        self._valid_level_vars: list[str] | None = None
        self._valid_levels: list[int] | None = None

    def _ensure_valid_vars(self):
        """Lazily discover available variables from the Google store."""
        if self._valid_surface_vars is not None:
            return

        fs = fsspec.filesystem("gs", token="anon")
        t = pd.Timestamp("2000-01-01")

        base_surface = (
            "gcp-public-data-arco-era5/raw/date-variable-single_level/"
            f"{t:%Y}/{t:%m}/{t:%d}"
        )
        base_levels = (
            "gcp-public-data-arco-era5/raw/date-variable-pressure_level/"
            f"{t:%Y}/{t:%m}/{t:%d}"
        )

        def get_name(f):
            return f.split("/")[-1].replace(".nc", "")

        self._valid_surface_vars = [get_name(f) for f in fs.ls(base_surface)]
        self._valid_level_vars = [get_name(f) for f in fs.ls(base_levels)]

        level_path = f"{base_levels}/temperature"
        self._valid_levels = sorted([int(get_name(f)) for f in fs.ls(level_path)])

        logger.debug("Valid surface vars: %s", self._valid_surface_vars)
        logger.debug("Valid level vars: %s", self._valid_level_vars)
        logger.debug("Valid levels: %s", self._valid_levels)

    def _surface_or_level(self, variable: str) -> str:
        """Classify a variable as 'surface', 'level', or 'both'."""
        self._ensure_valid_vars()
        in_surf = variable in self._valid_surface_vars
        in_level = variable in self._valid_level_vars
        if in_surf and in_level:
            return "both"
        if in_surf:
            return "surface"
        if in_level:
            return "level"
        raise ValueError(
            f"Variable '{variable}' not found in Google ARCO-ERA5. "
            f"Surface: {self._valid_surface_vars}, Level: {self._valid_level_vars}"
        )

    def _make_uri_list(self, t: pd.Timestamp) -> Iterator[str]:
        """Generate URIs for all variables and levels for a given day."""
        all_vars = set(self.surf_vars) | set(self.plev_vars)
        for variable in all_vars:
            var_type = self._surface_or_level(variable)
            if var_type in ("surface", "both"):
                yield URI_SURFACE.format(t=t, variable=variable)
            if var_type in ("level", "both"):
                for level in self.pressure_levels:
                    yield URI_LEVELS.format(t=t, variable=variable, level=level)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=60), reraise=True)
    def _download_single_file(self, uri: str, cache_dir: pathlib.Path) -> str:
        """Download a single file via fsspec. Thread-safe.

        Respects the adaptive throttle if enabled, limiting total concurrent
        downloads across all day-workers to avoid GCS rate limiting.

        Args:
            uri: Remote URI to download.
            cache_dir: Per-file cache directory (avoids cache metadata conflicts).

        Returns:
            Local file path.
        """
        if self._throttle:
            self._throttle.acquire()
        try:
            local_path = fsspec.open_local(
                url=uri,
                filecache=dict(cache_storage=str(cache_dir)),
                gs=dict(token="anon"),
            )
            return local_path
        finally:
            if self._throttle:
                self._throttle.release()

    def _download_files(self, uri_list: list[str]) -> tuple[list[str], pathlib.Path, dict]:
        """Download files in parallel using ThreadPoolExecutor.

        Each file is downloaded to its own cache sub-directory to avoid
        fsspec cache metadata conflicts between threads.

        Returns:
            Tuple of (file_list, tmp_dir, uri_map) where uri_map maps
            local filename to original URI for preprocess lookups.
        """
        tmp_dir = pathlib.Path(tempfile.mkdtemp(dir=str(self._cache_dir)))
        n_files = len(uri_list)
        logger.debug("Downloading %d files to %s", n_files, tmp_dir)

        results = [None] * n_files

        def _dl(idx_uri):
            idx, uri = idx_uri
            sub_dir = tmp_dir / f"dl_{idx:04d}"
            sub_dir.mkdir(exist_ok=True)
            local_path = self._download_single_file(uri, sub_dir)
            return idx, local_path, uri

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {
                executor.submit(_dl, (i, uri)): i
                for i, uri in enumerate(uri_list)
            }
            completed = as_completed(futures)
            if self.show_progress:
                completed = tqdm(
                    completed,
                    total=n_files,
                    desc="Downloading",
                    unit="file",
                    leave=False,
                )
            for future in completed:
                idx, local_path, orig_uri = future.result()
                results[idx] = (local_path, orig_uri)

        flist = [r[0] for r in results]

        # Build URI mapping: local filename -> {"original": uri}
        # (replaces fsspec cache JSON parsing)
        fsspec_cache = {}
        for local_path, orig_uri in results:
            fname = pathlib.Path(local_path).name
            fsspec_cache[fname] = {"fn": fname, "original": orig_uri}

        return flist, tmp_dir, fsspec_cache

    def _get_original_uri(self, ds: xr.Dataset, fsspec_cache: dict) -> str:
        """Look up the original URI from fsspec cache.

        Args:
            ds: Dataset with encoding["source"] pointing to cached file.
            fsspec_cache: Cache dict mapping file hashes to metadata (local to fetch_day call).
        """
        fname_hash = pathlib.Path(ds.encoding["source"]).name
        entry = fsspec_cache.get(fname_hash, {})
        return entry.get("original", "")

    def _preprocess(self, ds: xr.Dataset, fsspec_cache: dict) -> xr.Dataset:
        """Preprocess a single file: subset, rename, add level dim.

        Args:
            ds: Raw dataset from a single netCDF file.
            fsspec_cache: Cache dict for URI lookup (local to fetch_day call).
        """
        uri = self._get_original_uri(ds, fsspec_cache)
        is_level = "pressure_level" in uri

        # Variable name from URI path
        parts = uri.split("/")
        if len(parts) >= 2:
            request_variable = parts[-2]
        else:
            request_variable = ""

        var_type = "level" if is_level else "surface"
        if request_variable in VARS_BOTH_SURFACE_AND_LEVEL:
            var_type = "level" if is_level else "both_surface"

        # Temporal + latitude subset
        bb = self._bbox_360
        ds = ds.sel(
            latitude=slice(bb.north, bb.south),
            time=ds.time.dt.hour.isin(self._time_steps),
        )
        # Longitude: select using -180:180 semantics so prime-meridian-crossing
        # boxes (west<0, e.g. a European box) work. ARCO longitude is 0:360 and
        # the 0:360 BBox conversion collapses a wrapped box to a point, so map
        # ARCO longitudes to -180:180, mask by the original bbox, and store the
        # axis monotonic in -180:180 — the convention the local cache backend
        # selects with (local.py uses the raw bbox, no 0:360 conversion).
        o = self.bbox
        ds = ds.assign_coords(
            longitude=(((ds.longitude + 180.0) % 360.0) - 180.0)
        )
        if o.west <= o.east:
            mask = (ds.longitude >= o.west) & (ds.longitude <= o.east)
        else:  # box crosses the antimeridian (180/-180)
            mask = (ds.longitude >= o.west) | (ds.longitude <= o.east)
        ds = ds.sel(longitude=ds.longitude[mask]).sortby("longitude")

        # Get the data variable key
        key = list(ds.data_vars)[0]

        # Rename surface geopotential to avoid clash with pressure-level geopotential
        if var_type == "both_surface":
            ds = ds.rename({key: key + "_surf"})
            key = key + "_surf"

        # Add level dimension for pressure-level files
        if is_level:
            level_str = uri.split("/")[-1].replace(".nc", "")
            if level_str.isdigit():
                ds = ds.expand_dims(level=[int(level_str)])

        ds = ds.astype("float32")
        return ds

    def _clear_cache(self, tmp_dir: pathlib.Path | None = None):
        """Remove temporary cache files.

        Args:
            tmp_dir: Temporary directory to clean up. If None, does nothing.
                     Each fetch_day call passes its own tmp_dir to avoid race conditions.
        """
        if tmp_dir is None or not tmp_dir.exists():
            return
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data from Google ARCO-ERA5.

        This method is thread-safe: each call uses its own temporary directory
        and cache, avoiding race conditions when downloading multiple days in parallel.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev) with standardised names and dims.
        """
        date = pd.Timestamp(date)
        logger.info("Fetching Google ARCO-ERA5 data for %s", date.strftime("%Y-%m-%d"))

        uri_list = list(self._make_uri_list(date))
        flist, tmp_dir, fsspec_cache = self._download_files(uri_list)

        logger.debug("Opening and preprocessing %d files", len(flist))
        # Open each file individually, preprocess, load into memory, then merge
        # (open_mfdataset with combine_by_coords fails when files have
        # different variables but same coordinates - they need merge, not concat)

        def _open_and_preprocess(filepath):
            # ARCO switched recent ERA5 raw files from NetCDF-3 to NetCDF-4/HDF5
            # around 2026-07 (older dates stay NetCDF-3). "netcdf4" reads both;
            # the previous "scipy" engine only reads NetCDF-3 and broke on the
            # new HDF5 files ("not a valid NetCDF 3 file").
            # Hold the process-wide HDF5 lock across the open AND the .load()
            # below (both touch libhdf5, which is not thread-safe).
            with _HDF5_LOCK:
                ds = xr.open_dataset(filepath, engine="netcdf4")
                # The new format also renames the time coord (valid_time) and
                # adds scalar number/expver coords + a size-1 pressure_level
                # dim. Normalise back to the legacy schema _preprocess expects:
                # a "time" dim, no level dim (the level is re-added from the URI
                # in _preprocess). No-op on old NetCDF-3 files.
                if "valid_time" in ds.dims or "valid_time" in ds.coords:
                    ds = ds.rename({"valid_time": "time"})
                ds = ds.drop_vars(["number", "expver"], errors="ignore")
                if "pressure_level" in ds.dims:
                    ds = ds.squeeze("pressure_level", drop=True)
                elif "pressure_level" in ds.coords:
                    ds = ds.drop_vars("pressure_level")
                return self._preprocess(ds, fsspec_cache).load()

        n_files = len(flist)
        datasets = [None] * n_files
        # Open/preprocess SERIALLY: netCDF4/libhdf5 is NOT thread-safe for
        # concurrent reads and SIGSEGVs (exit 139) when opening the HDF5 files
        # ARCO now serves. The prior ThreadPoolExecutor was safe only because
        # the old scipy engine read NetCDF-3. Download parallelism is retained
        # in _download_files above; only the open/preprocess is serialised.
        file_iter = enumerate(flist)
        if self.show_progress:
            file_iter = tqdm(
                file_iter,
                total=n_files,
                desc=f"Google {date.strftime('%Y-%m-%d')}",
                unit="file",
                leave=False,
            )
        for i, filepath in file_iter:
            datasets[i] = _open_and_preprocess(filepath)

        # Separate surface and pressure-level datasets before merging
        # Pressure-level data: group by variable, concat along level, then merge
        surf_datasets = []
        plev_by_var = {}  # dict of var_name -> list of single-level datasets

        for ds_single in datasets:
            has_level = any("level" in ds_single[v].dims for v in ds_single.data_vars)
            if has_level:
                # Group by variable name for later concatenation
                for var_name in ds_single.data_vars:
                    if "level" in ds_single[var_name].dims:
                        if var_name not in plev_by_var:
                            plev_by_var[var_name] = []
                        # Extract just this variable as a dataset
                        plev_by_var[var_name].append(ds_single[[var_name]])
            else:
                surf_datasets.append(ds_single)

        # Merge surface datasets (different variables, same coords)
        if surf_datasets:
            ds_surf = xr.merge(surf_datasets, compat="override", join="outer")
        else:
            ds_surf = xr.Dataset()

        # For pressure-level data: concat each variable along level, then merge
        if plev_by_var:
            plev_merged = []
            for var_name, var_datasets in plev_by_var.items():
                # Concatenate this variable along level dimension
                var_concat = xr.concat(var_datasets, dim="level", combine_attrs="override")
                plev_merged.append(var_concat)
            # Merge all concatenated variables
            ds_plev = xr.merge(plev_merged, compat="override", join="outer")
        else:
            ds_plev = xr.Dataset()

        # Clean up this call's temporary directory (thread-safe: each call has its own)
        self._clear_cache(tmp_dir)

        # Scale precipitation for subsampled data.
        # When time_resolution > 1H, the preprocess step picks every Nth hour,
        # but each tp value is still a 1-hour accumulation.  Multiply by the
        # step ratio so tp represents the total for the timestep interval.
        res_hours = 24 // len(self._time_steps)
        if res_hours > 1 and "tp" in ds_surf:
            ds_surf["tp"] = ds_surf["tp"] * res_hours
            logger.info(
                "Scaled tp by %dx for %dH resolution (approximation)",
                res_hours, res_hours,
            )

        # Sort pressure levels ascending
        if "level" in ds_plev.dims:
            ds_plev = ds_plev.sortby("level", ascending=True)

        # Note: Surface geopotential is kept as z_surf (renamed in preprocess)
        # to avoid conflict with pressure-level z. Other variables already
        # have their short names from the netCDF files.

        logger.info(
            "Fetched: SURF vars=%s, PLEV vars=%s, levels=%s",
            list(ds_surf.data_vars),
            list(ds_plev.data_vars),
            list(ds_plev.level.values) if "level" in ds_plev.dims else [],
        )

        return ds_surf, ds_plev

    def probe_date(self, date: pd.Timestamp) -> bool:
        """Check if ERA5 data exists for a given date (single file probe)."""
        fs = fsspec.filesystem("gs", token="anon")
        test_uri = (
            "gcp-public-data-arco-era5/raw/date-variable-single_level/"
            f"{date:%Y}/{date:%m}/{date:%d}/surface_pressure/surface.nc"
        )
        try:
            return fs.exists(test_uri)
        except Exception:
            return False

    def close(self):
        """Clean up any remaining cache.

        Note: With the thread-safe design, each fetch_day() cleans its own cache,
        so this is typically a no-op. Kept for interface compatibility.
        """
        pass
