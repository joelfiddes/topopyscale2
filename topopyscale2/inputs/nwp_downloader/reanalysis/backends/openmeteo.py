"""Open-Meteo backend for ERA5 reanalysis and IFS historical forecasts.

Two modes controlled by the ``model`` parameter:

* **model="era5"** (default): ERA5 reanalysis via the Open-Meteo archive API.
  Surface variables only, 1940–present, ~0.25° resolution.

* **model="ifs"**: IFS historical forecasts via the Open-Meteo historical
  forecast API.  Surface + 19 pressure levels, 2022–present.

Fast, free, no credentials required.  Requires ``requests``.

Pre-fetch strategy
------------------
Open-Meteo returns the full date range in one request per grid point.  On the
first ``fetch_day()`` call the backend pre-fetches the *entire* date range for
all grid points, caches the result in memory, and subsequent ``fetch_day()``
calls slice from the cache.
"""

from __future__ import annotations

import logging
import math
import threading
import time as _time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.variables import TIME_RESOLUTION_HOURS

logger = logging.getLogger(__name__)

# ── API endpoints ────────────────────────────────────────────────────────────

ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
IFS_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"

# ── Variable mappings ────────────────────────────────────────────────────────

# Open-Meteo parameter → (ERA5 short name, conversion function)
# Conversion: callable(numpy array) → numpy array in ERA5 units.
#
# For accumulated variables (ssrd, tp) the conversion produces per-HOUR
# values.  The backend then sums over the time-resolution window in
# fetch_day() so the output matches ERA5 convention (accumulated per
# output timestep).

_C_TO_K = 273.15
_G = 9.80665

SURFACE_MAP = {
    "temperature_2m":       ("t2m",  lambda x: x + _C_TO_K),            # °C → K
    "dew_point_2m":         ("d2m",  lambda x: x + _C_TO_K),            # °C → K
    "surface_pressure":     ("sp",   lambda x: x * 100.0),              # hPa → Pa
    "shortwave_radiation":  ("ssrd", lambda x: x * 3600.0),             # W/m² → J/m² per hour
    "precipitation":        ("tp",   lambda x: x / 1000.0),             # mm → m per hour
}

# Accumulated surface variables that need summing over the time-resolution
# window (see fetch_day).
_ACCUM_SURF_VARS = {"ssrd", "tp"}

# Pressure-level parameters (IFS mode only).
# These are *templates*; the actual API parameter names include the level,
# e.g. "temperature_850hPa".  We handle the level suffix in code.
PLEV_MAP = {
    "temperature":          ("t",  lambda x: x + _C_TO_K),              # °C → K
    "geopotential_height":  ("z",  lambda x: x * _G),                   # gpm → m²/s²
    "relative_humidity":    ("r",  lambda x: x),                        # % → %
    # wind_speed + wind_direction → u, v  (handled specially)
}

# IFS pressure levels available on Open-Meteo (hPa)
IFS_AVAILABLE_LEVELS = [
    1000, 975, 950, 925, 900, 875, 850, 800, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
    150, 100, 70, 50, 30,
]

# ── Rate limiter ─────────────────────────────────────────────────────────────

_MAX_REQUESTS_PER_MINUTE = 500


class _RateLimiter:
    """Sliding-window rate limiter."""

    def __init__(self, max_per_minute: int = _MAX_REQUESTS_PER_MINUTE):
        self._window: deque[float] = deque()
        self._max = max_per_minute
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = _time.monotonic()
            # Purge entries older than 60 s
            while self._window and self._window[0] < now - 60:
                self._window.popleft()
            if len(self._window) >= self._max:
                sleep_for = 60.0 - (now - self._window[0])
                if sleep_for > 0:
                    logger.debug("Rate-limiting: sleeping %.1f s", sleep_for)
                    _time.sleep(sleep_for)
            self._window.append(_time.monotonic())


# ── Backend ──────────────────────────────────────────────────────────────────


class OpenMeteoBackend(ERA5Backend):
    """Open-Meteo backend for ERA5 archive and IFS historical forecasts.

    Args:
        bbox: Bounding box (west, south, east, north) in -180:180 coords.
        pressure_levels: Pressure levels in hPa.
        time_resolution: '1H', '2H', '3H', or '6H'.
        start_date: Start date (YYYY-MM-DD).  **Required.**
        end_date: End date (YYYY-MM-DD).  **Required.**
        model: ``"era5"`` (surface only, 1940+) or ``"ifs"`` (surface + plev, 2022+).
        batch_size: Number of grid points per API request.
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int],
        time_resolution: str = "1H",
        start_date: str | None = None,
        end_date: str | None = None,
        model: str = "era5",
        batch_size: int = 20,
        show_progress: bool = True,
        **kwargs,
    ):
        super().__init__(bbox, pressure_levels, time_resolution, **kwargs)

        if start_date is None or end_date is None:
            raise ValueError(
                "OpenMeteoBackend requires start_date and end_date "
                "(pass via backend_kwargs)"
            )

        self._start_date = str(start_date)
        self._end_date = str(end_date)

        if model not in ("era5", "ifs"):
            raise ValueError(f"model must be 'era5' or 'ifs', got '{model}'")
        self._model = model
        self.show_progress = show_progress

        self._time_steps = TIME_RESOLUTION_HOURS.get(time_resolution)
        if self._time_steps is None:
            raise ValueError(
                f"Invalid time_resolution '{time_resolution}'. "
                f"Must be one of: {list(TIME_RESOLUTION_HOURS.keys())}"
            )

        self._batch_size = batch_size
        self._rate_limiter = _RateLimiter()

        # Cache populated on first fetch_day()
        self._cache_surf: xr.Dataset | None = None
        self._cache_plev: xr.Dataset | None = None
        self._lock = threading.Lock()

        # Validate requested pressure levels for IFS mode
        if self._model == "ifs":
            bad = [l for l in self.pressure_levels if l not in IFS_AVAILABLE_LEVELS]
            if bad:
                raise ValueError(
                    f"Pressure levels {bad} not available on Open-Meteo IFS. "
                    f"Available: {IFS_AVAILABLE_LEVELS}"
                )

        # Warn about missing variables
        logger.warning(
            "Open-Meteo does not provide longwave radiation (strd). "
            "This variable will be missing from the output."
        )
        if self._model == "era5":
            logger.info(
                "Open-Meteo ERA5 mode: surface variables only, no pressure levels."
            )

    # ── Grid construction ────────────────────────────────────────────────

    def _build_grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Build a regular 0.25° grid from the bounding box.

        Grid points are snapped to 0.25° and only those falling *within*
        the bounding box are included (matching Google/CDS backend behaviour).

        Returns:
            (lats, lons) 1-D arrays snapped to 0.25° resolution.
        """
        step = 0.25
        lat_min = math.ceil(self.bbox.south / step) * step
        lat_max = math.floor(self.bbox.north / step) * step
        lon_min = math.ceil(self.bbox.west / step) * step
        lon_max = math.floor(self.bbox.east / step) * step

        lats = np.arange(lat_min, lat_max + step / 2, step)
        lons = np.arange(lon_min, lon_max + step / 2, step)
        return lats, lons

    # ── API calls ────────────────────────────────────────────────────────

    def _query_api(
        self,
        lats: list[float],
        lons: list[float],
        start: str,
        end: str,
    ) -> list[dict]:
        """Query Open-Meteo API for a batch of grid points.

        Args:
            lats: Latitudes of grid points in the batch.
            lons: Longitudes of grid points in the batch.
            start: Start date YYYY-MM-DD.
            end: End date YYYY-MM-DD.

        Returns:
            List of JSON response dicts, one per grid point.
        """
        try:
            import requests
        except ImportError:
            raise ImportError(
                "The 'requests' package is required for the openmeteo backend. "
                "Install it with: pip install ecmwf-downloader[openmeteo]"
            )
        from tenacity import retry, stop_after_attempt, wait_exponential

        # Build parameter list
        hourly_params = list(SURFACE_MAP.keys())

        if self._model == "ifs":
            for level in self.pressure_levels:
                for base_param in PLEV_MAP:
                    hourly_params.append(f"{base_param}_{level}hPa")
                # Wind components
                hourly_params.append(f"wind_speed_{level}hPa")
                hourly_params.append(f"wind_direction_{level}hPa")

        url = ERA5_ARCHIVE_URL if self._model == "era5" else IFS_FORECAST_URL

        params = {
            "latitude": ",".join(str(lat) for lat in lats),
            "longitude": ",".join(str(lon) for lon in lons),
            "hourly": ",".join(hourly_params),
            "start_date": start,
            "end_date": end,
            "timeformat": "unixtime",
            "wind_speed_unit": "ms",
        }

        # For IFS, add elevation for each point (Open-Meteo includes it)
        # We also request elevation for surface geopotential computation
        if self._model == "era5":
            params["models"] = "era5"

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(min=2, max=30),
            reraise=True,
        )
        def _do_request():
            self._rate_limiter.wait()
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json()

        data = _do_request()

        # Single point → wrap in list
        if isinstance(data, dict) and "latitude" in data and not isinstance(data["latitude"], list):
            return [data]

        # Multiple points: Open-Meteo returns a list
        if isinstance(data, list):
            return data

        # Multiple points may come as a dict with list fields
        # In the multi-point case, Open-Meteo returns a list at the top level
        return [data]

    # ── Pre-fetch ────────────────────────────────────────────────────────

    def _prefetch_all(self) -> None:
        """Pre-fetch the full date range for all grid points."""
        if self._cache_surf is not None:
            return

        lats, lons = self._build_grid()
        logger.info(
            "Open-Meteo %s: pre-fetching %s to %s for %d×%d grid (%d points)",
            self._model.upper(),
            self._start_date,
            self._end_date,
            len(lats),
            len(lons),
            len(lats) * len(lons),
        )

        # Build all (lat, lon) pairs
        grid_lats = []
        grid_lons = []
        for lat in lats:
            for lon in lons:
                grid_lats.append(float(lat))
                grid_lons.append(float(lon))

        # Batch API calls (parallel)
        n_batches = math.ceil(len(grid_lats) / self._batch_size)
        batch_indices = list(range(0, len(grid_lats), self._batch_size))
        max_api_workers = min(4, n_batches)

        all_responses = [None] * n_batches
        with ThreadPoolExecutor(max_workers=max_api_workers) as executor:
            futures = {}
            for batch_idx, i in enumerate(batch_indices):
                batch_lats = grid_lats[i : i + self._batch_size]
                batch_lons = grid_lons[i : i + self._batch_size]
                logger.debug(
                    "Submitting batch %d/%d (%d points)",
                    batch_idx + 1,
                    n_batches,
                    len(batch_lats),
                )
                fut = executor.submit(
                    self._query_api,
                    batch_lats, batch_lons,
                    self._start_date, self._end_date,
                )
                futures[fut] = batch_idx

            completed = as_completed(futures)
            if self.show_progress:
                completed = tqdm(
                    completed,
                    total=n_batches,
                    desc=f"OpenMeteo {self._model.upper()}",
                    unit="batch",
                    leave=False,
                )
            for fut in completed:
                all_responses[futures[fut]] = fut.result()

        # Flatten ordered batches
        all_responses = [resp for batch in all_responses for resp in batch]

        # Parse and assemble into gridded datasets
        self._cache_surf, self._cache_plev = self._assemble_grid(
            all_responses, lats, lons
        )

        logger.info(
            "Open-Meteo pre-fetch complete: SURF vars=%s, PLEV vars=%s",
            list(self._cache_surf.data_vars),
            list(self._cache_plev.data_vars) if self._cache_plev else [],
        )

    def _assemble_grid(
        self,
        responses: list[dict],
        lats: np.ndarray,
        lons: np.ndarray,
    ) -> tuple[xr.Dataset, xr.Dataset]:
        """Assemble API responses into gridded xarray Datasets.

        Args:
            responses: List of per-point JSON responses.
            lats: 1-D array of grid latitudes (sorted).
            lons: 1-D array of grid longitudes (sorted).

        Returns:
            (ds_surf, ds_plev) with standard dimension names and variable names.
        """
        # Parse time from the first response
        first = responses[0]
        times_unix = first["hourly"]["time"]
        times = pd.to_datetime(times_unix, unit="s", utc=True).tz_localize(None)
        n_times = len(times)
        n_lats = len(lats)
        n_lons = len(lons)

        # ── Surface variables ────────────────────────────────────────────
        surf_arrays: dict[str, np.ndarray] = {}
        for om_param, (era5_name, convert) in SURFACE_MAP.items():
            arr = np.full((n_times, n_lats, n_lons), np.nan, dtype="float32")
            for idx, resp in enumerate(responses):
                ilat = idx // n_lons
                ilon = idx % n_lons
                values = resp["hourly"].get(om_param)
                if values is not None:
                    col = np.array(values, dtype="float32")
                    col = np.where(np.isnan(col), np.nan, convert(col))
                    arr[:, ilat, ilon] = col
            surf_arrays[era5_name] = arr

        # Surface geopotential from elevation
        z_surf = np.full((n_times, n_lats, n_lons), np.nan, dtype="float32")
        for idx, resp in enumerate(responses):
            ilat = idx // n_lons
            ilon = idx % n_lons
            elev = resp.get("elevation", None)
            if elev is not None:
                z_surf[:, ilat, ilon] = float(elev) * _G
        surf_arrays["z"] = z_surf

        ds_surf = xr.Dataset(
            {
                name: (["time", "latitude", "longitude"], data)
                for name, data in surf_arrays.items()
            },
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
        )

        # ── Pressure-level variables (IFS mode only) ────────────────────
        ds_plev = xr.Dataset()

        if self._model == "ifs" and self.pressure_levels:
            levels = np.array(sorted(self.pressure_levels), dtype="int32")
            n_levels = len(levels)

            plev_arrays: dict[str, np.ndarray] = {}

            # Scalar plev variables (temperature, geopotential_height, relative_humidity)
            for base_param, (era5_name, convert) in PLEV_MAP.items():
                arr = np.full(
                    (n_times, n_levels, n_lats, n_lons), np.nan, dtype="float32"
                )
                for li, level in enumerate(levels):
                    om_key = f"{base_param}_{level}hPa"
                    for idx, resp in enumerate(responses):
                        ilat = idx // n_lons
                        ilon = idx % n_lons
                        values = resp["hourly"].get(om_key)
                        if values is not None:
                            col = np.array(values, dtype="float32")
                            col = np.where(np.isnan(col), np.nan, convert(col))
                            arr[:, li, ilat, ilon] = col
                plev_arrays[era5_name] = arr

            # Wind: speed + direction → u, v
            u_arr = np.full(
                (n_times, n_levels, n_lats, n_lons), np.nan, dtype="float32"
            )
            v_arr = np.full(
                (n_times, n_levels, n_lats, n_lons), np.nan, dtype="float32"
            )
            for li, level in enumerate(levels):
                speed_key = f"wind_speed_{level}hPa"
                dir_key = f"wind_direction_{level}hPa"
                for idx, resp in enumerate(responses):
                    ilat = idx // n_lons
                    ilon = idx % n_lons
                    speed_vals = resp["hourly"].get(speed_key)
                    dir_vals = resp["hourly"].get(dir_key)
                    if speed_vals is not None and dir_vals is not None:
                        speed = np.array(speed_vals, dtype="float32")
                        direction = np.array(dir_vals, dtype="float32")
                        dir_rad = np.deg2rad(direction)
                        u_arr[:, li, ilat, ilon] = -speed * np.sin(dir_rad)
                        v_arr[:, li, ilat, ilon] = -speed * np.cos(dir_rad)

            plev_arrays["u"] = u_arr
            plev_arrays["v"] = v_arr

            ds_plev = xr.Dataset(
                {
                    name: (["time", "level", "latitude", "longitude"], data)
                    for name, data in plev_arrays.items()
                },
                coords={
                    "time": times,
                    "level": levels,
                    "latitude": lats,
                    "longitude": lons,
                },
            )

        return ds_surf, ds_plev

    # ── fetch_day (main interface) ───────────────────────────────────────

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of data from the pre-fetched cache.

        On the first call, triggers a full pre-fetch of the configured date
        range.  Subsequent calls slice from the in-memory cache.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev) with standardised names and dims.
        """
        date = pd.Timestamp(date).normalize()

        with self._lock:
            if self._cache_surf is None:
                self._prefetch_all()

        res_hours = {"1H": 1, "2H": 2, "3H": 3, "6H": 6}[self.time_resolution]

        day_start = date
        day_end = date + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        # Slice wider to allow rolling sum of accumulated vars at day
        # boundaries (e.g. at 00:00 with 3H resolution we need hours 22,23
        # from the previous day).
        wider_start = day_start - pd.Timedelta(hours=res_hours - 1)
        ds_surf = self._cache_surf.sel(time=slice(wider_start, day_end))

        # For accumulated variables, sum hourly values over the time-
        # resolution window so the output represents the full-period
        # accumulation (matching ERA5 / IFS convention).
        if res_hours > 1:
            for var in _ACCUM_SURF_VARS:
                if var in ds_surf:
                    ds_surf[var] = (
                        ds_surf[var]
                        .rolling(time=res_hours, min_periods=1)
                        .sum()
                    )

        # Filter to requested time steps and trim back to this day
        ds_surf = ds_surf.sel(time=ds_surf.time.dt.hour.isin(self._time_steps))
        ds_surf = ds_surf.sel(time=slice(day_start, day_end))

        if self._cache_plev is not None and len(self._cache_plev.data_vars) > 0:
            ds_plev = self._cache_plev.sel(time=slice(day_start, day_end))
            ds_plev = ds_plev.sel(time=ds_plev.time.dt.hour.isin(self._time_steps))
        else:
            ds_plev = xr.Dataset()

        ds_surf = ds_surf.load()
        if ds_plev.data_vars:
            ds_plev = ds_plev.load()

        logger.info(
            "Open-Meteo %s: fetched %s — SURF vars=%s, PLEV vars=%s",
            self._model.upper(),
            date.strftime("%Y-%m-%d"),
            list(ds_surf.data_vars),
            list(ds_plev.data_vars) if ds_plev.data_vars else [],
        )

        return ds_surf, ds_plev

    def close(self):
        """Release cached data."""
        self._cache_surf = None
        self._cache_plev = None
