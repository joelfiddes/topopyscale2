"""Hybrid backend — fetches from multiple sources per variable group.

The hybrid backend optimizes for both speed and completeness by routing
different variable groups to different backends:

- Surface variables (t2m, d2m, sp, ssrd, u10, v10): OpenMeteo S3
- Precipitation (tp): IFS 9km (2022+) or OpenMeteo S3 (pre-2022)
- Pressure levels + longwave (strd): Google ARCO

Open-Meteo policy: the free Open-Meteo *REST API* is per-coordinate and rate
limits (HTTP 429) on area requests, so it is only preferred for point-like
domains (see ``_is_point_like``); for larger areas it is pushed to last and
Google ARCO / the un-throttled Open-Meteo S3 mirror lead. Any sub-backend fetch
failure additionally falls back to Google/CDS at download time rather than
aborting the run.

Special case: If bbox is within Central Asia and time ≤ 2023, uses s3zarr
for everything (fastest + complete).

See docs/backend-priority.md for the full priority logic.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pandas as pd
import xarray as xr

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ── Regional coverage ────────────────────────────────────────────────────────

# A regional pre-built Zarr store (read through the s3zarr backend) is used only when
# one is configured: ``regional_zarr_url`` plus, optionally, the store's extent and
# last date. There is no built-in store — see ``tps2 build-cache`` to make your own.

# ── Variable groups ──────────────────────────────────────────────────────────

SURFACE_VARS = {"t2m", "d2m", "sp", "ssrd", "u10", "v10", "msl"}
PRECIP_VARS = {"tp"}
PLEV_VARS = {"t", "z", "u", "v", "q", "r"}
STRD_VARS = {"strd"}

# Variables that must come from Google/CDS (OpenMeteo blacklisted)
GOOGLE_ONLY_VARS = {"strd", "q"}  # strd missing, q missing (only r available)
GOOGLE_ONLY_PLEV = {"u", "v"}    # 43m/s and 28m/s errors on OpenMeteo


def _source_label(backend) -> str:
    """Short provenance label for a sub-backend, e.g. 'openmeteo:ifs' or 'google'."""
    name = type(backend).__name__.replace("Backend", "").lower()
    model = (getattr(backend, "model", None) or getattr(backend, "_model", None)
             or getattr(backend, "dataset", None))
    return f"{name}:{model}" if model else name


def _bbox_within(inner: BBox, outer: BBox) -> bool:
    """Check if inner bbox is fully contained within outer bbox."""
    return (
        inner.west >= outer.west
        and inner.east <= outer.east
        and inner.south >= outer.south
        and inner.north <= outer.north
    )


# Open-Meteo's free REST API is a *per-coordinate* service: an area request fans
# out to many grid points and the free tier returns HTTP 429 (rate limited). It
# is excellent for point / single-cell sims and impractical for areas. We treat a
# bbox spanning <= ~1 ERA5 cell in each direction as "point-like" and only then
# let the REST API take priority; for anything larger we push it to last and
# prefer Google ARCO / the (un-throttled) Open-Meteo S3 mirror.
POINT_LIKE_MAX_SPAN_DEG = 0.3


def _is_point_like(bbox: BBox) -> bool:
    """True if *bbox* is small enough that the Open-Meteo REST API is safe to prefer."""
    return (
        (bbox.east - bbox.west) <= POINT_LIKE_MAX_SPAN_DEG
        and (bbox.north - bbox.south) <= POINT_LIKE_MAX_SPAN_DEG
    )


def _surface_candidates(bbox: BBox) -> list[str]:
    """Ordered surface-variable backends, deprioritising the rate-limited REST API for areas."""
    if _is_point_like(bbox):
        return ["openmeteo_s3", "openmeteo", "google", "cds"]
    # Area: skip the rate-limited Open-Meteo REST API up front; keep it as a last resort.
    return ["openmeteo_s3", "google", "cds", "openmeteo"]


def _precip_candidates(bbox: BBox, precip_model: str) -> list[str]:
    """Ordered precip backends for the configured model.

    Candidates are mirrors of ONE physical source, so falling back between them never
    changes the data. ERA5 is served by several mirrors; IFS 9 km only by the Open-Meteo
    REST API, so asking for IFS has no fallback — an unavailable IFS is an error, never
    a silent switch to ERA5.
    """
    if precip_model == "ifs":
        return ["openmeteo"]
    if _is_point_like(bbox):
        return ["openmeteo_s3", "openmeteo", "google", "cds"]
    # Area: push the rate-limited REST API to last.
    return ["openmeteo_s3", "google", "cds", "openmeteo"]


class HybridBackend(ERA5Backend):
    """Hybrid backend fetching from multiple sources per variable group.

    Routes different variables to the fastest mirror of the configured source:
    - Surface: ERA5 (OpenMeteo S3, Open-Meteo API, Google or CDS)
    - Precip: ERA5 by default; IFS 9 km only when ``precip_model="ifs"`` is configured
    - Pressure levels + strd: ERA5 (Google ARCO or CDS)

    The source is never chosen from the date or the domain; each fetched day records
    the backend that served every variable group in ``attrs["tps2_source_*"]``.

    Args:
        bbox: Bounding box (west, south, east, north).
        pressure_levels: Pressure levels to fetch (hPa).
        time_resolution: Time resolution ('1H', '3H', '6H').
        cache_dir: Cache directory for backends.
        include_strd: Whether to fetch longwave radiation (requires Google).
        **kwargs: Additional arguments passed to sub-backends.

    Note:
        If bbox is within Central Asia and end_date ≤ 2023, automatically
        uses s3zarr for all variables (fastest + complete).
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int] | None = None,
        time_resolution: str = "1H",
        cache_dir: str = "/tmp/hybrid_cache",
        include_strd: bool = True,
        start_date: str | pd.Timestamp | None = None,
        end_date: str | pd.Timestamp | None = None,
        regional_zarr_url: str | None = None,
        regional_bbox: tuple[float, float, float, float] | None = None,
        regional_end_date: str | pd.Timestamp | None = None,
        precip_model: str = "era5",
        **kwargs,
    ):
        super().__init__(bbox, pressure_levels or [], time_resolution, **kwargs)

        if precip_model not in ("era5", "ifs"):
            raise ValueError(f"precip_model must be 'era5' or 'ifs', got {precip_model!r}")
        self.precip_model = precip_model
        self.regional_zarr_url = regional_zarr_url or None
        self._regional_bbox = BBox.from_tuple(regional_bbox) if regional_bbox else None
        self._regional_end_date = pd.Timestamp(regional_end_date) if regional_end_date else None

        self.cache_dir = cache_dir
        self.include_strd = include_strd
        self._start_date = pd.Timestamp(start_date) if start_date else None
        self._end_date = pd.Timestamp(end_date) if end_date else None

        # Determine mode: regional (s3zarr) vs hybrid
        self._use_regional = self._check_regional_coverage()

        # Initialize sub-backends lazily
        self._surface_backend: ERA5Backend | None = None
        self._precip_backend: ERA5Backend | None = None
        self._plev_backend: ERA5Backend | None = None
        self._regional_backend: ERA5Backend | None = None
        self._fallback_backend: ERA5Backend | None = None

        logger.info(
            "HybridBackend: bbox=%s, plev=%s, regional=%s",
            (bbox.west, bbox.south, bbox.east, bbox.north),
            pressure_levels,
            self._use_regional,
        )

    def _check_regional_coverage(self) -> bool:
        """Check if the configured regional store can serve the whole request."""
        if not self.regional_zarr_url:
            return False
        if self._regional_bbox and not _bbox_within(self.bbox, self._regional_bbox):
            return False
        if self._regional_end_date and self._end_date and self._end_date > self._regional_end_date:
            return False
        # Check if s3zarr is available
        try:
            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import _check_available
            if not _check_available("s3zarr"):
                return False
        except ImportError:
            return False

        # Check if S3 credentials are configured
        import os
        has_creds = (
            os.environ.get("AWS_ACCESS_KEY_ID")
            and os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        if not has_creds:
            logger.warning(
                "S3 credentials not found (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY). "
                "Regional s3zarr backend disabled, falling back to Google ARCO-ERA5."
            )
            return False

        return True

    def _get_surface_backend(self) -> ERA5Backend:
        """Get or create the surface variables backend."""
        if self._surface_backend is None:
            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import (
                _check_available,
                get_backend,
            )

            # Try backends in order (REST API deprioritised for area requests).
            for name in _surface_candidates(self.bbox):
                if _check_available(name):
                    try:
                        kwargs = {
                            "bbox": self.bbox,
                            "cache_dir": self.cache_dir,
                            "pressure_levels": [],  # Surface only
                        }
                        if name == "openmeteo":
                            kwargs["model"] = "era5"
                            # OpenMeteo needs date range
                            if self._start_date and self._end_date:
                                kwargs["start_date"] = str(self._start_date.date())
                                kwargs["end_date"] = str(self._end_date.date())
                        if name == "openmeteo_s3":
                            # S3 backend doesn't need pressure_levels in kwargs
                            del kwargs["pressure_levels"]
                        self._surface_backend = get_backend(name, **kwargs)
                        logger.info("HybridBackend: surface backend = %s", name)
                        break
                    except Exception as e:
                        logger.warning("Failed to init %s for surface: %s", name, e)

            if self._surface_backend is None:
                raise RuntimeError("No surface backend available")

        return self._surface_backend

    def _get_precip_backend(self) -> ERA5Backend:
        """Get or create the precipitation backend for the configured precip_model."""
        use_ifs = self.precip_model == "ifs"

        if self._precip_backend is None:
            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import (
                _check_available,
                get_backend,
            )

            candidates = _precip_candidates(self.bbox, self.precip_model)

            for name in candidates:
                if _check_available(name):
                    try:
                        kwargs = {
                            "bbox": self.bbox,
                            "cache_dir": self.cache_dir,
                            "pressure_levels": [],
                        }
                        if name == "openmeteo" and use_ifs:
                            kwargs["model"] = "ifs"  # 9km precipitation
                            # OpenMeteo needs date range
                            if self._start_date and self._end_date:
                                kwargs["start_date"] = str(self._start_date.date())
                                kwargs["end_date"] = str(self._end_date.date())
                        elif name == "openmeteo":
                            kwargs["model"] = "era5"
                            if self._start_date and self._end_date:
                                kwargs["start_date"] = str(self._start_date.date())
                                kwargs["end_date"] = str(self._end_date.date())
                        if name == "openmeteo_s3":
                            del kwargs["pressure_levels"]
                        self._precip_backend = get_backend(name, **kwargs)
                        logger.info("HybridBackend: precip backend = %s (ifs=%s)", name, use_ifs)
                        break
                    except Exception as e:
                        logger.warning("Failed to init %s for precip: %s", name, e)

            if self._precip_backend is None:
                raise RuntimeError(
                    f"No backend available for precip_model={self.precip_model!r} "
                    f"(tried {candidates})"
                )

        return self._precip_backend

    def _get_plev_backend(self) -> ERA5Backend:
        """Get or create the pressure level + strd backend."""
        if self._plev_backend is None:
            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import (
                _check_available,
                get_backend,
            )

            # Only Google/CDS have complete plev + strd
            # Use explicit variable lists to avoid relative_humidity issue
            plev_vars = [
                "temperature",
                "geopotential",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
            ]

            for name in ["google", "cds"]:
                if _check_available(name):
                    try:
                        self._plev_backend = get_backend(
                            name,
                            bbox=self.bbox,
                            pressure_levels=self.pressure_levels,
                            cache_dir=self.cache_dir,
                            plev_vars=plev_vars,
                        )
                        logger.info("HybridBackend: plev backend = %s", name)
                        break
                    except Exception as e:
                        logger.warning("Failed to init %s for plev: %s", name, e)

            if self._plev_backend is None:
                raise RuntimeError("No plev backend available (need Google or CDS)")

        return self._plev_backend

    def _get_fallback_backend(self) -> ERA5Backend | None:
        """Get or create a Google/CDS backend used to recover from sub-backend fetch failures.

        Google/CDS are not per-coordinate rate-limited, so they make a reliable
        fallback when e.g. the Open-Meteo REST API returns HTTP 429 mid-download.
        Returns None if neither is available.
        """
        if self._fallback_backend is None:
            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import (
                _check_available,
                get_backend,
            )

            for name in ["google", "cds"]:
                if _check_available(name):
                    try:
                        self._fallback_backend = get_backend(
                            name,
                            bbox=self.bbox,
                            pressure_levels=self.pressure_levels,
                            cache_dir=self.cache_dir,
                        )
                        logger.info("HybridBackend: fallback backend = %s", name)
                        break
                    except Exception as e:
                        logger.warning("Failed to init %s as fallback: %s", name, e)

        return self._fallback_backend

    def _get_regional_backend(self) -> ERA5Backend:
        """Get or create the regional (s3zarr) backend."""
        if self._regional_backend is None:
            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import get_backend

            self._regional_backend = get_backend(
                "s3zarr",
                bbox=self.bbox,
                pressure_levels=self.pressure_levels,
                cache_dir=self.cache_dir,
                zarr_url=self.regional_zarr_url,
            )
            logger.info("HybridBackend: using regional s3zarr backend at %s", self.regional_zarr_url)

        return self._regional_backend

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data from multiple backends.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev).
        """
        date = pd.Timestamp(date).normalize()

        # ─── Regional mode: s3zarr for everything ────────────────────────
        if self._use_regional:
            backend = self._get_regional_backend()
            ds_surf, ds_plev = backend.fetch_day(date)
            ds_surf.attrs["tps2_source_all"] = f"s3zarr:{self.regional_zarr_url}"
            return ds_surf, ds_plev

        # ─── Hybrid mode: multiple backends (parallel dispatch) ─────────

        # Initialize backends (must happen before thread dispatch)
        surface_backend = self._get_surface_backend()
        precip_backend = self._get_precip_backend()
        sources = {
            "surface": _source_label(surface_backend),
            "precip": _source_label(precip_backend),
        }
        need_plev = bool(self.pressure_levels or self.include_strd)
        plev_backend = self._get_plev_backend() if need_plev else None

        # Launch independent fetches concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_surf = executor.submit(surface_backend.fetch_day, date)

            # Only launch precip separately if it's a different backend
            fut_precip = None
            if precip_backend is not surface_backend:
                fut_precip = executor.submit(precip_backend.fetch_day, date)

            fut_plev = None
            if plev_backend is not None:
                fut_plev = executor.submit(plev_backend.fetch_day, date)

        # Collect results — recover from a sub-backend fetch failure (e.g. an
        # Open-Meteo HTTP 429) by falling back to Google/CDS rather than crashing
        # the whole pipeline.
        try:
            ds_surf_base, _ = fut_surf.result()
        except Exception as e:
            fallback = self._get_fallback_backend()
            if fallback is None:
                raise
            logger.warning(
                "Surface backend %s failed for %s (%s); falling back to %s.",
                type(surface_backend).__name__,
                date.strftime("%Y-%m-%d"),
                e,
                type(fallback).__name__,
            )
            ds_surf_base, _ = fallback.fetch_day(date)
            sources["surface"] = _source_label(fallback)
            if precip_backend is surface_backend:
                sources["precip"] = _source_label(fallback)

        if fut_precip is not None:
            try:
                ds_precip, _ = fut_precip.result()
                if "tp" in ds_precip:
                    ds_surf_base["tp"] = ds_precip["tp"]
            except Exception as e:
                fallback = self._get_fallback_backend()
                if fallback is None or self.precip_model != "era5":
                    # The fallback serves ERA5; substituting it for another model
                    # would change the precipitation source silently.
                    raise
                logger.warning(
                    "Precip backend %s failed for %s (%s); falling back to %s.",
                    type(precip_backend).__name__,
                    date.strftime("%Y-%m-%d"),
                    e,
                    type(fallback).__name__,
                )
                ds_precip_fb, _ = fallback.fetch_day(date)
                if "tp" in ds_precip_fb:
                    ds_surf_base["tp"] = ds_precip_fb["tp"]
                sources["precip"] = _source_label(fallback)

        ds_plev = xr.Dataset()
        if fut_plev is not None:
            ds_surf_google, ds_plev = fut_plev.result()

            # Add strd from Google to surface dataset
            if self.include_strd and "strd" in ds_surf_google:
                # Align coordinates before merging
                strd = ds_surf_google["strd"]
                # Interpolate to match surface backend grid if needed
                if not _coords_match(ds_surf_base, ds_surf_google):
                    strd = strd.interp(
                        latitude=ds_surf_base.latitude,
                        longitude=ds_surf_base.longitude,
                        method="linear",
                    )
                ds_surf_base["strd"] = strd
            sources["plev"] = _source_label(plev_backend)
            if self.include_strd:
                sources["strd"] = _source_label(plev_backend)

        # Provenance: which backend served each variable group on this day.
        for group, label in sources.items():
            ds_surf_base.attrs[f"tps2_source_{group}"] = label
            if ds_plev is not None and len(ds_plev.data_vars):
                ds_plev.attrs[f"tps2_source_{group}"] = label

        logger.info(
            "HybridBackend: fetched %s — SURF vars=%s, PLEV vars=%s",
            date.strftime("%Y-%m-%d"),
            list(ds_surf_base.data_vars),
            list(ds_plev.data_vars) if ds_plev else [],
        )

        return ds_surf_base, ds_plev

    def close(self):
        """Clean up all sub-backends."""
        for backend in [
            self._surface_backend,
            self._precip_backend,
            self._plev_backend,
            self._regional_backend,
            self._fallback_backend,
        ]:
            if backend is not None:
                try:
                    backend.close()
                except Exception:
                    pass


def _coords_match(ds1: xr.Dataset, ds2: xr.Dataset, tol: float = 0.01) -> bool:
    """Check if two datasets have matching lat/lon coordinates."""
    try:
        lat1 = ds1.latitude.values
        lat2 = ds2.latitude.values
        lon1 = ds1.longitude.values
        lon2 = ds2.longitude.values

        if len(lat1) != len(lat2) or len(lon1) != len(lon2):
            return False

        return (
            abs(lat1 - lat2).max() < tol
            and abs(lon1 - lon2).max() < tol
        )
    except Exception:
        return False
