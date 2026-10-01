"""CacheBuilder — build and manage regional ERA5 Zarr caches.

Downloads ERA5 data via ERA5Loader and merges daily stores into a single
regional Zarr cache with optimised chunking for local/mounted access.

Usage::

    builder = CacheBuilder(
        backend="google",
        bbox=(60, 25, 105, 50),
        start_date="2000-01-01",
        end_date="2024-12-31",
        pressure_levels=[1000, 925, 850, 800, 700, 600, 550, 500, 400, 300],
        output_path=Path("/data/era5/hma.zarr"),
    )
    builder.build()

    # Later: extend the cache
    CacheBuilder.update(Path("/data/era5/hma.zarr"), backend="google")

    # Inspect
    CacheBuilder.info(Path("/data/era5/hma.zarr"))
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class CacheBuilder:
    """Build and manage a regional ERA5 Zarr cache.

    The cache is a single Zarr store containing both surface and pressure-level
    variables with regional chunking optimised for spatial access patterns.
    """

    DEFAULT_CHUNKS = {
        "time": 24,
        "latitude": 45,
        "longitude": 45,
        "level": -1,
    }

    SURF_VARS = [
        "2m_temperature",
        "2m_dewpoint_temperature",
        "surface_pressure",
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "total_precipitation",
        "geopotential",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "toa_incident_solar_radiation",
    ]

    PLEV_VARS = [
        "temperature",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "specific_humidity",
    ]

    # Short names as stored in the merged Zarr (after ZarrWriter processing)
    SURF_SHORT = ["d2m", "sp", "ssrd", "strd", "t2m", "tp", "z_surf", "u10", "v10", "tisr"]
    PLEV_SHORT = ["q", "t", "u", "v", "z"]

    def __init__(
        self,
        backend: str,
        bbox: tuple[float, float, float, float],
        start_date: str,
        end_date: str,
        pressure_levels: list[int],
        output_path: Path,
        chunks: dict | None = None,
        max_workers: int = 4,
        time_resolution: str = "1H",
        backend_kwargs: dict | None = None,
    ):
        self.backend = backend
        self.bbox = bbox
        self.start_date = start_date
        self.end_date = end_date
        self.pressure_levels = sorted(pressure_levels)
        self.output_path = Path(output_path)
        self.chunks = chunks or dict(self.DEFAULT_CHUNKS)
        self.max_workers = max_workers
        self.time_resolution = time_resolution
        self.backend_kwargs = backend_kwargs or {}

    def build(self, append: bool = False) -> None:
        """Run the full cache build: download via ERA5Loader then merge.

        When ``append`` is True, the freshly downloaded daily stores are
        appended in-place to an existing cache along the time axis (no full
        rewrite). Used by :meth:`update` to keep weekly refreshes cheap.
        """
        from topopyscale2.inputs.nwp_downloader import ERA5Loader

        staging_dir = self.output_path.parent / ".cache_staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Building cache: bbox=%s, time=%s to %s, levels=%s",
            self.bbox, self.start_date, self.end_date, self.pressure_levels,
        )
        logger.info("Staging directory: %s", staging_dir)

        # Build backend kwargs for Google backend (need to pass variable lists)
        download_kwargs = dict(self.backend_kwargs)
        if self.backend == "google":
            download_kwargs.setdefault("surf_vars", self.SURF_VARS)
            download_kwargs.setdefault("plev_vars", self.PLEV_VARS)

        loader = ERA5Loader(
            backend=self.backend,
            bbox=self.bbox,
            start_date=self.start_date,
            end_date=self.end_date,
            pressure_levels=self.pressure_levels,
            output_format="zarr",
            output_dir=str(staging_dir),
            time_resolution=self.time_resolution,
            max_workers=self.max_workers,
            compute_rh=False,  # Store raw q, compute RH at read time
            writer_kwargs={"merge": False, "cleanup_daily": False},
            backend_kwargs=download_kwargs,
        )
        loader.download()

        if append:
            self._append_to_cache(staging_dir)
        else:
            self._merge_to_cache(staging_dir)

    def _merge_to_cache(self, staging_dir: Path) -> None:
        """Merge daily Zarr stores into the final regional cache.

        Batches by year to avoid opening thousands of stores at once,
        which causes dask task graph explosion and memory exhaustion.
        """
        import zarr

        daily_dir = staging_dir / "daily"
        daily_stores = sorted(daily_dir.glob("day_*.zarr"))

        if not daily_stores:
            logger.warning("No daily stores found in staging directory")
            return

        # Group daily stores by year from filename: day_YYYYMMDD.zarr
        from collections import defaultdict
        yearly_groups: dict[str, list[Path]] = defaultdict(list)
        for p in daily_stores:
            year = p.stem.split("_")[1][:4]  # "day_20200101" -> "2000"
            yearly_groups[year].append(p)

        years = sorted(yearly_groups.keys())
        logger.info(
            "Merging %d daily stores (%d years: %s–%s) into cache",
            len(daily_stores), len(years), years[0], years[-1],
        )

        # Build compression encoding helper
        compressor, comp_key = self._get_compressor()

        # Cache metadata (written on first batch, preserved on appends)
        cache_attrs = {
            "tps2_cache_version": "1.0",
            "tps2_surf_vars": self.SURF_SHORT,
            "tps2_plev_vars": self.PLEV_SHORT,
            "tps2_bbox": list(self.bbox),
            "tps2_pressure_levels": self.pressure_levels,
            "tps2_time_resolution": self.time_resolution,
        }

        # Write to temp path, then atomic swap at the end
        tmp_path = self.output_path.with_suffix(".zarr.tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        # Include existing cache as first "year" if updating
        existing_sources = []
        if self.output_path.exists():
            existing_sources.append(str(self.output_path))
            logger.info("Including existing cache in merge: %s", self.output_path)

        for i, year in enumerate(years):
            stores = yearly_groups[year]
            sources = [str(p) for p in stores]

            # On first batch, include existing cache if present
            if i == 0 and existing_sources:
                sources = existing_sources + sources

            logger.info(
                "Batch %d/%d: year %s (%d stores)",
                i + 1, len(years), year, len(stores),
            )

            ds = xr.open_mfdataset(
                sources,
                engine="zarr",
                combine="by_coords",
                parallel=True,
                preprocess=self._drop_expver,
            )

            ds = ds.sortby("time")

            # Surface vars are static w.r.t. pressure level, but
            # combine="by_coords" can spuriously broadcast a static surface var
            # (e.g. z_surf) across the plev 'level' axis on some batches. That
            # makes the first-written year's dims (time,lat,lon) disagree with a
            # later append's (level,time,lat,lon) and breaks to_zarr(append).
            # Collapse any stray level dim so every batch writes surf vars alike.
            for v in self.SURF_SHORT:
                if v in ds.data_vars and "level" in ds[v].dims:
                    ds[v] = ds[v].isel(level=0, drop=True)

            # Rechunk for regional access patterns
            chunk_spec = {k: v for k, v in self.chunks.items() if k in ds.dims}
            ds = ds.chunk(chunk_spec)

            # Strip encoding from source datasets
            for var in ds.data_vars:
                ds[var].encoding.clear()
            for coord in ds.coords:
                ds[coord].encoding.clear()

            # Build per-variable compression encoding
            encoding = {}
            if compressor is not None:
                for var in ds.data_vars:
                    encoding[var] = {comp_key: compressor}

            if i == 0:
                # First batch: create the store
                ds.attrs.update(cache_attrs)
                ds.to_zarr(
                    str(tmp_path),
                    mode="w",
                    encoding=encoding if encoding else None,
                )
            else:
                # Subsequent batches: append along time
                ds.to_zarr(
                    str(tmp_path),
                    mode="a",
                    append_dim="time",
                )

            ds.close()
            logger.info("Batch %d/%d complete (year %s)", i + 1, len(years), year)

        # Atomic replace
        if self.output_path.exists():
            shutil.rmtree(self.output_path)
        tmp_path.rename(self.output_path)

        # Consolidate metadata
        try:
            zarr.consolidate_metadata(str(self.output_path))
        except Exception as e:
            logger.warning("Failed to consolidate metadata: %s", e)

        logger.info("Cache written: %s", self.output_path)

        # Clean up staging
        logger.info("Cleaning up staging directory")
        shutil.rmtree(staging_dir)

    def _append_to_cache(self, staging_dir: Path) -> None:
        """Append freshly downloaded daily stores in-place to an existing cache.

        Unlike :meth:`_merge_to_cache`, this does NOT rewrite the whole store:
        it streams only the new daily stores onto the time axis via
        ``to_zarr(mode="a", append_dim="time")``. The existing store ends on a
        24-hour chunk boundary and we only append whole days, so chunks stay
        aligned. Variable set, level order and lat/lon order are reindexed to
        match the existing store before writing.
        """
        import numpy as np
        import zarr

        daily_dir = staging_dir / "daily"
        daily_stores = sorted(daily_dir.glob("day_*.zarr"))
        if not daily_stores:
            logger.warning("No daily stores found in staging directory")
            return
        if not self.output_path.exists():
            raise FileNotFoundError(
                f"Append requires an existing cache: {self.output_path}"
            )

        logger.info(
            "Appending %d daily stores in-place to %s",
            len(daily_stores), self.output_path,
        )

        ds = xr.open_mfdataset(
            [str(p) for p in daily_stores],
            engine="zarr",
            combine="by_coords",
            parallel=True,
            preprocess=self._drop_expver,
        )
        ds = ds.sortby("time")

        # Align structure to the existing store
        existing = xr.open_zarr(str(self.output_path), chunks={})
        ds = ds[[v for v in existing.data_vars if v in ds.data_vars]]
        # Guard against combine_by_coords broadcasting static surface vars
        # across the plev 'level' axis (see _merge_to_cache); the existing
        # store has them as (time,lat,lon) so a level dim would break append.
        for v in self.SURF_SHORT:
            if v in ds.data_vars and "level" in ds[v].dims:
                ds[v] = ds[v].isel(level=0, drop=True)
        if "level" in ds.dims and "level" in existing.dims:
            ds = ds.reindex(level=existing.level.values)
        for d in ("latitude", "longitude"):
            if d in ds.dims and d in existing.dims:
                ds = ds.reindex({d: existing[d].values})

        # Only keep timesteps strictly after the last cached one (no overlap)
        last_t = np.datetime64(pd.Timestamp(existing.time.values[-1]))
        existing.close()
        mask = ds.time.values > last_t
        if not mask.any():
            logger.info("No new timesteps to append; cache already current.")
            ds.close()
            shutil.rmtree(staging_dir)
            return
        ds = ds.isel(time=mask)

        # Match cache chunking; clear source encoding so existing array
        # encoding (compressor/chunks) is used on append.
        chunk_spec = {k: v for k, v in self.chunks.items() if k in ds.dims}
        ds = ds.chunk(chunk_spec)
        for var in ds.data_vars:
            ds[var].encoding.clear()
        for coord in ds.coords:
            ds[coord].encoding.clear()

        n_new = int(ds.time.size)
        ds.to_zarr(str(self.output_path), mode="a", append_dim="time")
        ds.close()

        # Backfill metadata attrs (older caches may lack them) and re-consolidate
        self._ensure_cache_attrs()
        try:
            zarr.consolidate_metadata(str(self.output_path))
        except Exception as e:
            logger.warning("Failed to consolidate metadata: %s", e)

        logger.info("Appended %d new timesteps to cache: %s", n_new, self.output_path)

        logger.info("Cleaning up staging directory")
        shutil.rmtree(staging_dir)

    def _ensure_cache_attrs(self) -> None:
        """Write the tps2_* cache metadata attrs if missing (idempotent)."""
        import zarr

        grp = zarr.open_group(str(self.output_path), mode="r+")
        wanted = {
            "tps2_cache_version": "1.0",
            "tps2_surf_vars": self.SURF_SHORT,
            "tps2_plev_vars": self.PLEV_SHORT,
            "tps2_bbox": list(self.bbox),
            "tps2_pressure_levels": self.pressure_levels,
            "tps2_time_resolution": self.time_resolution,
        }
        for k, v in wanted.items():
            if k not in grp.attrs:
                grp.attrs[k] = v

    @staticmethod
    def _last_available_day(
        backend: str,
        bbox: tuple[float, float, float, float],
        min_lag_days: int = 5,
        max_lag_days: int = 20,
    ) -> pd.Timestamp:
        """Most recent day with real data available from the backend.

        ERA5 (incl. the ARCO mirror) lags the present and the lag drifts, so a
        fixed ``today - 5`` can request not-yet-published days. For the google
        backend this probes the ARCO store at the region centre, scanning back
        from ``today - min_lag_days`` until it finds a finite value. Falls back
        to ``today - min_lag_days`` for other backends or on probe failure.
        """
        today = pd.Timestamp.now().normalize()
        if backend == "google":
            try:
                import numpy as np

                arco = xr.open_zarr(
                    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
                    chunks={},
                    storage_options={"token": "anon"},
                )
                lat0 = (bbox[1] + bbox[3]) / 2.0
                lon0 = ((bbox[0] + bbox[2]) / 2.0) % 360.0
                for lag in range(min_lag_days, max_lag_days + 1):
                    day = today - pd.Timedelta(days=lag)
                    val = arco["2m_temperature"].sel(
                        time=day + pd.Timedelta(hours=12),
                        latitude=lat0,
                        longitude=lon0,
                        method="nearest",
                    ).values
                    if np.isfinite(val):
                        logger.info("Last available ARCO day: %s (lag %d d)",
                                    day.strftime("%Y-%m-%d"), lag)
                        return day
                logger.warning(
                    "No finite ARCO data within %d days; using today-%d",
                    max_lag_days, min_lag_days,
                )
            except Exception as e:
                logger.warning(
                    "ARCO availability probe failed (%s); using today-%d",
                    e, min_lag_days,
                )
        return today - pd.Timedelta(days=min_lag_days)

    @staticmethod
    def _drop_expver(ds):
        """Drop expver coordinate/dimension that breaks combine_by_coords."""
        import numpy as np

        if "expver" in ds.dims:
            ds_merged = ds.isel(expver=0).copy()
            for var in ds.data_vars:
                if "expver" in ds[var].dims:
                    v0 = ds[var].isel(expver=0).values
                    v1 = ds[var].isel(expver=1).values
                    ds_merged[var].values = np.where(np.isnan(v0), v1, v0)
            ds = ds_merged
        if "expver" in ds.coords:
            ds = ds.drop_vars("expver")
        return ds

    @staticmethod
    def _get_compressor():
        """Get compressor, handling zarr 2.x vs 3.x."""
        import zarr

        zarr_version = int(zarr.__version__.split(".")[0])

        if zarr_version >= 3:
            try:
                from zarr.codecs import ZstdCodec
                return ZstdCodec(level=3), "compressors"
            except ImportError:
                return None, "compressors"
        else:
            try:
                import numcodecs
                return numcodecs.Zstd(level=3), "compressor"
            except ImportError:
                return None, "compressor"

    @classmethod
    def update(
        cls,
        output_path: Path,
        backend: str = "google",
        max_workers: int = 4,
        backend_kwargs: dict | None = None,
    ) -> None:
        """Extend an existing cache with new dates up to ERA5 availability.

        Reads bbox, levels, and resolution from existing store attrs,
        then downloads from last date + 1 day to ~5 days before today.
        """
        output_path = Path(output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Cache not found: {output_path}")

        ds = xr.open_zarr(str(output_path), chunks={})
        attrs = ds.attrs

        # bbox / levels / resolution: prefer stored attrs, fall back to coords
        # (older caches were built without the tps2_* attrs).
        if attrs.get("tps2_bbox"):
            bbox = tuple(attrs["tps2_bbox"])
        else:
            bbox = (
                float(ds.longitude.values.min()),
                float(ds.latitude.values.min()),
                float(ds.longitude.values.max()),
                float(ds.latitude.values.max()),
            )
            logger.warning("tps2_bbox attr missing; derived from coords: %s", bbox)

        if attrs.get("tps2_pressure_levels"):
            pressure_levels = list(attrs["tps2_pressure_levels"])
        elif "level" in ds.dims:
            pressure_levels = sorted(int(x) for x in ds.level.values.tolist())
            logger.warning(
                "tps2_pressure_levels attr missing; derived from coords: %s",
                pressure_levels,
            )
        else:
            pressure_levels = []

        time_resolution = attrs.get("tps2_time_resolution", "1H")

        # Find last date in cache
        last_time = pd.Timestamp(ds.time.values[-1])
        ds.close()

        # Start from the day after the last complete day
        new_start = (last_time.normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # End at the last day actually available from the backend (lag drifts).
        new_end = cls._last_available_day(backend, bbox).strftime("%Y-%m-%d")

        if new_start > new_end:
            logger.info(
                "Cache is already up to date (last: %s, available: %s)",
                last_time.strftime("%Y-%m-%d"), new_end,
            )
            return

        logger.info(
            "Updating cache: %s to %s (last in cache: %s)",
            new_start, new_end, last_time.strftime("%Y-%m-%d"),
        )

        builder = cls(
            backend=backend,
            bbox=bbox,
            start_date=new_start,
            end_date=new_end,
            pressure_levels=pressure_levels,
            output_path=output_path,
            max_workers=max_workers,
            time_resolution=time_resolution,
            backend_kwargs=backend_kwargs or {},
        )
        builder.build(append=True)

    @staticmethod
    def info(output_path: Path) -> dict:
        """Get metadata about an existing cache store.

        Returns dict with: bbox, time_range, n_days, expected_days,
        missing_days, pressure_levels, surf_vars, plev_vars,
        time_resolution, chunks, size_gb, cache_version.
        """
        output_path = Path(output_path)
        if not output_path.exists():
            raise FileNotFoundError(f"Cache not found: {output_path}")

        ds = xr.open_zarr(str(output_path), chunks={})
        attrs = ds.attrs

        time_vals = pd.DatetimeIndex(ds.time.values)
        first = time_vals[0]
        last = time_vals[-1]

        # Count unique days
        unique_days = time_vals.normalize().unique()
        n_days = len(unique_days)

        # Expected days (inclusive)
        expected = pd.date_range(first.normalize(), last.normalize(), freq="D")
        expected_days = len(expected)
        missing_days = sorted(
            d.strftime("%Y-%m-%d")
            for d in expected
            if d not in unique_days
        )

        # Get chunks from first data variable
        chunks = {}
        for var in ds.data_vars:
            if hasattr(ds[var], "encoding") and "chunks" in ds[var].encoding:
                chunks = dict(zip(ds[var].dims, ds[var].encoding["chunks"]))
                break

        # Compute size on disk
        size_bytes = sum(
            f.stat().st_size
            for f in output_path.rglob("*")
            if f.is_file()
        )

        # Pressure levels
        pressure_levels = []
        if "level" in ds.dims:
            pressure_levels = sorted(ds.level.values.tolist())

        result = {
            "bbox": list(attrs.get("tps2_bbox", [])),
            "time_range": [first.strftime("%Y-%m-%d"), last.strftime("%Y-%m-%d")],
            "n_days": n_days,
            "expected_days": expected_days,
            "missing_days": missing_days,
            "pressure_levels": pressure_levels,
            "surf_vars": list(attrs.get("tps2_surf_vars", [])),
            "plev_vars": list(attrs.get("tps2_plev_vars", [])),
            "time_resolution": attrs.get("tps2_time_resolution", "unknown"),
            "chunks": chunks,
            "size_gb": round(size_bytes / (1024**3), 2),
            "cache_version": attrs.get("tps2_cache_version", "unknown"),
        }

        ds.close()
        return result
