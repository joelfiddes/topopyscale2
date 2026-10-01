"""Output writers for ERA5 data: NetCDF and Zarr formats."""

from __future__ import annotations

import logging
import shutil
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


class BaseWriter(ABC):
    """Abstract writer interface."""

    @abstractmethod
    def write_day(
        self, ds_surf: xr.Dataset, ds_plev: xr.Dataset, date_str: str
    ) -> None:
        """Write one day of data."""
        ...

    @abstractmethod
    def finalize(self) -> None:
        """Finalize output (merge, consolidate, etc.)."""
        ...


class NetCDFWriter(BaseWriter):
    """Write ERA5 data as daily netCDF files, merged to yearly files.

    Output structure:
        {output_dir}/daily/dSURF_YYYYMMDD.nc
        {output_dir}/daily/dPLEV_YYYYMMDD.nc
        {output_dir}/yearly/SURF_YYYY.nc  (after finalize)
        {output_dir}/yearly/PLEV_YYYY.nc  (after finalize)

    Args:
        output_dir: Base output directory.
        complevel: zlib compression level (1-9). Default 1.
    """

    def __init__(self, output_dir: str | Path, complevel: int = 1):
        self.output_dir = Path(output_dir)
        self.complevel = complevel

        self.daily_dir = self.output_dir / "daily"
        self.yearly_dir = self.output_dir / "yearly"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.yearly_dir.mkdir(parents=True, exist_ok=True)

        self._written_dates: list[str] = []

    def _encoding(self, ds: xr.Dataset) -> dict:
        """Build per-variable encoding dict for netCDF compression."""
        return {
            var: {"zlib": True, "complevel": self.complevel, "dtype": "float32"}
            for var in ds.data_vars
        }

    def _atomic_write(self, ds: xr.Dataset, target: Path) -> None:
        """Write dataset atomically via temp file + rename."""
        tmp_fd, tmp_path = tempfile.mkstemp(
            suffix=".nc.tmp", dir=str(target.parent)
        )
        try:
            import os
            os.close(tmp_fd)
            ds.to_netcdf(tmp_path, encoding=self._encoding(ds), engine="netcdf4")
            shutil.move(tmp_path, str(target))
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def write_day(
        self, ds_surf: xr.Dataset, ds_plev: xr.Dataset, date_str: str
    ) -> None:
        """Write daily SURF and PLEV netCDF files.

        Args:
            ds_surf: Surface dataset.
            ds_plev: Pressure level dataset.
            date_str: Date string in YYYYMMDD format.
        """
        surf_path = self.daily_dir / f"dSURF_{date_str}.nc"
        plev_path = self.daily_dir / f"dPLEV_{date_str}.nc"

        if ds_surf.data_vars:
            self._atomic_write(ds_surf, surf_path)
            logger.debug("Wrote %s", surf_path.name)

        if ds_plev.data_vars:
            self._atomic_write(ds_plev, plev_path)
            logger.debug("Wrote %s", plev_path.name)

        self._written_dates.append(date_str)

    def finalize(self) -> None:
        """Merge daily files into yearly files using xr.open_mfdataset.

        Replaces CDO mergetime with pure xarray.
        Checks for consistent pressure levels across files before merging.
        """
        # Collect years from written dates
        years = set()
        for d in self._written_dates:
            years.add(d[:4])

        # Also discover years from existing daily files
        for f in self.daily_dir.glob("dSURF_*.nc"):
            years.add(f.stem.split("_")[1][:4])

        for year in sorted(years):
            for prefix, out_prefix in [("dSURF", "SURF"), ("dPLEV", "PLEV")]:
                pattern = f"{prefix}_{year}*.nc"
                files = sorted(self.daily_dir.glob(pattern))
                if not files:
                    continue

                # Check consistency of pressure levels for PLEV files
                if prefix == "dPLEV":
                    files = self._filter_consistent_levels(files)
                    if not files:
                        logger.warning("No consistent PLEV files found for %s", year)
                        continue

                out_path = self.yearly_dir / f"{out_prefix}_{year}.nc"
                logger.info("Merging %d files → %s", len(files), out_path.name)

                ds = xr.open_mfdataset(
                    [str(f) for f in files],
                    combine="by_coords",
                    parallel=True,
                )
                # Write lazily — dask computes and writes chunk by chunk
                # without loading the entire year into RAM.
                self._atomic_write(ds, out_path)
                ds.close()

        logger.info("NetCDF yearly merge complete")

    def _filter_consistent_levels(self, files: list) -> list:
        """Filter PLEV files to only those with consistent pressure levels.

        Checks all files have the same levels. If inconsistent, uses the most
        common level set and discards files with different levels.

        Args:
            files: List of PLEV netCDF file paths.

        Returns:
            Filtered list of files with consistent levels.
        """
        if not files:
            return files

        # Get levels from each file
        level_sets = {}
        for f in files:
            try:
                with xr.open_dataset(f) as ds:
                    if "level" in ds.dims:
                        levels = tuple(sorted(ds.level.values.tolist()))
                        if levels not in level_sets:
                            level_sets[levels] = []
                        level_sets[levels].append(f)
            except Exception as e:
                logger.warning("Failed to read %s: %s", f, e)

        if not level_sets:
            return files

        # If all consistent, return all
        if len(level_sets) == 1:
            return files

        # Find the most common level set
        most_common = max(level_sets.items(), key=lambda x: len(x[1]))
        levels, consistent_files = most_common

        # Log the inconsistency
        total = sum(len(v) for v in level_sets.values())
        discarded = total - len(consistent_files)
        logger.warning(
            "Pressure level inconsistency: %d files have levels %s, "
            "discarding %d files with different levels",
            len(consistent_files), list(levels), discarded
        )
        for lvls, flist in level_sets.items():
            if lvls != levels:
                logger.warning("  Discarding %d files with levels %s", len(flist), list(lvls))

        return consistent_files


class ZarrWriter(BaseWriter):
    """Write ERA5 data as daily Zarr stores, merged to single store.

    Output structure:
        {output_dir}/daily/day_YYYYMMDD.zarr  (per-day stores)
        {output_dir}/ERA5.zarr                (merged, after finalize)

    This approach is parallel-safe: each day writes to its own store,
    avoiding race conditions when downloading with multiple threads.

    Surface geopotential stored as 'z_surf' to avoid name clash with
    pressure-level geopotential.

    Uses zstd compression for good balance of speed and ratio.

    Args:
        output_dir: Base output directory.
        chunks: Chunk sizes for the merged zarr store.
        merge: Whether to merge daily stores in finalize(). Default True.
        cleanup_daily: Whether to delete daily stores after successful merge. Default True.
    """

    def __init__(
        self,
        output_dir: str | Path,
        chunks: dict | None = None,
        merge: bool = True,
        cleanup_daily: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.daily_dir = self.output_dir / "daily"
        self.daily_dir.mkdir(parents=True, exist_ok=True)
        self.zarr_path = self.output_dir / "ERA5.zarr"
        self.merge = merge
        self.cleanup_daily = cleanup_daily

        # Chunks optimized for typical regional domains
        # time=24 matches daily download granularity
        # level=-1 keeps full vertical profile together
        # lat/lon=-1 for small domains, adjust for large domains
        self.chunks = chunks or {
            "time": 24,
            "latitude": -1,
            "longitude": -1,
            "level": -1,
        }

        self._written_dates: list[str] = []

    def _get_compressor(self):
        """Get compressor, handling different zarr versions.

        Returns (compressor, key_name) where key_name is 'compressors' for
        zarr 3.x or 'compressor' for zarr 2.x.
        """
        import zarr
        zarr_version = int(zarr.__version__.split(".")[0])

        if zarr_version >= 3:
            # zarr 3.x uses zarr.codecs
            try:
                from zarr.codecs import ZstdCodec
                return ZstdCodec(level=3), "compressors"
            except ImportError:
                try:
                    from zarr.codecs import BloscCodec
                    return BloscCodec(cname="lz4", clevel=5), "compressors"
                except ImportError:
                    return None, "compressors"
        else:
            # zarr 2.x uses numcodecs
            try:
                import numcodecs
                return numcodecs.Zstd(level=3), "compressor"
            except ImportError:
                try:
                    import numcodecs
                    return numcodecs.Blosc(cname="lz4", clevel=5), "compressor"
                except ImportError:
                    return None, "compressor"

    def write_day(
        self, ds_surf: xr.Dataset, ds_plev: xr.Dataset, date_str: str
    ) -> None:
        """Write one day of data to a per-day Zarr store.

        Each day gets its own store, making parallel writes safe.

        Args:
            ds_surf: Surface dataset.
            ds_plev: Pressure level dataset.
            date_str: Date string in YYYYMMDD format.
        """
        # Rename surface z to z_surf to avoid clash with pressure-level z
        if "z" in ds_surf:
            ds_surf = ds_surf.rename({"z": "z_surf"})

        # Merge surface and plev into one dataset
        ds = xr.merge([ds_surf, ds_plev])

        # Strip encoding from source datasets to avoid conflicts with zarr 3.x
        # Source zarr datasets may have numcodecs compressors in their encoding
        # which are incompatible with zarr 3.x's new codec API
        for var in ds.data_vars:
            ds[var].encoding.clear()
        for coord in ds.coords:
            ds[coord].encoding.clear()

        # Path for this day's store
        day_path = self.daily_dir / f"day_{date_str}.zarr"

        # Build encoding with compression
        compressor, comp_key = self._get_compressor()
        encoding = {}
        if compressor is not None:
            for var in ds.data_vars:
                encoding[var] = {comp_key: compressor}

        # Write complete store for this day (parallel-safe)
        ds.to_zarr(str(day_path), mode="w", encoding=encoding if encoding else None)
        logger.debug("Wrote %s", day_path.name)

        self._written_dates.append(date_str)

    def finalize(self) -> None:
        """Merge daily Zarr stores into single ERA5.zarr.

        If merge=False, skips merging (use open_mfdataset to read daily stores).
        """
        import zarr

        # Collect all daily stores
        daily_stores = sorted(self.daily_dir.glob("day_*.zarr"))

        if not daily_stores:
            logger.warning("No daily zarr stores found to merge")
            return

        if not self.merge:
            logger.info(
                "Skipping merge (merge=False). Use xr.open_mfdataset() to read %d daily stores.",
                len(daily_stores),
            )
            return

        logger.info("Merging %d daily stores → %s", len(daily_stores), self.zarr_path.name)

        # Include existing ERA5.zarr so incremental downloads don't lose data
        existing = self.zarr_path.exists()
        sources = []
        if existing:
            sources.append(str(self.zarr_path))
        sources.extend(str(p) for p in daily_stores)

        # Open all stores (existing + new) as single dataset.
        # combine="nested" + concat_dim="time" lets us handle duplicate
        # timestamps when a day is re-downloaded (combine="by_coords" fails
        # with "not monotonic" on any overlap between existing ERA5.zarr and
        # the new daily stores). We sort + dedupe below.
        ds = xr.open_mfdataset(
            sources,
            engine="zarr",
            combine="nested",
            concat_dim="time",
            parallel=True,
        )

        # Sort by time and drop duplicate timestamps (keep first occurrence,
        # which comes from the existing ERA5.zarr per ``sources`` ordering).
        ds = ds.sortby("time")
        _, unique_idx = np.unique(ds["time"].values, return_index=True)
        if len(unique_idx) < ds.sizes["time"]:
            n_dup = ds.sizes["time"] - len(unique_idx)
            logger.info("Dropping %d duplicate timestamps from merged store", n_dup)
            ds = ds.isel(time=np.sort(unique_idx))

        # Rechunk for merged store
        chunks = {k: v for k, v in self.chunks.items() if k in ds.dims}
        ds = ds.chunk(chunks)

        # Strip encoding from source datasets to avoid conflicts with zarr 3.x
        for var in ds.data_vars:
            ds[var].encoding.clear()
        for coord in ds.coords:
            ds[coord].encoding.clear()

        # Build encoding with compression
        compressor, comp_key = self._get_compressor()
        encoding = {}
        if compressor is not None:
            for var in ds.data_vars:
                encoding[var] = {comp_key: compressor}

        # Write to temp path to avoid reading and overwriting same store
        tmp_path = self.zarr_path.with_suffix(".zarr.tmp")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)

        ds.to_zarr(
            str(tmp_path),
            mode="w",
            encoding=encoding if encoding else None,
        )
        ds.close()

        # Atomic replace: remove old, rename temp → final
        if self.zarr_path.exists():
            shutil.rmtree(self.zarr_path)
        tmp_path.rename(self.zarr_path)

        # Consolidate metadata
        try:
            zarr.consolidate_metadata(str(self.zarr_path))
        except Exception as e:
            logger.warning("Failed to consolidate metadata: %s", e)

        logger.info("Zarr merge complete: %s", self.zarr_path)

        # Clean up daily stores after successful merge
        if self.cleanup_daily:
            logger.info("Cleaning up %d daily stores...", len(daily_stores))
            for p in daily_stores:
                shutil.rmtree(p)
            # Remove empty daily directory
            try:
                self.daily_dir.rmdir()
            except OSError:
                pass  # Directory not empty (other files present)
            logger.info("Daily stores removed")
