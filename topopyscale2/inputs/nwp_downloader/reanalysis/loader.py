"""ERA5Loader — orchestrator wiring backend + writer + derived variables."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import pandas as pd
from tqdm import tqdm

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.derived import compute_relative_humidity
from topopyscale2.inputs.nwp_downloader.validation import get_existing_dates
from topopyscale2.inputs.nwp_downloader.writer import BaseWriter, NetCDFWriter, ZarrWriter

logger = logging.getLogger(__name__)


class ERA5Loader:
    """Orchestrates ERA5 downloads: backend → (optional RH) → writer.

    Pipeline:
        1. Build date list from start_date to end_date.
        2. Filter out already-downloaded dates (resume support).
        3. ThreadPoolExecutor: fetch_day → compute RH if needed → write_day.
        4. Finalize (merge yearly netCDFs or consolidate zarr).

    Args:
        backend: Backend name ('google', 'cds', 's3zarr') or an ERA5Backend instance.
        bbox: Bounding box as (W, S, E, N) tuple or BBox instance.
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        pressure_levels: Pressure levels in hPa.
        output_format: 'netcdf' or 'zarr'.
        output_dir: Directory for output files.
        time_resolution: '1H', '2H', '3H', or '6H'.
        max_workers: Number of parallel download threads.
        compute_rh: If True, compute relative humidity from q if missing.
        skip_existing: If True, skip dates already downloaded.
        backend_kwargs: Extra keyword arguments passed to the backend constructor.
        writer_kwargs: Extra keyword arguments passed to the writer constructor.
            For ZarrWriter: merge (bool), cleanup_daily (bool), chunks (dict).

    Example::

        loader = ERA5Loader(
            backend="google",
            bbox=(6.5, 60.0, 8.5, 61.5),
            start_date="2020-01-01",
            end_date="2020-12-31",
            pressure_levels=[300, 500, 700, 850, 1000],
            output_format="netcdf",
            output_dir="./inputs/climate/",
        )
        loader.download()
    """

    def __init__(
        self,
        backend: Union[str, ERA5Backend],
        bbox: Union[tuple, BBox],
        start_date: str,
        end_date: str,
        pressure_levels: list[int] = None,
        output_format: str = "netcdf",
        output_dir: str = "./inputs/climate/",
        time_resolution: str = "1H",
        max_workers: int = 4,
        compute_rh: bool = True,
        skip_existing: bool = True,
        backend_kwargs: dict = None,
        writer_kwargs: dict = None,
    ):
        # Resolve bbox
        if isinstance(bbox, (tuple, list)):
            self.bbox = BBox.from_tuple(tuple(bbox))
        else:
            self.bbox = bbox

        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.pressure_levels = pressure_levels or [300, 500, 700, 850, 1000]
        self.output_format = output_format
        self.output_dir = output_dir
        self.time_resolution = time_resolution
        self.max_workers = max_workers
        self.compute_rh = compute_rh
        self.skip_existing = skip_existing

        # Resolve backend
        if isinstance(backend, str):
            if backend == "auto":
                from topopyscale2.inputs.nwp_downloader.reanalysis.backends import select_backend

                backend, auto_kwargs = select_backend(
                    start_date=self.start_date,
                    end_date=self.end_date,
                    pressure_levels=self.pressure_levels,
                )
                backend_kwargs = {**auto_kwargs, **(backend_kwargs or {})}

            from topopyscale2.inputs.nwp_downloader.reanalysis.backends import get_backend

            self._backend = get_backend(
                backend,
                bbox=self.bbox,
                pressure_levels=self.pressure_levels,
                time_resolution=self.time_resolution,
                **(backend_kwargs or {}),
            )
        else:
            self._backend = backend

        self.effective_end_date: pd.Timestamp | None = None

        # Resolve writer
        writer_kwargs = writer_kwargs or {}
        if output_format == "netcdf":
            self._writer: BaseWriter = NetCDFWriter(output_dir, **writer_kwargs)
        elif output_format == "zarr":
            self._writer = ZarrWriter(output_dir, **writer_kwargs)
        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def _detect_end_date(self) -> pd.Timestamp:
        """Walk backward from end_date to find last available ERA5 date.

        Only probes when end_date is recent (within last 10 days) since
        ERA5 has a ~5-day lag from present. Returns end_date unchanged
        for historical periods.
        """
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=10)
        if self.end_date < cutoff:
            return self.end_date  # Not recent, skip probing

        logger.info("Probing ERA5 availability (end_date is recent)...")
        date = self.end_date.normalize()
        while date >= self.start_date:
            if self._backend.probe_date(date):
                if date < self.end_date.normalize():
                    logger.warning(
                        "ERA5 available through %s (requested %s, ~%d day lag)",
                        date.strftime("%Y-%m-%d"),
                        self.end_date.strftime("%Y-%m-%d"),
                        (self.end_date.normalize() - date).days,
                    )
                return date
            date -= pd.Timedelta(days=1)

        raise FileNotFoundError(
            f"No ERA5 data found between {self.start_date:%Y-%m-%d} "
            f"and {self.end_date:%Y-%m-%d}"
        )

    def _build_date_list(self) -> list[pd.Timestamp]:
        """Build list of dates to download."""
        return list(pd.date_range(self.start_date, self.end_date, freq="D"))

    def _filter_existing(
        self, dates: list[pd.Timestamp]
    ) -> list[pd.Timestamp]:
        """Remove dates that are already downloaded."""
        if not self.skip_existing:
            return dates

        existing = get_existing_dates(self.output_dir, self.output_format)
        filtered = [d for d in dates if d.normalize() not in existing]

        n_skip = len(dates) - len(filtered)
        if n_skip > 0:
            logger.info("Skipping %d already-downloaded dates", n_skip)

        return filtered

    def _process_day(self, date: pd.Timestamp) -> str:
        """Fetch, compute derived vars, and write one day."""
        ds_surf, ds_plev = self._backend.fetch_day(date)

        # Compute RH if requested and missing from plev data
        if self.compute_rh and "r" not in ds_plev and "q" in ds_plev and "t" in ds_plev:
            logger.debug("Computing relative humidity for %s", date.strftime("%Y-%m-%d"))
            # Build pressure array matching plev dims

            pressure_pa = ds_plev["level"].astype("float32") * 100.0  # hPa → Pa
            rh = compute_relative_humidity(
                temperature=ds_plev["t"],
                pressure=pressure_pa,
                specific_humidity=ds_plev["q"],
            )
            ds_plev["r"] = rh

        date_str = date.strftime("%Y%m%d")
        self._writer.write_day(ds_surf, ds_plev, date_str)
        return date_str

    def download(self) -> None:
        """Run the full download pipeline."""
        # Detect actual ERA5 availability for recent end dates
        detected_end = self._detect_end_date()
        self.effective_end_date = detected_end

        # Temporarily narrow end_date for date list construction
        original_end = self.end_date
        self.end_date = detected_end

        dates = self._build_date_list()
        dates = self._filter_existing(dates)

        # Restore original end_date
        self.end_date = original_end

        if not dates:
            logger.info("Nothing to download — all dates already present")
            self._writer.finalize()
            return

        logger.info(
            "Downloading %d days (%s to %s) with backend=%s",
            len(dates),
            dates[0].strftime("%Y-%m-%d"),
            dates[-1].strftime("%Y-%m-%d"),
            type(self._backend).__name__,
        )

        unavailable: list[pd.Timestamp] = []

        if self.max_workers <= 1:
            # Sequential download
            for date in tqdm(dates, desc="Downloading ERA5"):
                try:
                    self._process_day(date)
                except FileNotFoundError:
                    logger.warning(
                        "ERA5 data not available for %s — skipping",
                        date.strftime("%Y-%m-%d"),
                    )
                    unavailable.append(date)
        else:
            # Parallel download
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._process_day, d): d for d in dates
                }
                for future in tqdm(
                    as_completed(futures), total=len(dates), desc="Downloading ERA5"
                ):
                    date = futures[future]
                    try:
                        future.result()
                    except FileNotFoundError:
                        logger.warning(
                            "ERA5 data not available for %s — skipping",
                            date.strftime("%Y-%m-%d"),
                        )
                        unavailable.append(date)
                    except Exception:
                        logger.error(
                            "Failed to download %s",
                            date.strftime("%Y-%m-%d"),
                            exc_info=True,
                        )
                        raise

        n_ok = len(dates) - len(unavailable)
        if unavailable:
            logger.warning(
                "Downloaded %d/%d days (%d unavailable)",
                n_ok, len(dates), len(unavailable),
            )
        else:
            logger.info("Downloaded all %d days successfully", n_ok)

        logger.info("Download complete, finalizing output")
        self._writer.finalize()

        try:
            self._backend.close()
        except Exception:
            pass

        logger.info("Done")
