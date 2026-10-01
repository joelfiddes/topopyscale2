"""Abstract base class for ERA5 data backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
import xarray as xr

from topopyscale2.inputs.nwp_downloader.bbox import BBox


class ERA5Backend(ABC):
    """Abstract base class for ERA5 data backends.

    Each backend must implement ``fetch_day()`` which returns a pair of
    xarray Datasets (surface, pressure-level) with standardised variable
    names and dimension names.

    Standard surface variables: z, d2m, strd, ssrd, sp, tp, t2m
    Standard pressure variables: z, t, u, v, q, r
    Standard dimensions: time, latitude, longitude, level (hPa ascending)
    """

    def __init__(
        self,
        bbox: BBox,
        pressure_levels: list[int],
        time_resolution: str = "1H",
        **kwargs,
    ):
        self.bbox = bbox
        self.pressure_levels = sorted(pressure_levels)
        self.time_resolution = time_resolution

    @abstractmethod
    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        """Fetch one day of ERA5 data.

        Args:
            date: The date to fetch.

        Returns:
            Tuple of (ds_surf, ds_plev) with standardised names and dims.
        """
        ...

    def probe_date(self, date: pd.Timestamp) -> bool:
        """Check if data exists for a given date.

        Override in subclasses that support availability probing (e.g. Google).
        Default returns True (assume available).
        """
        return True

    def close(self):
        """Clean up resources. Override in subclasses if needed."""
        pass
