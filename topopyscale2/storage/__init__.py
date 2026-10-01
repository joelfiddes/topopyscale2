"""3-tier Zarr storage architecture for efficient I/O.

This module provides a layered storage architecture optimized for
meteorological forcing data:

- **ForcingStore protocol**: Abstract interface for storage backends
- **ZarrStore**: Zarr-based implementation with chunking and compression
- **NWPCache**: LRU cache for NWP data with rechunking support
- **Chunking utilities**: Optimal chunk computation for various data patterns
- **Format conversion**: NetCDF to Zarr conversion utilities
"""

from topopyscale2.storage.base import ForcingStore
from topopyscale2.storage.chunking import (
    chunks_for_nwp,
    compute_optimal_chunks,
)
from topopyscale2.storage.conversion import (
    netcdf_to_zarr,
    zarr_to_netcdf,
)

# Zarr-dependent classes are imported conditionally
try:
    from topopyscale2.storage.cache import CacheEntry, NWPCache
    from topopyscale2.storage.zarr_store import ZarrStore
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    ZarrStore = None  # type: ignore
    NWPCache = None  # type: ignore
    CacheEntry = None  # type: ignore


__all__ = [
    "ForcingStore",
    "ZarrStore",
    "NWPCache",
    "CacheEntry",
    "compute_optimal_chunks",
    "chunks_for_nwp",
    "netcdf_to_zarr",
    "zarr_to_netcdf",
    "HAS_ZARR",
]
