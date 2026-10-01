"""NWP data cache with LRU eviction and rechunking support."""

import datetime
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import xarray as xr

try:
    import zarr
    ZARR_VERSION = int(zarr.__version__.split(".")[0])
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    ZARR_VERSION = 0

from topopyscale2.storage.chunking import chunks_for_nwp, compressor_encoding, resolve_chunks


@dataclass
class CacheEntry:
    """Metadata for a cached NWP dataset.

    Attributes
    ----------
    source_name : str
        NWP source identifier (e.g., "era5", "hres").
    store_path : Path
        Path to Zarr store.
    time_start : datetime.datetime
        Start of time range.
    time_end : datetime.datetime
        End of time range.
    bbox : tuple
        Bounding box (west, south, east, north).
    size_bytes : int
        Store size in bytes.
    created : datetime.datetime
        When entry was created.
    last_accessed : datetime.datetime
        When entry was last accessed.
    variables : list
        List of variable names.
    """

    source_name: str
    store_path: Path
    time_start: datetime.datetime
    time_end: datetime.datetime
    bbox: tuple[float, float, float, float]
    size_bytes: int
    created: datetime.datetime
    last_accessed: datetime.datetime
    variables: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_name": self.source_name,
            "store_path": str(self.store_path),
            "time_start": self.time_start.isoformat(),
            "time_end": self.time_end.isoformat(),
            "bbox": self.bbox,
            "size_bytes": self.size_bytes,
            "created": self.created.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "variables": self.variables,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CacheEntry":
        """Create from dictionary."""
        return cls(
            source_name=d["source_name"],
            store_path=Path(d["store_path"]),
            time_start=datetime.datetime.fromisoformat(d["time_start"]),
            time_end=datetime.datetime.fromisoformat(d["time_end"]),
            bbox=tuple(d["bbox"]),
            size_bytes=d["size_bytes"],
            created=datetime.datetime.fromisoformat(d["created"]),
            last_accessed=datetime.datetime.fromisoformat(d["last_accessed"]),
            variables=d.get("variables", []),
        )


class NWPCache:
    """LRU cache for NWP data with rechunking support.

    Stores NWP data in Zarr format with automatic eviction when
    cache exceeds size limit.

    Parameters
    ----------
    cache_dir : str or Path
        Directory for cache storage.
    max_gb : float, optional
        Maximum cache size in gigabytes. None = no limit (default).
    default_chunks : dict, optional
        Default chunk sizes. If None, uses source-specific presets.

    Examples
    --------
    >>> cache = NWPCache("/data/cache", max_gb=100)
    >>> cache.ingest(era5_ds, "era5")
    >>> subset = cache.get("era5", time_range=("2020-01-01", "2020-01-31"),
    ...                    bbox=(10, 45, 12, 47))
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        max_gb: Optional[float] = None,
        default_chunks: Optional[dict[str, int]] = None,
    ):
        if not HAS_ZARR:
            raise ImportError(
                "zarr is required for NWPCache. "
                "Install with: pip install zarr"
            )

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # None = no limit (use infinity)
        self.max_bytes = max_gb * 1024 * 1024 * 1024 if max_gb else float("inf")
        self.default_chunks = default_chunks

        # Entries indexed by source_name
        self._entries: dict[str, list[CacheEntry]] = {}

        # Load existing cache index
        self._index_path = self.cache_dir / ".cache_index.json"
        self._load_index()

    def _load_index(self):
        """Load cache index from disk."""
        if self._index_path.exists():
            try:
                with open(self._index_path) as f:
                    data = json.load(f)
                for source_name, entries in data.items():
                    self._entries[source_name] = [
                        CacheEntry.from_dict(e) for e in entries
                    ]
            except (json.JSONDecodeError, KeyError) as e:
                import warnings
                warnings.warn(f"Failed to load cache index: {e}")
                self._entries = {}

    def _save_index(self):
        """Save cache index to disk."""
        data = {}
        for source_name, entries in self._entries.items():
            data[source_name] = [e.to_dict() for e in entries]
        with open(self._index_path, "w") as f:
            json.dump(data, f, indent=2)

    def _compute_store_size(self, path: Path) -> int:
        """Compute total size of a Zarr store in bytes."""
        total = 0
        for item in path.rglob("*"):
            if item.is_file():
                total += item.stat().st_size
        return total

    def _generate_store_name(self, source_name: str, ds: xr.Dataset) -> str:
        """Generate unique store name based on source and time range."""
        # Get time bounds
        if "time" in ds.dims:
            t_start = str(ds.time.values[0])[:10].replace("-", "")
            t_end = str(ds.time.values[-1])[:10].replace("-", "")
            return f"{source_name}_{t_start}_{t_end}.zarr"
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            return f"{source_name}_{timestamp}.zarr"

    def _get_compressor(self, compression: str = "zstd", level: int = 3):
        """Get compressor based on zarr version."""
        # Zarr 3.x uses zarr.codecs
        if ZARR_VERSION >= 3:
            try:
                from zarr.codecs import GzipCodec, ZstdCodec
                if compression == "zstd":
                    return ZstdCodec(level=level)
                elif compression == "gzip":
                    return GzipCodec(level=level)
                return None
            except ImportError:
                return None

        # Zarr 2.x uses numcodecs
        try:
            import numcodecs
            if compression == "zstd":
                return numcodecs.Zstd(level=level)
            elif compression == "lz4":
                return numcodecs.LZ4(acceleration=level)
            elif compression == "gzip":
                return numcodecs.GZip(level=level)
            return None
        except ImportError:
            return None

    @property
    def total_size_bytes(self) -> int:
        """Total size of all cached data in bytes."""
        return sum(
            entry.size_bytes
            for entries in self._entries.values()
            for entry in entries
        )

    @property
    def total_size_gb(self) -> float:
        """Total size of all cached data in gigabytes."""
        return self.total_size_bytes / (1024 * 1024 * 1024)

    def ingest(
        self,
        nwp_dataset: xr.Dataset,
        source_name: str,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> CacheEntry:
        """Add NWP data to cache.

        Parameters
        ----------
        nwp_dataset : xr.Dataset
            NWP dataset to cache.
        source_name : str
            NWP source identifier (e.g., "era5", "hres").
        compression : str
            Compression algorithm. Default: "zstd".
        compression_level : int
            Compression level. Default: 3.

        Returns
        -------
        CacheEntry
            Metadata for the cached dataset.
        """
        # Get chunks for this source type
        chunks = self.default_chunks or chunks_for_nwp(source_name)

        # Resolve chunks against actual dimensions
        shape_dict = dict(nwp_dataset.sizes)
        resolved_chunks = resolve_chunks(chunks, shape_dict)

        # Chunk the dataset
        ds = nwp_dataset.chunk(resolved_chunks)

        # Generate store path
        store_name = self._generate_store_name(source_name, ds)
        store_path = self.cache_dir / source_name / store_name
        store_path.parent.mkdir(parents=True, exist_ok=True)

        # Build encoding
        compressor = self._get_compressor(compression, compression_level)
        encoding = {}
        if compressor is not None:
            for var in ds.data_vars:
                encoding[var] = compressor_encoding(compressor)

        # Write to Zarr
        ds.to_zarr(
            store_path,
            mode="w",
            encoding=encoding if encoding else None,
            consolidated=True,
        )

        # Compute size
        size_bytes = self._compute_store_size(store_path)

        # Extract time bounds
        if "time" in ds.dims:
            time_start = np.datetime64(ds.time.values[0], "ns")
            time_end = np.datetime64(ds.time.values[-1], "ns")
            # Convert to Python datetime
            time_start = datetime.datetime.utcfromtimestamp(
                time_start.astype("datetime64[s]").astype(int)
            )
            time_end = datetime.datetime.utcfromtimestamp(
                time_end.astype("datetime64[s]").astype(int)
            )
        else:
            time_start = time_end = datetime.datetime.utcnow()

        # Extract bbox
        lat_dim = "latitude" if "latitude" in ds.dims else "lat"
        lon_dim = "longitude" if "longitude" in ds.dims else "lon"

        if lat_dim in ds.coords and lon_dim in ds.coords:
            bbox = (
                float(ds[lon_dim].values.min()),
                float(ds[lat_dim].values.min()),
                float(ds[lon_dim].values.max()),
                float(ds[lat_dim].values.max()),
            )
        else:
            bbox = (-180.0, -90.0, 180.0, 90.0)

        # Create entry
        now = datetime.datetime.utcnow()
        entry = CacheEntry(
            source_name=source_name,
            store_path=store_path,
            time_start=time_start,
            time_end=time_end,
            bbox=bbox,
            size_bytes=size_bytes,
            created=now,
            last_accessed=now,
            variables=list(ds.data_vars),
        )

        # Add to index
        if source_name not in self._entries:
            self._entries[source_name] = []
        self._entries[source_name].append(entry)

        # Save index
        self._save_index()

        # Check if eviction needed
        if self.total_size_bytes > self.max_bytes:
            self.evict(self.max_bytes / (1024 * 1024 * 1024))

        return entry

    def get(
        self,
        source_name: str,
        time_range: Optional[tuple[str, str]] = None,
        bbox: Optional[tuple[float, float, float, float]] = None,
    ) -> Optional[xr.Dataset]:
        """Retrieve cached data.

        Parameters
        ----------
        source_name : str
            NWP source identifier.
        time_range : tuple, optional
            Time range as (start, end) ISO strings.
        bbox : tuple, optional
            Bounding box (west, south, east, north).

        Returns
        -------
        xr.Dataset or None
            Cached data if found, None otherwise.
        """
        if source_name not in self._entries:
            return None

        # Find matching entries
        matching = []
        for entry in self._entries[source_name]:
            if not entry.store_path.exists():
                continue

            # Check time overlap
            if time_range is not None:
                req_start = datetime.datetime.fromisoformat(time_range[0])
                req_end = datetime.datetime.fromisoformat(time_range[1])

                # Must have some overlap
                if entry.time_end < req_start or entry.time_start > req_end:
                    continue

            # Check bbox overlap
            if bbox is not None:
                west, south, east, north = bbox
                e_west, e_south, e_east, e_north = entry.bbox

                # Must have spatial overlap
                if e_east < west or e_west > east:
                    continue
                if e_north < south or e_south > north:
                    continue

            matching.append(entry)

        if not matching:
            return None

        # Use most recent entry
        matching.sort(key=lambda e: e.last_accessed, reverse=True)
        entry = matching[0]

        # Update access time
        entry.last_accessed = datetime.datetime.utcnow()
        self._save_index()

        # Load dataset
        ds = xr.open_zarr(entry.store_path, consolidated=True)

        # Subset if requested
        if time_range is not None:
            ds = ds.sel(time=slice(time_range[0], time_range[1]))

        if bbox is not None:
            lat_dim = "latitude" if "latitude" in ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds.dims else "lon"

            west, south, east, north = bbox
            if lat_dim in ds.coords:
                ds = ds.sel(**{lat_dim: slice(south, north)})
            if lon_dim in ds.coords:
                ds = ds.sel(**{lon_dim: slice(west, east)})

        return ds

    def rechunk(
        self,
        source_name: str,
        target_chunks: dict[str, int],
    ) -> list[CacheEntry]:
        """Rechunk all entries for a source to new chunk sizes.

        Parameters
        ----------
        source_name : str
            NWP source identifier.
        target_chunks : dict
            New chunk sizes.

        Returns
        -------
        list[CacheEntry]
            Updated cache entries.
        """
        if source_name not in self._entries:
            return []

        updated = []
        for entry in self._entries[source_name]:
            if not entry.store_path.exists():
                continue

            # Load and rechunk
            ds = xr.open_zarr(entry.store_path, consolidated=True)

            # Resolve chunks
            shape_dict = dict(ds.sizes)
            resolved = resolve_chunks(target_chunks, shape_dict)
            ds = ds.chunk(resolved)

            # Write to temp location
            temp_path = entry.store_path.with_suffix(".zarr.tmp")
            compressor = self._get_compressor()
            encoding = {}
            if compressor is not None:
                for var in ds.data_vars:
                    encoding[var] = compressor_encoding(compressor)

            ds.to_zarr(
                temp_path,
                mode="w",
                encoding=encoding if encoding else None,
                consolidated=True,
            )
            ds.close()

            # Replace original
            shutil.rmtree(entry.store_path)
            temp_path.rename(entry.store_path)

            # Update size
            entry.size_bytes = self._compute_store_size(entry.store_path)
            updated.append(entry)

        self._save_index()
        return updated

    def evict(self, max_gb: float) -> list[CacheEntry]:
        """Evict entries using LRU until cache is under limit.

        Parameters
        ----------
        max_gb : float
            Maximum cache size in gigabytes.

        Returns
        -------
        list[CacheEntry]
            Evicted entries.
        """
        max_bytes = max_gb * 1024 * 1024 * 1024
        evicted = []

        # Get all entries sorted by last access (oldest first)
        all_entries = []
        for source_name, entries in self._entries.items():
            for entry in entries:
                all_entries.append((source_name, entry))

        all_entries.sort(key=lambda x: x[1].last_accessed)

        # Evict until under limit
        while self.total_size_bytes > max_bytes and all_entries:
            source_name, entry = all_entries.pop(0)

            # Remove from disk
            if entry.store_path.exists():
                shutil.rmtree(entry.store_path)

            # Remove from index
            self._entries[source_name].remove(entry)
            evicted.append(entry)

        # Clean up empty source lists
        self._entries = {
            k: v for k, v in self._entries.items() if v
        }

        self._save_index()
        return evicted

    def list_entries(
        self,
        source_name: Optional[str] = None,
    ) -> list[CacheEntry]:
        """List cache entries.

        Parameters
        ----------
        source_name : str, optional
            Filter by source name.

        Returns
        -------
        list[CacheEntry]
            Cache entries.
        """
        if source_name:
            return self._entries.get(source_name, [])

        return [
            entry
            for entries in self._entries.values()
            for entry in entries
        ]

    def clear(self, source_name: Optional[str] = None):
        """Clear cache entries.

        Parameters
        ----------
        source_name : str, optional
            If provided, only clear entries for this source.
        """
        if source_name:
            # Clear specific source
            if source_name in self._entries:
                for entry in self._entries[source_name]:
                    if entry.store_path.exists():
                        shutil.rmtree(entry.store_path)
                del self._entries[source_name]

                # Remove source directory if empty
                source_dir = self.cache_dir / source_name
                if source_dir.exists() and not any(source_dir.iterdir()):
                    source_dir.rmdir()
        else:
            # Clear all
            for entries in self._entries.values():
                for entry in entries:
                    if entry.store_path.exists():
                        shutil.rmtree(entry.store_path)
            self._entries = {}

        self._save_index()

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns
        -------
        dict
            Cache statistics.
        """
        entries = self.list_entries()
        return {
            "total_entries": len(entries),
            "total_size_gb": self.total_size_gb,
            "max_size_gb": self.max_bytes / (1024 * 1024 * 1024),
            "utilization": self.total_size_bytes / self.max_bytes if self.max_bytes > 0 else 0,
            "sources": list(self._entries.keys()),
            "entries_by_source": {
                k: len(v) for k, v in self._entries.items()
            },
        }
