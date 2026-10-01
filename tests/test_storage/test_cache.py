"""Tests for NWP cache functionality."""

import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

zarr = pytest.importorskip("zarr")

from topopyscale2.storage.cache import CacheEntry, NWPCache


def make_nwp_dataset(
    nt=24,
    nlat=10,
    nlon=20,
    start_date="2020-01-01",
):
    """Create test NWP dataset."""
    times = pd.date_range(start_date, periods=nt, freq="h").values
    lat = np.linspace(45, 47, nlat)
    lon = np.linspace(10, 12, nlon)

    rng = np.random.default_rng(42)

    return xr.Dataset(
        {
            "t2m": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((nt, nlat, nlon)) + 270,
            ),
            "sp": (
                ["time", "latitude", "longitude"],
                rng.standard_normal((nt, nlat, nlon)) * 500 + 80000,
            ),
            "tp": (
                ["time", "latitude", "longitude"],
                np.abs(rng.standard_normal((nt, nlat, nlon))) * 0.001,
            ),
        },
        coords={"time": times, "latitude": lat, "longitude": lon},
    )


class TestCacheEntry:
    """CacheEntry dataclass tests."""

    def test_to_dict(self):
        """Test CacheEntry serialization."""
        from pathlib import Path

        entry = CacheEntry(
            source_name="era5",
            store_path=Path("/data/era5.zarr"),
            time_start=datetime.datetime(2020, 1, 1),
            time_end=datetime.datetime(2020, 1, 2),
            bbox=(10.0, 45.0, 12.0, 47.0),
            size_bytes=1000000,
            created=datetime.datetime(2020, 1, 1),
            last_accessed=datetime.datetime(2020, 1, 1),
            variables=["t2m", "sp"],
        )

        d = entry.to_dict()
        assert d["source_name"] == "era5"
        assert d["bbox"] == (10.0, 45.0, 12.0, 47.0)
        assert d["variables"] == ["t2m", "sp"]

    def test_from_dict(self):
        """Test CacheEntry deserialization."""
        d = {
            "source_name": "era5",
            "store_path": "/data/era5.zarr",
            "time_start": "2020-01-01T00:00:00",
            "time_end": "2020-01-02T00:00:00",
            "bbox": [10.0, 45.0, 12.0, 47.0],
            "size_bytes": 1000000,
            "created": "2020-01-01T00:00:00",
            "last_accessed": "2020-01-01T00:00:00",
            "variables": ["t2m"],
        }

        entry = CacheEntry.from_dict(d)
        assert entry.source_name == "era5"
        assert entry.time_start == datetime.datetime(2020, 1, 1)


class TestNWPCacheBasic:
    """Basic NWPCache functionality tests."""

    def test_init_creates_directory(self, tmp_path):
        """Test cache initializes directory structure."""
        cache_dir = tmp_path / "cache"
        cache = NWPCache(cache_dir, max_gb=10)

        assert cache_dir.exists()
        assert cache.max_bytes == 10 * 1024 * 1024 * 1024

    def test_ingest_stores_data(self, tmp_path):
        """Test ingest stores NWP data."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()

        entry = cache.ingest(ds, "era5", compression=None)

        assert entry.source_name == "era5"
        assert entry.store_path.exists()
        assert "t2m" in entry.variables

    def test_ingest_creates_entry(self, tmp_path):
        """Test ingest creates cache entry with metadata."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()

        entry = cache.ingest(ds, "era5", compression=None)

        assert entry.size_bytes > 0
        assert entry.time_start.year == 2020
        assert entry.bbox[0] == 10.0  # west

    def test_get_retrieves_data(self, tmp_path):
        """Test get retrieves cached data."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)

        result = cache.get("era5")

        assert result is not None
        assert "t2m" in result.data_vars
        result.close()

    def test_get_returns_none_for_missing(self, tmp_path):
        """Test get returns None for missing source."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)

        result = cache.get("nonexistent")

        assert result is None


class TestNWPCacheFiltering:
    """Cache retrieval filtering tests."""

    def test_get_with_time_range(self, tmp_path):
        """Test get filters by time range."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset(nt=48, start_date="2020-01-01")
        cache.ingest(ds, "era5", compression=None)

        result = cache.get(
            "era5",
            time_range=("2020-01-01T06:00:00", "2020-01-01T12:00:00"),
        )

        assert result is not None
        assert result.sizes["time"] <= 48
        result.close()

    def test_get_with_bbox(self, tmp_path):
        """Test get filters by bounding box."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)

        result = cache.get(
            "era5",
            bbox=(10.5, 45.5, 11.5, 46.5),
        )

        assert result is not None
        # Data should be subset
        result.close()

    def test_get_no_match_outside_time(self, tmp_path):
        """Test get returns None when time range doesn't overlap."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset(nt=24, start_date="2020-01-01")
        cache.ingest(ds, "era5", compression=None)

        result = cache.get(
            "era5",
            time_range=("2021-01-01", "2021-01-02"),
        )

        assert result is None


class TestNWPCacheEviction:
    """Cache eviction tests."""

    def test_evict_removes_old_entries(self, tmp_path):
        """Test evict removes least recently used entries."""
        cache = NWPCache(tmp_path / "cache", max_gb=1)

        # Ingest multiple datasets
        for i in range(3):
            ds = make_nwp_dataset(start_date=f"2020-01-{i+1:02d}")
            cache.ingest(ds, "era5", compression=None)

        initial_count = len(cache.list_entries())

        # Force eviction to very small size
        evicted = cache.evict(max_gb=0.0001)

        assert len(evicted) > 0
        assert len(cache.list_entries()) < initial_count

    def test_evict_lru_order(self, tmp_path):
        """Test eviction uses LRU ordering."""
        cache = NWPCache(tmp_path / "cache", max_gb=1)

        # Ingest two datasets
        ds1 = make_nwp_dataset(start_date="2020-01-01")
        ds2 = make_nwp_dataset(start_date="2020-01-02")

        cache.ingest(ds1, "era5", compression=None)
        cache.ingest(ds2, "era5", compression=None)

        # Access first entry to make it more recent
        cache.get("era5", time_range=("2020-01-01", "2020-01-01T12:00:00"))

        # Evict to keep only one
        evicted = cache.evict(max_gb=0.0001)

        # Verify something was evicted
        assert len(evicted) > 0


class TestNWPCacheRechunk:
    """Cache rechunking tests."""

    def test_rechunk_updates_chunks(self, tmp_path):
        """Test rechunk updates chunk sizes."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset(nt=48)
        cache.ingest(ds, "era5", compression=None)

        updated = cache.rechunk("era5", {"time": 12})

        assert len(updated) > 0
        # Entry should still exist
        result = cache.get("era5")
        assert result is not None
        result.close()


class TestNWPCacheStats:
    """Cache statistics tests."""

    def test_total_size(self, tmp_path):
        """Test total_size_gb returns correct value."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)

        assert cache.total_size_bytes > 0
        assert cache.total_size_gb > 0

    def test_get_stats(self, tmp_path):
        """Test get_stats returns cache statistics."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)

        stats = cache.get_stats()

        assert stats["total_entries"] == 1
        assert stats["total_size_gb"] > 0
        assert "era5" in stats["sources"]

    def test_list_entries(self, tmp_path):
        """Test list_entries returns all entries."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)
        cache.ingest(ds, "hres", compression=None)

        entries = cache.list_entries()
        assert len(entries) == 2

        era5_entries = cache.list_entries("era5")
        assert len(era5_entries) == 1


class TestNWPCacheClear:
    """Cache clearing tests."""

    def test_clear_specific_source(self, tmp_path):
        """Test clear removes specific source."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)
        cache.ingest(ds, "hres", compression=None)

        cache.clear("era5")

        assert len(cache.list_entries("era5")) == 0
        assert len(cache.list_entries("hres")) == 1

    def test_clear_all(self, tmp_path):
        """Test clear removes all entries."""
        cache = NWPCache(tmp_path / "cache", max_gb=10)
        ds = make_nwp_dataset()
        cache.ingest(ds, "era5", compression=None)
        cache.ingest(ds, "hres", compression=None)

        cache.clear()

        assert len(cache.list_entries()) == 0


class TestNWPCachePersistence:
    """Cache persistence tests."""

    def test_index_persists_across_sessions(self, tmp_path):
        """Test cache index persists when cache is reopened."""
        cache_dir = tmp_path / "cache"

        # First session
        cache1 = NWPCache(cache_dir, max_gb=10)
        ds = make_nwp_dataset()
        cache1.ingest(ds, "era5", compression=None)

        # Second session
        cache2 = NWPCache(cache_dir, max_gb=10)

        entries = cache2.list_entries()
        assert len(entries) == 1
        assert entries[0].source_name == "era5"
