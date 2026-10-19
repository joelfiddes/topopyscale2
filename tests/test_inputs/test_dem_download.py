"""Tests for the sequential DEM tile path (public issue #1).

stitch_dem reads tiles remotely and merges them in thread pools, which hangs
forever on some managed Windows machines. For glo_30/glo_90 download_dem now
fetches tiles itself (sequential, with timeouts) and merges them from disk
without threads. These tests run offline on synthetic tiles.
"""

import socket
import threading

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from topopyscale2.inputs import dem_download as dd

RES = 1 / 120  # glo_90-like spacing (3 arcsec)


def _write_tile(path, west, north, n=120, seed=0):
    """A 1°x1° Copernicus-style tile: EPSG:4326, float32, pixel-is-point."""
    rng = np.random.default_rng(seed)
    data = (1000 + 2000 * rng.random((n, n))).astype(np.float32)
    profile = dict(
        driver="GTiff", width=n, height=n, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_origin(west, north, RES, RES),
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(AREA_OR_POINT="Point")
    return path


@pytest.fixture
def four_tiles(tmp_path):
    """Four tiles meeting at (10E, 47N)."""
    tiles = []
    for i, (w, n) in enumerate([(9, 47), (10, 47), (9, 48), (10, 48)]):
        tiles.append(_write_tile(tmp_path / f"tile_{w}_{n}.tif", w, n, seed=i))
    return tiles


@pytest.fixture
def no_new_threads(monkeypatch):
    def refuse(self):
        raise AssertionError("DEM path started a thread")
    monkeypatch.setattr(threading.Thread, "start", refuse)


class TestRouting:
    def test_copernicus_uses_local_tiles(self):
        assert dd._uses_local_tiles([9.7, 46.72, 9.98, 46.88], "glo_90")
        assert dd._uses_local_tiles([9.7, 46.72, 9.98, 46.88], "glo_30")

    def test_other_sources_keep_stitch_dem(self):
        assert not dd._uses_local_tiles([9.7, 46.72, 9.98, 46.88], "srtm_v3")
        assert not dd._uses_local_tiles([9.7, 46.72, 9.98, 46.88], "nasadem")

    def test_dateline_keeps_stitch_dem(self):
        assert not dd._uses_local_tiles([179.5, 60.0, 180.5, 60.5], "glo_90")

    def test_glo30_gap_keeps_stitch_dem(self):
        # Armenia: glo_30 tiles missing, stitch_dem patches them with glo_90
        assert not dd._uses_local_tiles([44.4, 40.1, 44.6, 40.3], "glo_30")

    def test_tile_urls_come_from_local_index(self, monkeypatch):
        import urllib.request

        def offline(*a, **k):
            raise AssertionError("tile index must not need the network")
        monkeypatch.setattr(urllib.request, "urlopen", offline)
        urls = dd.dem_tile_urls([9.7, 46.72, 9.98, 46.88], "glo_90")
        assert urls == [
            "https://copernicus-dem-90m.s3.amazonaws.com/"
            "Copernicus_DSM_COG_30_N46_00_E009_00_DEM/"
            "Copernicus_DSM_COG_30_N46_00_E009_00_DEM.tif"
        ]


class TestFetchTiles:
    URL = "https://example.invalid/a/Copernicus_DSM_COG_30_N46_00_E009_00_DEM.tif"

    def test_existing_tile_is_not_downloaded(self, tmp_path, monkeypatch):
        import urllib.request

        cached = tmp_path / "Copernicus_DSM_COG_30_N46_00_E009_00_DEM.tif"
        cached.write_bytes(b"placed by hand")
        monkeypatch.setattr(urllib.request, "urlopen",
                            lambda *a, **k: pytest.fail("downloaded a cached tile"))
        assert dd.fetch_dem_tiles([self.URL], tmp_path) == [cached]

    def test_stall_fails_with_actionable_error(self, tmp_path, monkeypatch):
        import time
        import urllib.request

        calls = []

        def stall(url, timeout):
            calls.append(timeout)
            raise socket.timeout("timed out")
        monkeypatch.setattr(urllib.request, "urlopen", stall)
        monkeypatch.setattr(time, "sleep", lambda s: None)

        with pytest.raises(dd.DEMDownloadError) as err:
            dd.fetch_dem_tiles([self.URL], tmp_path, timeout=5, retries=3)

        assert calls == [5, 5, 5]
        msg = str(err.value)
        assert self.URL in msg            # what to fetch
        assert str(tmp_path) in msg       # where to put it
        assert "domain.dem" in msg        # the other way out
        assert not list(tmp_path.glob("*.part"))

    def test_download_writes_tile(self, tmp_path, monkeypatch):
        import io
        import urllib.request

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False
        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: Resp(b"tile"))
        (path,) = dd.fetch_dem_tiles([self.URL], tmp_path)
        assert path.read_bytes() == b"tile"


class TestStitchLocalTiles:
    def test_matches_dem_stitcher_merge(self, four_tiles):
        """Same array and georeferencing as dem_stitcher's (threaded) merge."""
        from dem_stitcher.stitcher import merge_and_transform_dem_tiles

        bbox = [9.8, 46.8, 10.2, 47.2]
        new, new_prof = dd.stitch_local_tiles(four_tiles, bbox)

        datasets = [rasterio.open(p) for p in four_tiles]
        try:
            ref, ref_prof = merge_and_transform_dem_tiles(
                datasets, bbox, "glo_90",
                dst_ellipsoidal_height=False, dst_area_or_point="Point",
            )
        finally:
            for ds in datasets:
                ds.close()

        assert new.shape == ref[0].shape
        np.testing.assert_array_equal(new, ref[0])
        assert new_prof["transform"] == ref_prof["transform"]
        assert new_prof["crs"] == ref_prof["crs"]

    def test_starts_no_threads(self, four_tiles, no_new_threads):
        arr, _ = dd.stitch_local_tiles(four_tiles, [9.8, 46.8, 10.2, 47.2])
        assert np.isfinite(arr).all()

    def test_no_overlap_raises(self, four_tiles):
        with pytest.raises(dd.DEMDownloadError, match="overlaps"):
            dd.stitch_local_tiles(four_tiles, [20.0, 10.0, 20.5, 10.5])


def test_download_dem_from_hand_placed_tiles(tmp_path, monkeypatch, no_new_threads):
    """Tiles dropped in dem_cache/tiles/ give a DEM with no network and no threads."""
    import urllib.request

    tile_dir = tmp_path / "dem_cache" / "tiles"
    tile_dir.mkdir(parents=True)
    _write_tile(tile_dir / "Copernicus_DSM_COG_30_N46_00_E009_00_DEM.tif", 9, 47)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: pytest.fail("network used"))

    out = dd.download_dem(
        bbox=[9.7, 46.72, 9.98, 46.88],
        output_path=str(tmp_path / "dem_cache" / "dem.tif"),
        source="glo_90",
    )
    with rasterio.open(out) as src:
        b = src.bounds
        assert src.crs.to_epsg() == 4326
        assert b.left == pytest.approx(9.7, abs=RES) and b.right == pytest.approx(9.98, abs=RES)
        assert b.bottom == pytest.approx(46.72, abs=RES) and b.top == pytest.approx(46.88, abs=RES)
