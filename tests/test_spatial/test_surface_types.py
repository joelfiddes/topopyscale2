"""Tests for surface type classification."""

import numpy as np
import pytest
import xarray as xr

from topopyscale2.spatial.clusters import TopoSUBClustering
from topopyscale2.spatial.surface_types import (
    SURFACE_TYPE_CODES,
    SurfaceTypeClassifier,
)


class TestReclassify:
    def test_basic_reclassification(self):
        raw = xr.DataArray(
            np.array([[10, 20], [30, 40]]),
            dims=["y", "x"],
        )
        mapping = {10: "glacier", 20: "forest", 30: "open", 40: "rock"}

        classifier = SurfaceTypeClassifier()
        result = classifier.reclassify(raw, mapping)

        # Canonical codes (surface_types.SURFACE_TYPE_CODES). These used to be
        # asserted as open=3/rock=4, which was reclassify's own private map and
        # disagreed with the codes the clusterer names units from -- so a
        # reclassified raster produced units labelled "rock" for open ground.
        assert result.values[0, 0] == 1  # glacier
        assert result.values[0, 1] == 2  # forest
        assert result.values[1, 0] == 0  # open
        assert result.values[1, 1] == 3  # rock

    def test_unmapped_codes_remain_zero(self):
        raw = xr.DataArray(np.array([[10, 99]]), dims=["y", "x"])
        mapping = {10: "glacier"}

        classifier = SurfaceTypeClassifier()
        result = classifier.reclassify(raw, mapping)

        assert result.values[0, 0] == 1
        assert result.values[0, 1] == 0

    def test_no_source_returns_none(self):
        dem = xr.Dataset({"elevation": (["y", "x"], np.zeros((5, 5)))})
        classifier = SurfaceTypeClassifier()
        result = classifier.classify(dem)
        assert result is None


class TestCodesAgreeWithClusterer:
    """The clusterer names units from these codes, so they must not diverge."""

    def test_reclassify_codes_match_cluster_names(self):
        raw = xr.DataArray(
            np.array([[c for c in SURFACE_TYPE_CODES.values()]]), dims=["y", "x"]
        )
        mapping = {code: name for name, code in SURFACE_TYPE_CODES.items()}
        result = SurfaceTypeClassifier().reclassify(raw, mapping)

        for code, name in zip(result.values[0], SURFACE_TYPE_CODES):
            assert TopoSUBClustering.SURFACE_TYPE_NAMES[int(code)] == name


class TestGlacierSourceRGI:
    """glacier_source='rgi' -- the raster that makes stratification possible."""

    @staticmethod
    def _dem(ny=6, nx=8):
        return xr.Dataset(
            {"elevation": (["y", "x"], np.zeros((ny, nx)))},
            coords={"y": np.arange(ny) * 100.0, "x": np.arange(nx) * 100.0},
            attrs={"crs": "EPSG:32642"},
        )

    def test_marks_glacier_pixels(self, monkeypatch):
        dem = self._dem()
        mask = np.zeros((6, 8), dtype=bool)
        mask[2:4, 1:5] = True
        monkeypatch.setattr(
            "topopyscale2.inputs.rgi.glacier_mask", lambda *a, **k: mask
        )

        out = SurfaceTypeClassifier().classify(
            dem, glacier_source="rgi", rgi_regions=["13_central_asia"]
        )

        assert out is not None
        assert (out.values[mask] == SURFACE_TYPE_CODES["glacier"]).all()
        assert (out.values[~mask] == SURFACE_TYPE_CODES["open"]).all()
        assert out.shape == dem["elevation"].shape

    def test_rgi_overrides_glacier_class_of_a_custom_raster(self, monkeypatch, tmp_path):
        """RGI is a glacier inventory; a land-cover raster is not. RGI wins on ice,
        the raster keeps everything it knows that RGI cannot say."""
        import rasterio
        from rasterio.transform import from_origin

        dem = self._dem()
        base = np.full((6, 8), SURFACE_TYPE_CODES["forest"], dtype=np.int32)
        path = tmp_path / "surface.tif"
        with rasterio.open(
            path, "w", driver="GTiff", height=6, width=8, count=1,
            dtype="int32", crs="EPSG:32642",
            transform=from_origin(-50, 550, 100, 100),
        ) as dst:
            dst.write(base, 1)

        mask = np.zeros((6, 8), dtype=bool)
        mask[0, 0] = True
        monkeypatch.setattr(
            "topopyscale2.inputs.rgi.glacier_mask", lambda *a, **k: mask
        )

        out = SurfaceTypeClassifier().classify(
            dem, raster_path=path, glacier_source="rgi",
            rgi_regions=["13_central_asia"],
        )

        assert out.values[0, 0] == SURFACE_TYPE_CODES["glacier"]
        assert (out.values[~mask] == SURFACE_TYPE_CODES["forest"]).all()

    def test_missing_regions_is_an_error_not_a_silent_no_op(self):
        """Without regions nothing would be loaded and every pixel would stay
        non-glacier -- a silent wrong answer, so it must raise."""
        with pytest.raises(ValueError, match="rgi_regions"):
            SurfaceTypeClassifier().classify(self._dem(), glacier_source="rgi")

    def test_crs_is_read_from_the_elevation_variable(self, monkeypatch):
        """A config without an explicit crs keeps the DEM's own CRS on the
        elevation variable, not on the dataset. Reading only the dataset attr
        made glacier_source="rgi" fail on every such config.

        Proven by getting past the CRS check: fetch_region is replaced with a
        sentinel, so reaching it means a CRS was resolved.
        """
        from topopyscale2.inputs import rgi

        dem = xr.Dataset(
            {"elevation": (["y", "x"], np.zeros((4, 4)))},
            coords={"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0},
        )
        dem["elevation"].attrs["crs"] = "EPSG:32642"   # dataset attrs stay empty
        assert "crs" not in dem.attrs

        class GotPastTheCrsCheck(Exception):
            pass

        def sentinel(region, cache=None):
            raise GotPastTheCrsCheck

        monkeypatch.setattr(rgi, "fetch_region", sentinel)
        with pytest.raises(GotPastTheCrsCheck):
            rgi.glacier_mask(dem, ["13_central_asia"])

    def test_missing_crs_everywhere_is_a_clear_error(self):
        from topopyscale2.inputs import rgi

        dem = xr.Dataset(
            {"elevation": (["y", "x"], np.zeros((4, 4)))},
            coords={"y": np.arange(4) * 100.0, "x": np.arange(4) * 100.0},
        )
        with pytest.raises(ValueError, match="crs"):
            rgi.glacier_mask(dem, ["13_central_asia"])

    def test_none_source_still_returns_none(self):
        assert SurfaceTypeClassifier().classify(self._dem()) is None
