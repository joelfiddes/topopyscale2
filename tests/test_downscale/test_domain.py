"""Tests for Domain class."""

from pathlib import Path

import numpy as np
import xarray as xr

from topopyscale2.config.schema import DomainConfig, TPS2Config
from topopyscale2.domain import Domain


def _make_dem(ny=20, nx=20):
    """Synthetic DEM in geographic coords (lon/lat) with a crs attr."""
    x = np.linspace(10.0, 11.0, nx)
    y = np.linspace(46.0, 47.0, ny)
    ds = xr.Dataset(
        {"elevation": (["y", "x"], np.full((ny, nx), 2000.0))},
        coords={"y": y, "x": x},
    )
    ds.attrs["crs"] = "EPSG:4326"
    return ds


class TestDomainMaskLoading:
    def test_no_mask_returns_none(self):
        config = TPS2Config(domain=DomainConfig(bbox=[10, 46, 11, 47]))
        domain = Domain(config)
        domain.dem_data = _make_dem()
        assert domain._load_domain_mask() is None

    def test_mask_shapefile_rasterized(self, tmp_path):
        gpd = __import__("geopandas")
        from shapely.geometry import box

        # Polygon covering the left half of the DEM extent
        shp = tmp_path / "mask.shp"
        gdf = gpd.GeoDataFrame(
            {"id": [1]}, geometry=[box(10.0, 46.0, 10.5, 47.0)], crs="EPSG:4326"
        )
        gdf.to_file(shp)

        config = TPS2Config(
            domain=DomainConfig(bbox=[10, 46, 11, 47], mask_shapefile=shp)
        )
        domain = Domain(config)
        domain.dem_data = _make_dem()
        mask = domain._load_domain_mask()

        assert mask is not None
        assert mask.shape == domain.dem_data["elevation"].shape
        # Some pixels inside, some outside
        assert mask.any() and not mask.all()
        # Left half kept, right edge dropped
        assert mask[:, 0].any()
        assert not mask[:, -1].any()


class TestDomain:
    def test_from_config(self):
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "central_asia_basic.yaml"
        )
        domain = Domain.from_config(yaml_path)
        assert domain.config.domain.bbox == [68.0, 38.0, 78.0, 43.0]

    def test_info_before_setup(self):
        config = TPS2Config(domain=DomainConfig(bbox=[0, 0, 1, 1]))
        domain = Domain(config)
        info = domain.info()
        assert "Domain Summary" in info
        assert "BBox" in info

    def test_info_has_kernel_backend(self):
        config = TPS2Config(domain=DomainConfig(bbox=[0, 0, 1, 1]))
        domain = Domain(config)
        info = domain.info()
        assert "rust" in info
