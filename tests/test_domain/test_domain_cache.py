"""Domain cache writes for domains configured with a local DEM.

A domain that points `domain.dem` at a file on disk never downloads a DEM, so
nothing creates `storage.dem_cache` for it. Both cache writers used to assume
that directory existed: `_save_domain_state` raised FileNotFoundError (swallowed
by a bare warning, so units were silently re-derived on every run) and
`_save_membership_map` returned early because `dem.tif` was absent.

Everything here uses a tiny synthetic DEM written with rasterio in tmp_path —
no network, no real data.
"""

import pickle

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

from topopyscale2.config.schema import (
    ClusteringConfig,
    DomainConfig,
    StorageConfig,
    TPS2Config,
)
from topopyscale2.domain import Domain

DEM_CRS = "EPSG:32632"
DEM_RES = 100.0
DEM_ORIGIN = (400000.0, 5100000.0)  # upper-left corner, metres
NY, NX = 24, 20


def _write_synthetic_dem(path):
    """A small tilted+bumpy DEM so k-means has something to separate."""
    yy, xx = np.meshgrid(np.arange(NY), np.arange(NX), indexing="ij")
    elevation = (
        2000.0
        + 20.0 * yy
        + 10.0 * xx
        + 50.0 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    ).astype(np.float32)

    transform = from_origin(DEM_ORIGIN[0], DEM_ORIGIN[1], DEM_RES, DEM_RES)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=NY,
        width=NX,
        count=1,
        dtype=rasterio.float32,
        crs=CRS.from_user_input(DEM_CRS),
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(elevation, 1)
    return path


def _make_config(tmp_path):
    """Config whose dem_cache directory deliberately does NOT exist yet."""
    dem_path = _write_synthetic_dem(tmp_path / "local_dem.tif")
    dem_cache = tmp_path / "dem_cache"
    assert not dem_cache.exists()
    return TPS2Config(
        domain=DomainConfig(
            dem=dem_path,
            crs=DEM_CRS,
            spatial_mode="clusters",
            svf_mode="approximate",
        ),
        clustering=ClusteringConfig(
            n_clusters=4,
            features=["elevation", "slope"],
        ),
        storage=StorageConfig(
            dem_cache=dem_cache,
            forcing_cache=tmp_path / "forcing_cache",
        ),
    )


class TestDomainStateCacheWithoutDemCacheDir:
    def test_setup_creates_cache_dir_and_writes_state(self, tmp_path):
        config = _make_config(tmp_path)
        dem_cache = config.storage.dem_cache

        domain = Domain(config)
        domain.setup()

        cache_path = dem_cache / "domain_state.pkl"
        assert dem_cache.is_dir(), "dem_cache should have been created by the save"
        assert cache_path.exists(), "domain_state.pkl was not written"

        state = pickle.loads(cache_path.read_bytes())
        assert state["config_hash"] == domain._domain_config_hash()
        assert len(state["units"]) == len(domain.units)

    def test_second_domain_loads_cached_units(self, tmp_path):
        config = _make_config(tmp_path)

        first = Domain(config)
        first.setup()
        first_ids = [u.id for u in first.units]

        second = Domain(_make_config_reusing(config))
        assert second.load_cached_units() is True
        assert [u.id for u in second.units] == first_ids
        assert second.dem_data is not None

    def test_setup_of_second_domain_reuses_cache(self, tmp_path):
        """setup() on a second Domain must not re-cluster."""
        config = _make_config(tmp_path)
        Domain(config).setup()

        second = Domain(_make_config_reusing(config))
        second.setup()
        # _clustering is only set when clustering actually ran
        assert second._clustering is None
        assert second.units is not None and len(second.units) > 0


def _make_config_reusing(config: TPS2Config) -> TPS2Config:
    """A fresh config object pointing at the same DEM and cache."""
    return TPS2Config(
        domain=config.domain.model_copy(deep=True),
        clustering=config.clustering.model_copy(deep=True),
        storage=config.storage.model_copy(deep=True),
    )


class TestAtomicDomainStateWrite:
    def test_failed_dump_leaves_no_partial_or_temp_file(self, tmp_path, monkeypatch):
        config = _make_config(tmp_path)
        dem_cache = config.storage.dem_cache

        domain = Domain(config)

        def _boom(*args, **kwargs):
            raise RuntimeError("disk went away mid-dump")

        monkeypatch.setattr("topopyscale2.domain.pickle.dump", _boom)

        # Non-fatal: setup() must complete even though the cache write failed
        domain.setup()

        assert not (dem_cache / "domain_state.pkl").exists()
        leftovers = [p.name for p in dem_cache.iterdir() if p.name.endswith(".tmp")]
        assert leftovers == [], f"temp files left behind: {leftovers}"
        assert not any(
            p.name.startswith(".domain_state-") for p in dem_cache.iterdir()
        )

    def test_failure_is_logged_at_error(self, tmp_path, monkeypatch, caplog):
        config = _make_config(tmp_path)
        domain = Domain(config)

        def _boom(*args, **kwargs):
            raise RuntimeError("disk went away mid-dump")

        monkeypatch.setattr("topopyscale2.domain.pickle.dump", _boom)
        with caplog.at_level("ERROR", logger="topopyscale2.domain"):
            domain.setup()

        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert errors, "a failed cache write must be logged at ERROR"
        assert "re-derived" in errors[0].getMessage()

    def test_existing_cache_survives_a_failed_rewrite(self, tmp_path, monkeypatch):
        """A half-written dump must not clobber a good cache."""
        config = _make_config(tmp_path)
        cache_path = config.storage.dem_cache / "domain_state.pkl"

        Domain(config).setup()
        good_bytes = cache_path.read_bytes()

        def _boom(*args, **kwargs):
            raise RuntimeError("interrupted")

        monkeypatch.setattr("topopyscale2.domain.pickle.dump", _boom)
        domain = Domain(_make_config_reusing(config))
        domain._try_load_cached_domain()
        domain._save_domain_state()

        assert cache_path.read_bytes() == good_bytes


class TestMembershipMapWithoutDemTif:
    def test_cluster_map_written_from_dem_data(self, tmp_path):
        config = _make_config(tmp_path)
        dem_cache = config.storage.dem_cache

        domain = Domain(config)
        domain.setup()

        assert not (dem_cache / "dem.tif").exists(), "local-DEM run must not cache dem.tif"
        cluster_map = dem_cache / "cluster_map.tif"
        assert cluster_map.exists(), "cluster_map.tif was not written"

        with rasterio.open(cluster_map) as src:
            assert (src.height, src.width) == (NY, NX)
            assert src.crs == CRS.from_user_input(DEM_CRS)
            assert src.dtypes[0] == "int16"
            assert src.nodata == -1
            assert src.res == pytest.approx((DEM_RES, DEM_RES))
            # Upper-left corner matches the source DEM's origin
            assert src.transform.c == pytest.approx(DEM_ORIGIN[0])
            assert src.transform.f == pytest.approx(DEM_ORIGIN[1])
            data = src.read(1)

        assert data.shape == (NY, NX)
        ids = set(np.unique(data)) - {-1}
        assert ids, "cluster map holds no unit ids"
        assert ids == {u_id for u_id in ids if u_id >= 0}
        np.testing.assert_array_equal(
            data, np.asarray(domain._membership).astype(np.int16)
        )

    def test_dem_tif_is_still_preferred_when_present(self, tmp_path):
        """With a cached dem.tif the georeferencing comes from it, as before."""
        config = _make_config(tmp_path)
        dem_cache = config.storage.dem_cache
        dem_cache.mkdir(parents=True, exist_ok=True)
        _write_synthetic_dem(dem_cache / "dem.tif")

        domain = Domain(config)
        domain.setup()

        with rasterio.open(dem_cache / "cluster_map.tif") as src:
            assert (src.height, src.width) == (NY, NX)
            assert src.crs == CRS.from_user_input(DEM_CRS)
            assert src.dtypes[0] == "int16"
            assert src.nodata == -1

    def test_profile_handles_descending_and_ascending_y(self, tmp_path):
        """The affine is derived from the coordinates, not assumed north-up."""
        config = _make_config(tmp_path)
        domain = Domain(config)
        domain.setup()

        profile = domain._membership_profile_from_dem_data((NY, NX))
        y = np.asarray(domain.dem_data["y"].values, dtype=float)
        expected_res_y = float(y[1] - y[0])
        assert profile["transform"].e == pytest.approx(expected_res_y)
        assert profile["transform"].a == pytest.approx(DEM_RES)
        assert profile["height"] == NY and profile["width"] == NX
