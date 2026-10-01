"""Tests for TopoSUB clustering."""

import numpy as np
import pytest
import xarray as xr

from topopyscale2.config.schema import ClusteringConfig, SurfaceTypeConfig
from topopyscale2.spatial.clusters import TopoSUBClustering


def make_dem_dataset(ny=20, nx=20, res=30.0, varied=False):
    """Create a synthetic DEM dataset for testing."""
    x = np.arange(nx) * res
    y = np.arange(ny) * res

    if varied:
        yy, xx = np.meshgrid(y, x, indexing="ij")
        elevation = 2000.0 + xx * 0.05 + yy * 0.03
        slope = np.full((ny, nx), 10.0) + np.random.default_rng(42).standard_normal((ny, nx)) * 5
        slope = np.clip(slope, 0, 60)
    else:
        elevation = np.full((ny, nx), 3000.0)
        slope = np.full((ny, nx), 0.0)

    aspect = np.random.default_rng(42).uniform(0, 360, (ny, nx))
    svf = np.ones((ny, nx)) * 0.95

    ds = xr.Dataset(
        {
            "elevation": (["y", "x"], elevation),
            "slope": (["y", "x"], slope),
            "aspect": (["y", "x"], aspect),
            "svf": (["y", "x"], svf),
        },
        coords={"y": y, "x": x},
    )
    ds["elevation"].attrs["resolution_m"] = res
    return ds


class TestTopoSUBClustering:
    def test_basic_clustering(self):
        ds = make_dem_dataset(varied=True)
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)

        units = clustering.fit(ds)
        assert len(units) == 5

    def test_cluster_count_matches_request(self):
        ds = make_dem_dataset(varied=True)
        for n in [3, 10, 20]:
            config = ClusteringConfig(n_clusters=n)
            clustering = TopoSUBClustering(config)
            units = clustering.fit(ds)
            assert len(units) == n

    def test_flat_terrain_fewer_effective_clusters(self):
        """Flat terrain: clusters should have similar attributes."""
        ds = make_dem_dataset(varied=False)
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)

        units = clustering.fit(ds)
        elevations = [u.elevation for u in units]
        # All clusters should have similar elevation (all 3000m)
        assert max(elevations) - min(elevations) < 1.0

    def test_varied_terrain_separation(self):
        """Varied terrain: clusters should separate by elevation/slope."""
        ds = make_dem_dataset(ny=50, nx=50, varied=True)
        config = ClusteringConfig(n_clusters=10)
        clustering = TopoSUBClustering(config)

        units = clustering.fit(ds)
        elevations = [u.elevation for u in units]
        # Should have spread in elevation
        assert max(elevations) - min(elevations) > 10.0

    def test_unit_attributes(self):
        ds = make_dem_dataset(varied=True)
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)

        units = clustering.fit(ds)
        for u in units:
            assert "elevation" in u.attributes
            assert "slope" in u.attributes
            assert u.area_m2 > 0
            assert u.surface_type == "open"

    def test_aspect_decomposition(self):
        """sin/cos aspect features should be in config by default."""
        config = ClusteringConfig()
        assert "cos_aspect" in config.features
        assert "sin_aspect" in config.features

    def test_membership_raster(self):
        ds = make_dem_dataset(varied=True)
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)

        clustering.fit(ds)
        membership = clustering.get_membership()
        assert membership is not None

    def test_no_standardization(self):
        ds = make_dem_dataset(varied=True)
        config = ClusteringConfig(n_clusters=5, feature_standardization=False)
        clustering = TopoSUBClustering(config)

        units = clustering.fit(ds)
        assert len(units) == 5


class TestDomainMask:
    """Clustering restricted to an external domain mask."""

    def test_mask_excludes_outside_pixels(self):
        ds = make_dem_dataset(ny=40, nx=40, varied=True)
        ny, nx = ds["elevation"].shape
        # Keep only the left half of the grid
        mask = np.zeros((ny, nx), dtype=bool)
        mask[:, : nx // 2] = True

        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)
        clustering.fit(ds, domain_mask=mask)

        membership = clustering.get_membership().values
        # Outside the mask: no pixel assigned (stays -1)
        assert np.all(membership[:, nx // 2 :] == -1)
        # Inside the mask: at least some pixels assigned
        assert np.any(membership[:, : nx // 2] >= 0)
        # Every assigned pixel lies inside the mask
        assert np.all(mask[membership >= 0])

    def test_mask_none_matches_unmasked(self):
        ds = make_dem_dataset(ny=30, nx=30, varied=True)
        config = ClusteringConfig(n_clusters=5)

        units_a = TopoSUBClustering(config).fit(ds, domain_mask=None)
        units_b = TopoSUBClustering(ClusteringConfig(n_clusters=5)).fit(ds)
        assert len(units_a) == len(units_b)

    def test_mask_shape_mismatch_raises(self):
        ds = make_dem_dataset(ny=20, nx=20, varied=True)
        bad_mask = np.ones((10, 10), dtype=bool)
        clustering = TopoSUBClustering(ClusteringConfig(n_clusters=3))
        with pytest.raises(ValueError, match="does not match"):
            clustering.fit(ds, domain_mask=bad_mask)


class TestClusterEvaluation:
    def test_evaluate_returns_metrics(self):
        ds = make_dem_dataset(ny=50, nx=50, varied=True)
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)
        clustering.fit(ds)

        metrics = clustering.evaluate()
        assert metrics["n_clusters"] == 5
        assert metrics["n_samples"] == 2500
        assert -1 <= metrics["silhouette"] <= 1
        assert metrics["calinski_harabasz"] > 0
        assert metrics["davies_bouldin"] > 0
        assert metrics["inertia"] > 0

    def test_evaluate_before_fit_raises(self):
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)
        with pytest.raises(RuntimeError, match="Must call fit"):
            clustering.evaluate()

    def test_elbow_analysis(self):
        ds = make_dem_dataset(ny=50, nx=50, varied=True)
        config = ClusteringConfig(n_clusters=5)
        clustering = TopoSUBClustering(config)
        clustering.fit(ds)

        results = clustering.elbow_analysis(ds, k_values=[3, 5, 10])
        assert len(results) == 3
        assert results[0]["n_clusters"] == 3
        assert results[1]["n_clusters"] == 5
        assert results[2]["n_clusters"] == 10
        # Inertia should decrease with more clusters
        assert results[0]["inertia"] > results[2]["inertia"]


class TestSurfaceTypeStratification:
    def test_stratified_clustering(self):
        ds = make_dem_dataset(ny=20, nx=20, varied=True)
        surface_types = xr.DataArray(
            np.where(ds["elevation"].values > 2015, 1, 2),
            dims=["y", "x"],
            coords={"y": ds.y, "x": ds.x},
        )

        config = ClusteringConfig(n_clusters=3)
        surface_config = SurfaceTypeConfig(stratify=True)
        clustering = TopoSUBClustering(config, surface_config)

        units = clustering.fit(ds, surface_types)
        # Should have clusters from both types
        assert len(units) > 0


class TestGlacierAwareClustering:
    """Stratified clustering must give glacier terrain its own units.

    Without this, a domain-wide k-means puts ice into mixed terrain classes:
    the class is then neither glacier nor not-glacier, and GFSM cannot be
    switched on for it without also switching it on for the rock in the same
    class. Measured on the production Central Asia domain, majority-ice
    clusters captured only 38% of the ice and sat at a median 5144 m -- the
    accumulation zones -- while the ablation tongues that actually melt were
    absorbed into mixed classes.
    """

    @staticmethod
    def _glaciated_dem(ny=40, nx=40):
        """A valley with an ice tongue running down it.

        Elevation falls to the south; ice occupies a band that spans a wide
        elevation range, so a correct stratification has to split the ice by
        elevation rather than lump it into one unit.
        """
        y = np.arange(ny) * 100.0
        x = np.arange(nx) * 100.0
        yy, xx = np.meshgrid(y, x, indexing="ij")
        elevation = 5500.0 - yy * 0.5 + xx * 0.02
        rng = np.random.default_rng(0)
        ds = xr.Dataset(
            {
                "elevation": (["y", "x"], elevation),
                "slope": (["y", "x"], np.full((ny, nx), 20.0) + rng.standard_normal((ny, nx))),
                "aspect": (["y", "x"], rng.uniform(0, 360, (ny, nx))),
                "svf": (["y", "x"], np.full((ny, nx), 0.9)),
            },
            coords={"y": y, "x": x},
        )
        ds["elevation"].attrs["resolution_m"] = 100.0

        surface = np.zeros((ny, nx), dtype=np.int32)
        surface[:, 12:20] = 1  # glacier: a north-south band, all elevations
        st = xr.DataArray(surface, dims=["y", "x"], coords={"y": y, "x": x})
        return ds, st

    def test_glacier_units_contain_only_glacier_pixels(self):
        ds, st = self._glaciated_dem()
        clustering = TopoSUBClustering(
            ClusteringConfig(n_clusters=20), SurfaceTypeConfig(stratify=True)
        )
        units = clustering.fit(ds, surface_types=st)
        membership = clustering.get_membership().values

        glacier_ids = {u.id for u in units if u.surface_type == "glacier"}
        assert glacier_ids, "stratification must produce glacier units"

        is_ice = st.values == 1
        by_index = {i: u.id for i, u in enumerate(units)}
        for idx in np.unique(membership[membership >= 0]):
            pixels_are_ice = is_ice[membership == idx]
            pure = pixels_are_ice.all() or (~pixels_are_ice).all()
            assert pure, f"unit {by_index[int(idx)]} mixes glacier and non-glacier pixels"

    def test_unstratified_clustering_mixes_them(self):
        """The behaviour this feature exists to replace, pinned so the
        comparison in the docs stays honest."""
        ds, st = self._glaciated_dem()
        clustering = TopoSUBClustering(ClusteringConfig(n_clusters=20))
        units = clustering.fit(ds)  # no surface types -> single k-means
        membership = clustering.get_membership().values

        is_ice = st.values == 1
        mixed = sum(
            1 for idx in np.unique(membership[membership >= 0])
            if is_ice[membership == idx].any() and not is_ice[membership == idx].all()
        )
        assert mixed > 0, "unstratified clustering is expected to mix ice with rock"
        assert all(u.surface_type == "open" for u in units)

    def test_tongue_is_split_by_elevation(self):
        """One glacier unit for 1500 m of ice would be useless: the tongue melts
        out and the accumulation zone never does, and a single unit cannot do
        both."""
        ds, st = self._glaciated_dem()
        clustering = TopoSUBClustering(
            ClusteringConfig(n_clusters=20), SurfaceTypeConfig(stratify=True)
        )
        units = clustering.fit(ds, surface_types=st)

        gl = [u for u in units if u.surface_type == "glacier"]
        assert len(gl) >= 2, "the ice band must be split into several units"

        elevs = sorted(u.centroid[2] for u in gl)
        assert elevs[-1] - elevs[0] > 300, (
            "glacier units should span the tongue's elevation range, "
            f"got {elevs[0]:.0f}-{elevs[-1]:.0f} m"
        )

    def test_per_type_cluster_budget_is_respected(self):
        """Ice is a small area fraction, so proportional allocation starves it.
        An explicit budget is how a user gets enough glacier units to resolve
        tongues."""
        from topopyscale2.config.schema import SurfaceTypeEntry

        ds, st = self._glaciated_dem()
        cfg = SurfaceTypeConfig(
            stratify=True, types={"glacier": SurfaceTypeEntry(n_clusters=6)}
        )
        clustering = TopoSUBClustering(ClusteringConfig(n_clusters=20), cfg)
        units = clustering.fit(ds, surface_types=st)

        assert len([u for u in units if u.surface_type == "glacier"]) == 6

    def test_proportional_allocation_counts_only_in_domain_pixels(self):
        """A domain mask must shrink the pool the shares are computed from.

        The glacier band here is entirely inside the mask while most open
        terrain is outside it, so counting raw raster pixels would hand the
        ice a share of a domain it does not live in.
        """
        ds, st = self._glaciated_dem()
        domain = np.zeros(ds["elevation"].shape, dtype=bool)
        domain[:, 10:22] = True  # keeps the ice band, drops most open ground

        clustering = TopoSUBClustering(
            ClusteringConfig(n_clusters=20), SurfaceTypeConfig(stratify=True)
        )
        units = clustering.fit(ds, surface_types=st, domain_mask=domain)
        membership = clustering.get_membership().values

        assert (membership[~domain] == -1).all(), "masked pixels must stay unassigned"

        n_glacier = len([u for u in units if u.surface_type == "glacier"])
        ice_share = (st.values[domain] == 1).mean()
        expected = max(1, int(20 * ice_share))
        assert abs(n_glacier - expected) <= 1, (
            f"glacier got {n_glacier} clusters, in-domain ice share {ice_share:.2f} "
            f"implies about {expected}"
        )

    def test_stratification_survives_terrain_adaptive(self):
        """Both splits must apply. Ranking them either-or silently discarded the
        surface types whenever terrain_adaptive was set, which is exactly the
        production Central Asia configuration: glacier stratification would have
        looked enabled in the config and done nothing at all."""
        ds, st = self._glaciated_dem()
        cfg = ClusteringConfig(n_clusters=20, terrain_adaptive=True,
                               terrain_slope_threshold=5.0)
        clustering = TopoSUBClustering(cfg, SurfaceTypeConfig(stratify=True))
        units = clustering.fit(ds, surface_types=st)
        membership = clustering.get_membership().values

        assert any(u.surface_type == "glacier" for u in units), (
            "terrain_adaptive must not swallow the surface-type split"
        )

        is_ice = st.values == 1
        for idx in np.unique(membership[membership >= 0]):
            px = is_ice[membership == idx]
            assert px.all() or (~px).all(), "units must stay surface-type pure"

    def test_unit_ids_are_unique_across_strata_and_zones(self):
        """Two nested splits means two id offsets; getting that wrong silently
        collides ids between the flat zone of one stratum and another."""
        ds, st = self._glaciated_dem()
        cfg = ClusteringConfig(n_clusters=20, terrain_adaptive=True)
        clustering = TopoSUBClustering(cfg, SurfaceTypeConfig(stratify=True))
        units = clustering.fit(ds, surface_types=st)

        ids = [u.id for u in units]
        assert len(ids) == len(set(ids)), "unit ids must be unique"

        membership = clustering.get_membership().values
        assigned = np.unique(membership[membership >= 0])
        assert assigned.max() < len(units), "membership indexes past the unit list"
