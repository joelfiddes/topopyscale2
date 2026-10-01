"""Tests for CatchmentUnitGenerator."""

import pytest

try:
    import geopandas as gpd
    from shapely.geometry import box

    HAS_GEO = True
except ImportError:
    HAS_GEO = False

from topopyscale2.spatial.catchments import CatchmentUnitGenerator

pytestmark = pytest.mark.skipif(not HAS_GEO, reason="geopandas/shapely not installed")


def _make_catchment_shapefile(tmp_path, n=4):
    """Create a simple catchment shapefile with a linear chain A→B→C→D."""
    ids = [str(i) for i in range(1, n + 1)]
    downstream = [str(i) for i in range(2, n + 1)] + ["0"]
    areas = [100.0 + i * 10 for i in range(n)]

    # Simple non-overlapping boxes
    geoms = [box(i, 0, i + 1, 1) for i in range(n)]

    gdf = gpd.GeoDataFrame(
        {
            "HYBAS_ID": ids,
            "NEXT_DOWN": downstream,
            "SUB_AREA": areas,
        },
        geometry=geoms,
    )

    shp_path = tmp_path / "catchments.shp"
    gdf.to_file(shp_path)
    return shp_path, ids, downstream, areas


@pytest.fixture
def catchment_setup(tmp_path):
    shp_path, ids, downstream, areas = _make_catchment_shapefile(tmp_path)
    gen = CatchmentUnitGenerator(shp_path)
    return gen, ids, downstream, areas


class TestCatchmentLoad:
    def test_load(self, catchment_setup):
        gen, ids, _, _ = catchment_setup
        gdf = gen.load()
        assert len(gdf) == len(ids)
        assert set(gdf["HYBAS_ID"].values) == set(ids)

    def test_missing_column_raises(self, tmp_path):
        gdf = gpd.GeoDataFrame(
            {"BAD_COL": ["1"]},
            geometry=[box(0, 0, 1, 1)],
        )
        shp_path = tmp_path / "bad.shp"
        gdf.to_file(shp_path)
        gen = CatchmentUnitGenerator(shp_path)
        with pytest.raises(ValueError, match="Column"):
            gen.load()


class TestGenerateUnits:
    def test_unit_count(self, catchment_setup):
        gen, ids, _, _ = catchment_setup
        units = gen.generate_units()
        assert len(units) == len(ids)

    def test_unit_ids(self, catchment_setup):
        gen, ids, _, _ = catchment_setup
        units = gen.generate_units()
        unit_ids = {u.id for u in units}
        assert unit_ids == set(ids)

    def test_unit_area(self, catchment_setup):
        gen, ids, _, areas = catchment_setup
        units = gen.generate_units()
        for unit in units:
            expected_area_km2 = areas[ids.index(unit.id)]
            assert unit.attributes["area_km2"] == expected_area_km2
            assert unit.area_m2 == expected_area_km2 * 1e6

    def test_unit_downstream_id(self, catchment_setup):
        gen, ids, downstream, _ = catchment_setup
        units = gen.generate_units()
        for unit in units:
            idx = ids.index(unit.id)
            assert unit.attributes["downstream_id"] == downstream[idx]


class TestUpstreamGraph:
    def test_graph_structure(self, catchment_setup):
        gen, ids, _, _ = catchment_setup
        graph = gen.build_upstream_graph()
        # Catchment 1 is headwater (no upstream)
        assert graph["1"] == []
        # Catchment 2 has 1 upstream
        assert graph["2"] == ["1"]
        # Last catchment has one upstream (n-1)
        n = len(ids)
        assert graph[str(n)] == [str(n - 1)]

    def test_find_all_upstream(self, catchment_setup):
        gen, ids, _, _ = catchment_setup
        # Headwater has no upstream
        assert gen.find_all_upstream("1") == set()
        # Outlet has all others upstream
        n = len(ids)
        upstream = gen.find_all_upstream(str(n))
        assert upstream == {str(i) for i in range(1, n)}

    def test_graph_cached(self, catchment_setup):
        gen, _, _, _ = catchment_setup
        g1 = gen.build_upstream_graph()
        g2 = gen.build_upstream_graph()
        assert g1 is g2  # same object, cached
