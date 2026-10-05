"""Forcing page (`tps2 view`): daily summary, compact file, deterministic page."""

import json
import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.outputs import forcing_page as fp

N_UNITS = 3


@pytest.fixture
def sim_dir(tmp_path):
    """A 3-unit, 3-day hourly run with known values."""
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    (tmp_path / "dem_cache").mkdir()
    (tmp_path / "output").mkdir()
    dem = np.array([[1000, 1500, 2000], [1000, 1500, 2000]], dtype="float32")
    cmap = np.array([[0, 1, 2], [0, 1, 2]], dtype="uint8")
    tr = from_origin(9.7, 46.9, 0.001, 0.001)
    for name, arr in (("dem.tif", dem), ("cluster_map.tif", cmap)):
        with rasterio.open(tmp_path / "dem_cache" / name, "w", driver="GTiff", height=2, width=3, count=1,
                           dtype=arr.dtype, crs="EPSG:4326", transform=tr) as r:
            r.write(arr, 1)

    time = pd.date_range("2024-01-01", periods=72, freq="h")
    shape = (len(time), N_UNITS)
    t_k = np.broadcast_to(273.15 + np.array([5.0, 2.0, -1.0]), shape)
    ds = xr.Dataset(
        {
            "temperature": (("time", "unit"), t_k.copy(), {"units": "K"}),
            "precipitation": (("time", "unit"), np.full(shape, 0.5), {"units": "kg m-2"}),     # 12 mm/day
            "snowfall": (("time", "unit"), np.broadcast_to([0.0, 0.25, 0.5], shape).copy(), {"units": "kg m-2"}),
            "shortwave_direct": (("time", "unit"), np.full(shape, 100.0), {"units": "W m-2"}),
            "shortwave_diffuse": (("time", "unit"), np.full(shape, 50.0), {"units": "W m-2"}),
            "longwave": (("time", "unit"), np.full(shape, 280.0), {"units": "W m-2"}),
            "wind_speed": (("time", "unit"), np.full(shape, 3.0), {"units": "m s-1"}),
            "humidity_relative": (("time", "unit"), np.full(shape, 0.8), {"units": "1"}),
        },
        coords={"time": time, "unit": np.arange(N_UNITS)},
    )
    ds.to_netcdf(tmp_path / "output" / "forcing.nc")
    (tmp_path / "config.yaml").write_text("domain:\n  bbox: [9.7, 46.8, 9.8, 46.9]\n")
    return tmp_path


def test_daily_summary_converts_units(sim_dir):
    d = fp.load_daily(sim_dir)
    assert d.sizes == {"time": 3, "unit": N_UNITS}
    np.testing.assert_allclose(d["t_mean"].isel(time=0), [5.0, 2.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(d["precip"], 12.0)                        # 24 × 0.5 mm per step
    np.testing.assert_allclose(d["snowfall"].isel(time=0), [0.0, 6.0, 12.0])
    np.testing.assert_allclose(d["sw"], 150.0)
    np.testing.assert_allclose(d["rh"], 80.0)                            # fraction -> percent


def test_rate_units_are_integrated_over_the_step():
    time = pd.date_range("2024-01-01", periods=48, freq="h")
    da = xr.DataArray(np.full(48, 1.0 / 3600.0), coords={"time": time}, dims="time",
                      attrs={"units": "kg m-2 s-1"})
    np.testing.assert_allclose(fp._amount_per_step(da, 1.0), 1.0)
    with pytest.raises(ValueError, match="not recognised"):
        fp._amount_per_step(da.assign_attrs(units="furlongs"), 1.0)


def test_compact_daily_file_round_trips(sim_dir):
    full = fp.load_daily(sim_dir)
    path = fp.write_daily_summary(sim_dir, engine_ref="test-ref", run_date="2026-01-01")
    assert path.name == fp.DAILY_FILE
    packed = fp.load_daily(sim_dir)   # now reads the daily file first
    for name, (*_, scale, _ramp, _rng) in fp.LAYERS.items():
        np.testing.assert_allclose(packed[name], full[name], atol=0.5 / scale + 1e-9)
    assert packed.attrs["tps2_engine"] == "test-ref"


def test_page_is_deterministic_and_embeds_every_layer(sim_dir, tmp_path):
    a = fp.generate_forcing_page(sim_dir, tmp_path / "a.html", engine_ref="x", run_date="y")
    b = fp.generate_forcing_page(sim_dir, tmp_path / "b.html", engine_ref="x", run_date="y")
    assert a.read_bytes() == b.read_bytes()
    html = a.read_text(encoding="utf-8")
    data = json.loads(re.search(r"const D = (\{.*?\});\n", html, re.S).group(1))
    assert data["layer_order"] == list(fp.LAYERS)
    assert data["n_units"] == N_UNITS and data["ndays"] == 3
    assert data["unit_elev"] == [1000, 1500, 2000]
    assert data["stamp"]["engine"] == "x"
    assert "/*__DATA__*/" not in html


def test_missing_forcing_says_run_first(tmp_path):
    (tmp_path / "output").mkdir()
    with pytest.raises(FileNotFoundError, match="tps2 run"):
        fp.load_daily(tmp_path)


def test_shipped_demo_is_reproducible_offline(tmp_path):
    """examples/forcing_demo/demo.html is exactly what `tps2 view` makes from the shipped inputs."""
    from pathlib import Path

    pytest.importorskip("rasterio")
    demo = Path(__file__).resolve().parents[2] / "examples" / "forcing_demo"
    if not (demo / "page.yaml").exists():
        pytest.skip("forcing demo not in this tree")
    out = fp.generate_forcing_page(demo / "run", tmp_path / "demo.html",
                                   page=fp.PageText.from_yaml(demo / "page.yaml"))
    assert out.read_bytes() == (demo / "demo.html").read_bytes(), (
        "demo.html is stale: rerun `tps2 view examples/forcing_demo/run --page "
        "examples/forcing_demo/page.yaml -o examples/forcing_demo/demo.html` and commit it")
