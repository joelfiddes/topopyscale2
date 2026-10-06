"""`tps2 export`: every format writes, and the physical values in the files are right.

The writers' own tests checked structure, not values, which is how SMET and HBV came to
export 1000x too much precipitation and FSM/FSM2 100x too little humidity (fixed 2026-10-06).
Here a run with known values goes through every format and the numbers are read back.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.outputs.base import with_cf_metadata
from topopyscale2.outputs.export import FORMATS, export, units_from_forcing

PRECIP_MM_PER_H = 2.0      # kg m-2 per hourly step, all as snow
RH = 0.8                   # fraction, as tps2 run writes it
T_K = 268.15               # -5 degC


@pytest.fixture
def sim_dir(tmp_path):
    time = pd.date_range("2024-01-01", periods=48, freq="h")
    n = 2
    full = lambda v: np.full((len(time), n), v)  # noqa: E731
    ds = xr.Dataset(
        {
            "temperature": (("time", "unit"), full(T_K)),
            "precipitation": (("time", "unit"), full(PRECIP_MM_PER_H)),
            "snowfall": (("time", "unit"), full(PRECIP_MM_PER_H)),
            "rainfall": (("time", "unit"), full(0.0)),
            "shortwave_direct": (("time", "unit"), full(100.0)),
            "shortwave_diffuse": (("time", "unit"), full(50.0)),
            "longwave": (("time", "unit"), full(250.0)),
            "humidity_relative": (("time", "unit"), full(RH)),
            "humidity_specific": (("time", "unit"), full(0.002)),
            "wind_speed": (("time", "unit"), full(3.0)),
            "wind_direction": (("time", "unit"), full(270.0)),
            "pressure": (("time", "unit"), full(75000.0)),
        },
        coords={"time": time, "unit": ["davos", "wfj"],
                "latitude": ("unit", [46.81, 46.83]), "longitude": ("unit", [9.85, 9.81]),
                "elevation": ("unit", [1560.0, 2536.0])},
    )
    (tmp_path / "output").mkdir()
    with_cf_metadata(ds, engine_ref="test", run_date="2026-10-06").to_netcdf(tmp_path / "output" / "forcing.nc")
    return tmp_path


def _rows(path):
    return [ln.split() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith(("#", "[", "SMET")) and "=" not in ln]


def test_units_come_from_the_forcing_file(sim_dir):
    ds = xr.open_dataset(sim_dir / "output" / "forcing.nc")
    units = units_from_forcing(ds)
    assert [u.id for u in units] == ["davos", "wfj"]
    assert units[1].elevation == 2536.0 and units[1].y == 46.83 and units[1].x == 9.81


@pytest.mark.parametrize("fmt", sorted(FORMATS))
def test_every_format_writes(sim_dir, fmt):
    paths = export(sim_dir, fmt)
    assert paths and all(p.exists() and p.stat().st_size > 0 for p in paths)
    assert all(p.parent == sim_dir / "output" / fmt for p in paths)


def test_smet_values(sim_dir):
    path = next(p for p in export(sim_dir, "smet") if "wfj" in p.name)
    text = path.read_text(encoding="utf-8")
    fields = next(ln for ln in text.splitlines() if ln.strip().startswith("fields")).split("=")[1].split()
    row = dict(zip(fields, _rows(path)[0]))
    assert float(row["PSUM"]) == pytest.approx(PRECIP_MM_PER_H)       # mm per step, not x1000
    assert float(row["RH"]) == pytest.approx(RH * 100)                  # percent
    assert float(row["TA"]) == pytest.approx(T_K)


def test_fsm_values(sim_dir):
    path = next(p for p in export(sim_dir, "fsm") if "wfj" in p.name)
    yr, mo, dy, hr, sw, lw, sf, rf, ta, rh, ua, ps = map(float, _rows(path)[0])
    assert sf == pytest.approx(PRECIP_MM_PER_H / 3600.0)               # kg m-2 s-1
    assert rh == pytest.approx(RH * 100)                                 # FSM reads RH in %
    assert sw == pytest.approx(150.0) and ps == pytest.approx(75000.0)


def test_fsm2_values(sim_dir):
    (path,) = export(sim_dir, "fsm2")
    rows = _rows(path)
    assert len(rows) == 48 * 2                                           # every unit, every hour
    assert float(rows[0][9]) == pytest.approx(RH * 100)


def test_hbv_daily_totals(sim_dir):
    path = next(p for p in export(sim_dir, "hbv") if "wfj" in p.name)
    date, t, p, pet = _rows(path)[0][:4]
    assert float(p) == pytest.approx(24 * PRECIP_MM_PER_H)              # mm/day, not x1000
    assert float(t) == pytest.approx(T_K - 273.15)


@pytest.mark.parametrize("fmt", ["crocus", "cryogrid"])
def test_netcdf_formats_open(sim_dir, fmt):
    for path in export(sim_dir, fmt):
        with xr.open_dataset(path) as ds:
            assert ds.sizes["time"] == 48


def test_unknown_format_and_missing_run(tmp_path):
    with pytest.raises(ValueError, match="unknown export format"):
        export(tmp_path, "grib")
    with pytest.raises(FileNotFoundError, match="tps2 run"):
        export(tmp_path, "smet")
