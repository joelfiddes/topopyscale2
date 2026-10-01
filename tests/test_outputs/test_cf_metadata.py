"""Forcing written by `tps2 run` carries units and provenance (with_cf_metadata)."""

import numpy as np
import xarray as xr

from topopyscale2.outputs.base import CF_ATTRIBUTES, with_cf_metadata


def _ds():
    names = ["temperature", "precipitation", "humidity_relative", "wind_direction", "custom_extra"]
    return xr.Dataset({n: (("time", "unit"), np.zeros((2, 3))) for n in names})


def test_every_known_variable_gets_its_units():
    out = with_cf_metadata(_ds(), engine_ref="e", run_date="d")
    for name in ("temperature", "precipitation", "humidity_relative", "wind_direction"):
        assert out[name].attrs["units"] == CF_ATTRIBUTES[name]["units"], name
    assert out["humidity_relative"].attrs["units"] == "1"          # a fraction, not percent
    assert out["precipitation"].attrs["units"] == "kg m-2"         # amount per time step
    assert "units" not in out["custom_extra"].attrs


def test_provenance_is_stamped_and_input_untouched():
    ds = _ds()
    out = with_cf_metadata(ds, engine_ref="0.1.0 @ abc", run_date="2026-09-30")
    assert out.attrs["tps2_engine"] == "0.1.0 @ abc" and out.attrs["tps2_run_date"] == "2026-09-30"
    assert out.attrs["Conventions"].startswith("CF-")
    assert not ds.attrs and not ds["temperature"].attrs


def test_netcdf_round_trip_keeps_units(tmp_path):
    with_cf_metadata(_ds(), engine_ref="e", run_date="d").to_netcdf(tmp_path / "f.nc")
    with xr.open_dataset(tmp_path / "f.nc") as back:
        assert back["temperature"].attrs["units"] == "K"
        assert back.attrs["tps2_engine"] == "e"
