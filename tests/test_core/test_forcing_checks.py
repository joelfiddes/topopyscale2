"""Physical range checks on forcing (core/forcing_checks.py)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.core.forcing_checks import ForcingRangeError, check_forcing, check_fsm_inputs


def forcing(**override):
    """Two units, 48 hours, values inside every real domain's measured range."""
    time = pd.date_range("2024-01-01", periods=48, freq="h")
    base = {"temperature": 268.0, "humidity_relative": 0.8, "humidity_specific": 0.003,
            "shortwave_direct": 300.0, "shortwave_diffuse": 80.0, "longwave": 250.0,
            "wind_speed": 4.0, "pressure": 75000.0, "precipitation": 1.5,
            "rainfall": 0.0, "snowfall": 1.5}
    base.update(override)
    return xr.Dataset({k: (("time", "unit"), np.full((48, 2), float(v))) for k, v in base.items()},
                      coords={"time": time, "unit": ["a", "b"]})


def test_realistic_forcing_passes():
    check_forcing(forcing())


@pytest.mark.parametrize("override,hint", [
    ({"humidity_relative": 80.0}, "percent"),
    ({"precipitation": 1500.0}, "unit factor"),          # mm treated as metres x1000
    ({"snowfall": 2000.0}, "unit factor"),
    ({"temperature": -5.0}, "Celsius"),
    ({"pressure": 750.0}, "hPa"),
    ({"wind_speed": -3.0}, "outside"),
])
def test_unit_mistakes_are_caught_with_a_hint(override, hint):
    with pytest.raises(ForcingRangeError, match=hint):
        check_forcing(forcing(**override), "test")


def test_nan_units_are_ignored():
    """Some domains carry empty units whose forcing is NaN throughout: not a range error."""
    ds = forcing()
    ds["temperature"][:, 1] = np.nan
    check_forcing(ds)


def test_precip_bound_scales_with_the_time_step():
    ds = forcing(precipitation=500.0).isel(time=slice(None, None, 3))   # 3-hourly: 500 mm/3h ok
    check_forcing(ds)
    with pytest.raises(ForcingRangeError):
        check_forcing(forcing(precipitation=500.0))                     # 500 mm in one hour


def test_warn_mode_logs_instead_of_raising(monkeypatch, caplog):
    monkeypatch.setenv("TPS2_FORCING_CHECKS", "warn")
    check_forcing(forcing(humidity_relative=80.0), "test")
    assert "implausible forcing" in caplog.text


def test_fsm_inputs_in_fsm_units():
    ok = dict(SW=np.array([400.0]), LW=np.array([250.0]), Sf=np.array([1.5 / 3600]),
              Rf=np.array([0.0]), Ta=np.array([268.0]), RH=np.array([80.0]),
              Ua=np.array([4.0]), Ps=np.array([75000.0]))
    check_fsm_inputs(**ok)
    check_fsm_inputs(**{**ok, "RH": np.array([0.8])})                   # one dry value: possible
    with pytest.raises(ForcingRangeError, match="fraction"):            # the FSM writer bug
        check_fsm_inputs(**{**ok, "RH": np.full(48, 0.8)})
    with pytest.raises(ForcingRangeError, match="multiplied by 100 again"):
        check_fsm_inputs(**{**ok, "RH": np.array([8000.0])})
    with pytest.raises(ForcingRangeError):
        check_fsm_inputs(**{**ok, "Sf": np.array([1.5])})              # mm/h passed as kg m-2 s-1


def test_fsm1_rejects_percent_humidity_given_as_forcing():
    """A dataset whose RH is already % would reach FSM at 8000 %: the native path must stop."""
    fsm1 = pytest.importorskip("topopyscale2.applications.snowmapper.fsm1_native")
    ds = forcing(humidity_relative=80.0).rename({"unit": "unit_id"})
    with pytest.raises(ForcingRangeError, match="multiplied by 100 again"):
        fsm1._extract_forcing(ds, 3600.0)
