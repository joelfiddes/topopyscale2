"""Forecast/IFS inputs are optional: a reanalysis-only release lacks them."""

import subprocess
import sys

import pytest

from topopyscale2 import _optional
from topopyscale2.config.schema import InputConfig


def test_importing_inputs_loads_no_forecast_or_ifs_module():
    code = ("import sys, topopyscale2.inputs, topopyscale2.inputs.nwp_downloader, topopyscale2.domain; "
            "bad = ('hres', 'blending', 'forecast_source', 'temporal_blend', "
            "'nwp_downloader.forecast', 'nwp_downloader.ifs'); "
            "print(sorted(m for m in sys.modules if any(m.startswith('topopyscale2.inputs.' + b) for b in bad)))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


def test_lazy_names_still_resolve():
    import topopyscale2.inputs as inputs
    import topopyscale2.inputs.nwp_downloader as nwp
    # In the private repo these exist; a core-only snapshot does not ship them.
    blending = pytest.importorskip("topopyscale2.inputs.blending")
    hres = pytest.importorskip("topopyscale2.inputs.hres")
    forecast = pytest.importorskip("topopyscale2.inputs.nwp_downloader.forecast")
    ifs = pytest.importorskip("topopyscale2.inputs.nwp_downloader.ifs")

    assert inputs.HRESSource is hres.HRESSource  # an alias of IFSSource
    assert inputs.SourceBlender is blending.SourceBlender
    assert nwp.IFSLoader is ifs.IFSLoader
    assert nwp.ForecastLoader is forecast.ForecastLoader
    with pytest.raises(AttributeError):
        inputs.NoSuchSource  # noqa: B018


def test_get_source_for_an_absent_module_says_not_in_release(monkeypatch):
    import topopyscale2.inputs as inputs

    monkeypatch.setitem(inputs._SOURCE_TYPES, "hres", ("topopyscale2.inputs._not_shipped", "HRESSource"))
    with pytest.raises(RuntimeError, match="not included in this release"):
        inputs.get_source("hres", InputConfig())
    assert "hres" not in inputs.list_available_sources()
    assert {"era5", "custom"} <= set(inputs.list_available_sources())


def test_module_absent_distinguishes_absence_from_a_broken_dependency():
    absent = ModuleNotFoundError("x", name="topopyscale2.inputs.hres")
    parent = ModuleNotFoundError("x", name="topopyscale2.inputs")
    other = ModuleNotFoundError("x", name="ecmwf_opendata")
    assert _optional.module_absent(absent, "topopyscale2.inputs.hres")
    assert _optional.module_absent(parent, "topopyscale2.inputs.hres")
    assert not _optional.module_absent(other, "topopyscale2.inputs.hres")
    assert isinstance(_optional.not_in_release("IFS", absent, "topopyscale2.inputs.hres"), RuntimeError)
    assert _optional.not_in_release("IFS", other, "topopyscale2.inputs.hres") is other
