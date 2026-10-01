"""`topopyscale2.outputs` imports no writer until one is asked for (#213)."""

import subprocess
import sys

import pytest

import topopyscale2.outputs as outputs


def test_importing_the_package_loads_no_writer_module():
    code = ("import sys, topopyscale2.outputs; "
            "print(sorted(m for m in sys.modules if m.startswith('topopyscale2.outputs.')))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


@pytest.mark.parametrize("name", sorted(outputs._EXPORTS))
def test_every_lazy_name_resolves(name):
    pytest.importorskip(f"topopyscale2.outputs.{outputs._EXPORTS[name]}")
    assert getattr(outputs, name).__name__ == name


def test_star_and_dir_list_the_lazy_names():
    assert set(outputs._EXPORTS) <= set(outputs.__all__)
    assert set(outputs._EXPORTS) <= set(dir(outputs))


def test_get_writer_builds_a_forcing_writer():
    assert type(outputs.get_writer("netcdf")).__name__ == "CFNetCDFWriter"


def test_unknown_attribute_is_an_attribute_error():
    with pytest.raises(AttributeError):
        outputs.NoSuchWriter  # noqa: B018


def test_format_whose_module_is_absent_says_so(monkeypatch):
    """In a v1 snapshot the snow-model writers are not shipped."""
    monkeypatch.setitem(outputs._EXPORTS, "FSMWriter", "_not_shipped")
    monkeypatch.delitem(outputs.__dict__, "FSMWriter", raising=False)
    with pytest.raises(ValueError, match="not available in this installation"):
        outputs.get_writer("fsm")
