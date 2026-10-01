"""Commands outside the v1 core register only when their module ships (#212)."""

import subprocess
import sys
import textwrap

import pytest

from tests._release import requires

V1_COMMANDS = {"init", "setup", "fetch-forcing", "run", "info", "preflight", "view", "ui",
               "evaluate-clusters", "build-cache"}


def _commands_with(prelude: str) -> subprocess.CompletedProcess:
    code = textwrap.dedent(prelude) + textwrap.dedent("""
        import typer.main
        from topopyscale2.cli.main import app
        print(sorted(typer.main.get_command(app).commands))
    """)
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


@requires("topopyscale2.cli.stations_cli", "topopyscale2.cli.forecast_cli", "topopyscale2.cli.snow_cli", "topopyscale2.cli.da_cli", "topopyscale2.cli.climate_cli", "topopyscale2.cli.lab_cli")
def test_all_commands_register_in_the_full_repo():
    out = _commands_with("")
    names = set(eval(out.stdout))
    assert V1_COMMANDS | {"snowmapper", "forecast", "da", "qc", "climate", "lab"} <= names


@requires("topopyscale2.cli.stations_cli", "topopyscale2.cli.forecast_cli")
def test_absent_module_drops_only_its_commands():
    out = _commands_with("""
        import importlib.util
        _real = importlib.util.find_spec
        importlib.util.find_spec = lambda n, *a, **k: None if n in (
            "topopyscale2.cli.snow_cli", "topopyscale2.cli.da_cli") else _real(n, *a, **k)
    """)
    assert out.returncode == 0, out.stderr
    names = set(eval(out.stdout))
    assert V1_COMMANDS <= names
    assert not names & {"snowmapper", "snow-page", "dashboard", "da"}
    assert {"forecast", "qc"} <= names


@requires("topopyscale2.cli.da_cli")
def test_present_but_broken_module_raises_instead_of_vanishing():
    out = _commands_with("""
        import importlib
        _real = importlib.import_module
        def _broken(n, *a, **k):
            if n == "topopyscale2.cli.da_cli":
                raise ImportError("simulated missing dependency")
            return _real(n, *a, **k)
        importlib.import_module = _broken
    """)
    assert out.returncode != 0
    assert "simulated missing dependency" in out.stderr


def test_run_works_without_the_dashboard_module(monkeypatch):
    from topopyscale2.cli import main

    monkeypatch.setitem(sys.modules, "topopyscale2.outputs.dashboard", None)
    update = main._live_dashboard_updater()
    assert update("any_dir", status="Running", step="x") is None


@pytest.mark.parametrize("name", sorted(V1_COMMANDS))
def test_v1_commands_live_in_main(name):
    """v1 commands must not be defined in an optional module."""
    import typer.main

    from topopyscale2.cli.main import app

    cmd = typer.main.get_command(app).commands[name]
    assert cmd.callback.__module__ == "topopyscale2.cli.main", cmd.callback.__module__


@requires("topopyscale2.cli.snow_cli", "topopyscale2.cli.forecast_cli")
def test_module_entry_point_has_the_optional_commands():
    """`python -m topopyscale2.cli.main` is how the operational scripts call TPS2.

    Regression: the __main__ guard once ran the app before the optional commands were
    registered, so `python -m ... snowmapper` said "No such command" (2026-09-30).
    """
    for cmd in (["snowmapper", "--help"], ["forecast", "run", "--help"], ["app", "run", "--help"]):
        out = subprocess.run([sys.executable, "-m", "topopyscale2.cli.main", *cmd],
                             capture_output=True, text=True)
        assert out.returncode == 0, f"{' '.join(cmd)}: {out.stderr[-500:]}"
