"""Every config-driven Downscaler parameter is passed by every CLI construction.

Regression: `downscaling.lapse_rate` was accepted by the schema but never passed to
`Downscaler`, so simple mode always used 6.5 K/km whatever the config said (2026-10-01).
"""

import ast
from pathlib import Path

import pytest

CLI = Path(__file__).resolve().parents[2] / "topopyscale2" / "cli"
CONFIG_DRIVEN = {"mode", "precip_gradient", "phase_method", "t_snow", "t_rain", "t_air_max", "lapse_rate"}


def _downscaler_calls():
    for path in sorted(CLI.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Downscaler":
                yield f"{path.name}:{node.lineno}", {k.arg for k in node.keywords}


CALLS = list(_downscaler_calls())


def test_the_cli_constructs_downscalers():
    assert len(CALLS) >= 2, "found no Downscaler constructions; did the CLI move?"


@pytest.mark.parametrize("where,kwargs", CALLS, ids=[w for w, _ in CALLS])
def test_every_config_driven_parameter_is_passed(where, kwargs):
    missing = CONFIG_DRIVEN - kwargs
    assert not missing, f"{where} does not pass {sorted(missing)} from the config"
