"""Markers for tests of features that a core-only public release does not ship.

The private repository has every module, so nothing here skips there. In an exported
snapshot (scripts/release/export_public.py) the tests for unshipped features skip with
a reason naming the module, instead of failing on a command or writer that is absent.
"""

import importlib.util

import pytest


def shipped(module: str) -> bool:
    """True if ``module`` is part of this installation."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package is absent
        return False


def requires(*modules: str):
    """Skip the test (or class) unless every module in ``modules`` ships."""
    missing = [m for m in modules if not shipped(m)]
    return pytest.mark.skipif(bool(missing), reason=f"not in this release: {', '.join(missing)}")
