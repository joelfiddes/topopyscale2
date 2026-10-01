"""Helpers for features that a release may not ship.

A public snapshot contains only the downscaling core; forecast, IFS, dashboards and
the rest are absent there. Code that reaches for them uses these helpers so that an
absent feature gives a clear message, while a feature that IS installed but has a
broken import (a missing third-party dependency, a bug) still raises as normal.
"""

from __future__ import annotations


def module_absent(err: ModuleNotFoundError, module: str) -> bool:
    """True if ``err`` means ``module`` itself (or a parent package) is not installed."""
    return err.name is not None and (module == err.name or module.startswith(err.name + "."))


def not_in_release(feature: str, err: ModuleNotFoundError, module: str) -> Exception:
    """The exception to raise when ``feature`` is unavailable because ``module`` is absent.

    Returns ``err`` unchanged when the failure is something else, so callers can
    always ``raise not_in_release(...) from err``.
    """
    if module_absent(err, module):
        return RuntimeError(f"{feature} is not included in this release of TopoPyScale.")
    return err
