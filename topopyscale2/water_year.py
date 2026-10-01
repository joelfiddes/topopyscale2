"""Water-year calendar: one definition used by every module.

A water year starts on the 1st of ``start_month`` and is **labelled by the calendar year
it ends in**. With the default ``start_month = 9``, WY2027 runs 1 Sep 2026 – 31 Aug 2027.
Southern-hemisphere domains set their own month (e.g. 3 for 1 March); ``start_month = 1``
makes the water year the calendar year.

The month is configuration (``water_year_start_month`` in the config), never inferred
from the domain. Every climatology store records the month it was built with in
``attrs["water_year_start_month"]``; readers take the month from the store they read, so a
store and the data compared against it always share one calendar. Stores written before
the setting existed carry no attribute and were built on 1 October
(:data:`LEGACY_START_MONTH`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_START_MONTH = 9     # 1 September (northern hemisphere)
LEGACY_START_MONTH = 10     # stores built before the setting existed used 1 October
ATTR = "water_year_start_month"


def _check(start_month: int) -> int:
    m = int(start_month)
    if not 1 <= m <= 12:
        raise ValueError(f"water_year_start_month must be 1-12, got {start_month!r}")
    return m


def to_water_year(time, start_month: int = DEFAULT_START_MONTH) -> np.ndarray:
    """Water year (labelled by its end year) for each timestamp."""
    m = _check(start_month)
    ts = pd.DatetimeIndex(np.atleast_1d(time))
    years = ts.year.to_numpy()
    if m == 1:
        return years
    return np.where(ts.month.to_numpy() >= m, years + 1, years)


def water_year_start(wy: int, start_month: int = DEFAULT_START_MONTH) -> pd.Timestamp:
    """First day of water year ``wy``."""
    m = _check(start_month)
    return pd.Timestamp(year=int(wy) if m == 1 else int(wy) - 1, month=m, day=1)


def water_year_end(wy: int, start_month: int = DEFAULT_START_MONTH) -> pd.Timestamp:
    """Last day of water year ``wy`` (the day before the next one starts)."""
    return water_year_start(int(wy) + 1, start_month) - pd.Timedelta(days=1)


def water_year_bounds(wy: int, start_month: int = DEFAULT_START_MONTH) -> tuple[str, str]:
    """``(first_day, last_day)`` of water year ``wy`` as ISO dates."""
    return (water_year_start(wy, start_month).strftime("%Y-%m-%d"),
            water_year_end(wy, start_month).strftime("%Y-%m-%d"))


def to_dowy(time, start_month: int = DEFAULT_START_MONTH) -> np.ndarray:
    """Day of water year, 0-based (0 = first day). A 366th day is folded into 365."""
    ts = pd.DatetimeIndex(np.atleast_1d(time))
    wy = to_water_year(ts, start_month)
    starts = pd.DatetimeIndex([water_year_start(int(y), start_month) for y in wy])
    return np.clip((ts.normalize() - starts).days.to_numpy(), 0, 365)


def water_years(start, end, start_month: int = DEFAULT_START_MONTH) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split ``[start, end]`` into water-year windows, clipped to the range."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out = []
    for wy in range(int(to_water_year(s, start_month)[0]), int(to_water_year(e, start_month)[0]) + 1):
        a = max(water_year_start(wy, start_month), s)
        b = min(water_year_end(wy, start_month), e)
        if a <= b:
            out.append((a, b))
    return out


def start_month_of(obj) -> int:
    """The water-year start month a climatology/comparison store was built with.

    ``obj`` is an xarray object or an attrs mapping. Stores without the attribute predate
    the setting and were built on 1 October.
    """
    attrs = getattr(obj, "attrs", obj) or {}
    return _check(attrs.get(ATTR, LEGACY_START_MONTH))


def dowy_base(dowy_values) -> int:
    """Numbering base of a store's ``dowy`` coordinate: 1 (built climatologies) or 0.

    ``build_climatology`` writes 1-based labels (1 = first day); older synthetic stores
    wrote 0-based ones. Reading the base from the store keeps every consumer on the
    same day as the store, whichever it is.
    """
    return 0 if int(np.min(dowy_values)) == 0 else 1


def dowy_labels(time, dowy_values, start_month: int) -> np.ndarray:
    """``dowy`` labels for ``time`` in the numbering a store with ``dowy_values`` uses."""
    return to_dowy(time, start_month) + dowy_base(dowy_values)


def dowy_to_dates(wy: int, dowy_values, start_month: int) -> list[str]:
    """ISO dates for a store's ``dowy`` labels within water year ``wy``."""
    start, base = water_year_start(wy, start_month), dowy_base(dowy_values)
    return [(start + pd.Timedelta(days=int(d) - base)).strftime("%Y-%m-%d") for d in dowy_values]


def start_month_from_sim(sim_dir) -> int:
    """``water_year_start_month`` from ``<sim_dir>/config.yaml``, else the default.

    Reads the one key without validating the whole config, so commands that only take
    a simulation directory (e.g. ``climatology-build``) use the domain's own calendar.
    """
    from pathlib import Path

    import yaml

    cfg = Path(sim_dir) / "config.yaml"
    if cfg.exists():
        try:
            raw = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            raw = {}
        if raw.get(ATTR) is not None:
            return _check(raw[ATTR])
    return DEFAULT_START_MONTH
