"""Physical range checks for forcing: stop a unit mistake before it reaches a file or a model.

Three unit bugs reached files in 2026 without any test noticing, because each produced output
that looked plausible: relative humidity written as a fraction where % was read (100x too dry),
and precipitation multiplied by 1000 (millimetres treated as metres). A check on physical
bounds catches that whole class at once, whichever code introduced it.

Bounds are deliberately wide: real extremes over every operational domain (13 forcing stores,
Alps to Himalaya, measured 2026-10-06) sit far inside them, and a 100x or 1000x error lands far
outside. NaN is ignored: some domains carry empty units whose forcing is NaN throughout, which
is a separate question from wrong units.

Set ``TPS2_FORCING_CHECKS=warn`` to log a violation instead of raising (an operational escape
hatch); the default is to raise.
"""

from __future__ import annotations

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# TPS2 forcing convention (what `tps2 run` writes): name -> (min, max, units).
# measured 2026-10-06 across all operational stores: T 235-320 K, RH 0-1.0, q 0-0.025,
# SWdir 0-1361, SWdif 0-620, LW 84-508, wind 0-24, p 37700-104200, P <= 32 mm/h.
FORCING_BOUNDS = {
    "temperature": (173.15, 333.15, "K"),
    "humidity_relative": (0.0, 1.05, "1 (fraction)"),
    "humidity_specific": (0.0, 0.05, "kg kg-1"),
    "shortwave_direct": (-1.0, 2000.0, "W m-2"),
    "shortwave_diffuse": (-1.0, 1000.0, "W m-2"),
    "longwave": (40.0, 700.0, "W m-2"),
    "wind_speed": (0.0, 75.0, "m s-1"),
    "pressure": (25000.0, 110000.0, "Pa"),
}
# Precipitation-type amounts per time step [kg m-2 = mm]: bound scales with the step.
PRECIP_VARS = ("precipitation", "rainfall", "snowfall")
PRECIP_MAX_MM_PER_HOUR = 300.0

# The same physics in the units FSM reads (after its input conversion).
FSM_BOUNDS = {
    "SW": (-1.0, 3000.0, "W m-2"),
    "LW": (40.0, 700.0, "W m-2"),
    "Sf": (-1e-9, PRECIP_MAX_MM_PER_HOUR / 3600.0, "kg m-2 s-1"),
    "Rf": (-1e-9, PRECIP_MAX_MM_PER_HOUR / 3600.0, "kg m-2 s-1"),
    "Ta": (173.15, 333.15, "K"),
    "RH": (0.0, 105.0, "%"),
    "Ua": (0.0, 75.0, "m s-1"),
    "Ps": (25000.0, 110000.0, "Pa"),
}


class ForcingRangeError(ValueError):
    """Forcing holds values no physical atmosphere produces: almost always a unit mistake."""


def _hint(name: str, lo: float, hi: float, vmin: float, vmax: float) -> str:
    if name in ("humidity_relative",) and vmax > 1.5:
        return "looks like percent, where a 0-1 fraction is expected"
    if name == "RH" and 0 < vmax <= 1.05:
        return "looks like a 0-1 fraction, where percent is expected"
    if name == "RH" and vmax > 105:
        return "looks like percent multiplied by 100 again"
    if name in ("temperature", "Ta") and vmax < 80:
        return "looks like degrees Celsius, where kelvin is expected"
    if name in ("pressure", "Ps") and vmax < 2000:
        return "looks like hPa, where Pa is expected"
    if name in ("precipitation", "rainfall", "snowfall", "Sf", "Rf") and vmax > hi:
        return ("beyond any rainfall ever recorded: a unit factor is likely "
                "(x1000 metres vs millimetres, or an amount per step passed as a rate)")
    if vmax > hi * 100:
        return f"about {vmax / max(hi, 1e-30):.0f}x the bound: a unit factor (x100, x1000) is likely"
    return "outside the physical range"


def _violations(arrays: dict, bounds: dict) -> list[str]:
    out = []
    for name, values in arrays.items():
        if name not in bounds:
            continue
        lo, hi, units = bounds[name]
        a = np.asarray(values, dtype=float)
        if a.size == 0 or not np.isfinite(a).any():
            continue
        vmin, vmax = float(np.nanmin(a)), float(np.nanmax(a))
        if name == "RH" and vmax <= 1.05 and np.isfinite(a).sum() >= 24:
            # Inside 0-105 %, but a day or more that never exceeds ~1 % humidity does not
            # happen: this is the 0-1 fraction where % is expected (the 2026 FSM writer bug).
            out.append(f"{name}: {vmin:.4g} .. {vmax:.4g} {units} over {np.isfinite(a).sum()} values; "
                       f"{_hint(name, lo, hi, vmin, vmax)}")
            continue
        if vmin < lo or vmax > hi:
            out.append(f"{name}: {vmin:.4g} .. {vmax:.4g} outside [{lo:g}, {hi:g}] {units}; "
                       f"{_hint(name, lo, hi, vmin, vmax)}")
    return out


def _report(problems: list[str], where: str) -> None:
    if not problems:
        return
    msg = f"implausible forcing in {where}:\n  " + "\n  ".join(problems)
    if os.environ.get("TPS2_FORCING_CHECKS", "").strip().lower() == "warn":
        log.warning("%s\n(TPS2_FORCING_CHECKS=warn: continuing)", msg)
        return
    raise ForcingRangeError(msg + "\n(set TPS2_FORCING_CHECKS=warn to continue anyway)")


def step_hours(ds) -> float:
    """Time step of a dataset in hours (1 when it has fewer than two steps)."""
    if "time" not in getattr(ds, "dims", {}) or ds.sizes["time"] < 2:
        return 1.0
    t = np.asarray(ds["time"].values[:2]).astype("datetime64[s]")
    return float((t[1] - t[0]) / np.timedelta64(1, "h")) or 1.0


def check_forcing(ds, where: str = "forcing") -> None:
    """Raise ForcingRangeError if any TPS2 forcing variable in ``ds`` is physically implausible."""
    arrays = {v: ds[v].values for v in FORCING_BOUNDS if v in ds}
    bounds = dict(FORCING_BOUNDS)
    max_step = PRECIP_MAX_MM_PER_HOUR * max(step_hours(ds), 1.0)
    for v in PRECIP_VARS:
        if v in ds:
            arrays[v] = ds[v].values
            bounds[v] = (-1e-6, max_step, "kg m-2 per step")
    _report(_violations(arrays, bounds), where)


def check_fsm_inputs(where: str = "FSM input", **arrays) -> None:
    """Range-check arrays already converted to FSM's units (SW, LW, Sf, Rf, Ta, RH, Ua, Ps)."""
    _report(_violations(arrays, FSM_BOUNDS), where)
