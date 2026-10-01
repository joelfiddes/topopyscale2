"""Input quality control for ERA5 (and other NWP) forcing data.

Detects and repairs corrupted values before they enter the downscaling
pipeline.  The primary target is single-timestep spikes in s3zarr ERA5
stores where t2m (or other variables) drop/spike by >30 K for one
timestep at specific grid cells, causing downstream blowup in the
longwave correction via the (t_unit/t_source)^4 ratio.

Usage
-----
Called automatically by :func:`topopyscale2.inputs.units.convert_era5`
when ``run_qc=True`` (default).  Can also be called directly::

    from topopyscale2.inputs.qc import qc_era5_surface
    ds_clean = qc_era5_surface(ds_raw)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# Default QC thresholds: max plausible single-step change per variable.
# Values exceeding these are flagged as corrupted and interpolated.
DEFAULT_THRESHOLDS = {
    "t2m": 20.0,   # K — max 20 K jump in one timestep
    "d2m": 20.0,   # K
    "sp": 5000.0,   # Pa — max 50 hPa jump
    "strd": 200.0,  # J/m² per second (pre-conversion) — proxy for ~720 kJ/m²/hr
}


def qc_era5_surface(
    ds: xr.Dataset,
    thresholds: Optional[dict[str, float]] = None,
) -> xr.Dataset:
    """Detect and repair single-timestep spikes in ERA5 surface data.

    Some ERA5 archives (notably s3zarr stores) contain corrupted values
    where a variable (typically t2m) drops or spikes by >30 K for a
    single timestep at specific grid cells.  These cause downstream
    blowup in longwave correction via the (t_unit/t_source)^4 ratio.

    Detection
    ---------
    For each grid cell, compute the forward difference along time.
    A "spike" is a timestep where:

    1. The absolute jump IN exceeds the threshold, AND
    2. The absolute jump OUT also exceeds it, AND
    3. The two jumps have opposite sign (it goes out and comes back).

    This targets single isolated bad values while ignoring legitimate
    weather events (cold fronts, etc.) that change monotonically.

    Repair
    ------
    Flagged values are replaced with linear interpolation from their
    temporal neighbours: ``0.5 * (arr[t-1] + arr[t+1])``.

    Parameters
    ----------
    ds : xr.Dataset
        ERA5 surface dataset (before unit conversion).
    thresholds : dict, optional
        Per-variable max plausible single-step change.  Defaults to
        :data:`DEFAULT_THRESHOLDS`.

    Returns
    -------
    xr.Dataset
        Cleaned dataset with corrupted values interpolated.
    """
    if "time" not in ds.dims or ds.sizes["time"] < 3:
        return ds

    thresholds = thresholds or DEFAULT_THRESHOLDS
    ds = ds.copy()
    total_fixed = 0

    for var, thresh in thresholds.items():
        if var not in ds:
            continue

        arr = ds[var].values  # shape: (time, ...) — operates in-place on copy
        if arr.ndim < 1:
            continue

        # Forward difference along time axis (axis=0)
        diff = np.diff(arr, axis=0)  # shape: (time-1, ...)

        # A spike at timestep t means:
        #   |arr[t] - arr[t-1]| > thresh  AND  |arr[t+1] - arr[t]| > thresh
        #   AND the two jumps have opposite sign (it goes out and comes back)
        # diff[t-1] = arr[t] - arr[t-1], diff[t] = arr[t+1] - arr[t]
        abs_in = np.abs(diff[:-1])   # jump into timestep t (t=1..T-2)
        abs_out = np.abs(diff[1:])   # jump out of timestep t
        sign_in = np.sign(diff[:-1])
        sign_out = np.sign(diff[1:])

        # Spike: large jump in, large jump out, opposite directions
        spike_mask = (abs_in > thresh) & (abs_out > thresh) & (sign_in != sign_out)

        n_spikes = int(np.sum(spike_mask))
        if n_spikes == 0:
            continue

        # Build full-time mask (False for t=0 and t=T-1, spikes for t=1..T-2)
        full_mask = np.zeros(arr.shape, dtype=bool)
        full_mask[1:-1] = spike_mask

        # Replace flagged values with average of neighbours
        prev = np.roll(arr, 1, axis=0)
        nxt = np.roll(arr, -1, axis=0)
        interpolated = 0.5 * (prev + nxt)

        arr[full_mask] = interpolated[full_mask]
        ds[var].values = arr

        total_fixed += n_spikes
        logger.warning(
            "QC: %s — fixed %d single-timestep spike(s) (threshold=%.1f)",
            var, n_spikes, thresh,
        )

    if total_fixed > 0:
        logger.info("QC: total %d corrupted values interpolated", total_fixed)

    return ds
