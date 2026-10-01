#!/usr/bin/env python
"""Backend comparison test script.

Downloads 3 days of ERA5 data from all 3 backends (Google, CDS, S3 Zarr)
using the same bbox (Fan Mountains, Tajikistan) and compares results
value-by-value.

Each backend runs in a subprocess to isolate crashes (e.g. segfaults from
netCDF4+dask parallel merge on macOS).

Usage:
    cd <repo root>
    python tests/test_backends_comparison.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BBOX = (68.0, 39.0, 69.0, 39.8)  # Fan Mountains, Tajikistan (W, S, E, N)
START_DATE = "2020-01-01"
END_DATE = "2020-01-03"
PRESSURE_LEVELS = [500, 700, 850, 1000]
TIME_RESOLUTION = "3H"
OUTPUT_FORMAT = "netcdf"
BASE_OUTPUT = Path("./test_output")

# The s3zarr backend needs a private store and its credentials file; both come from the
# environment so no machine path or bucket is committed. Unset = s3zarr is skipped.
S3_ENV_FILE = Path(os.environ["TPS2_ERA5_S3_ENV"]) if os.environ.get("TPS2_ERA5_S3_ENV") else None
S3_ZARR_URL = os.environ.get("TPS2_ERA5_S3_ZARR_URL", "")

BACKENDS = ["google", "cds", "openmeteo"] + (["s3zarr"] if S3_ZARR_URL else [])

# Google ARCO-ERA5 doesn't have relative_humidity; compute_rh derives it from q+t
GOOGLE_PLEV_VARS = [
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_s3_env(env_file: Path) -> dict[str, str]:
    """Parse KEY=VALUE pairs from env file, return as dict."""
    env = {}
    if not env_file.exists():
        print(f"  WARNING: S3 env file not found: {env_file}")
        return env
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            env[key.strip()] = value.strip()
    return env


def output_dir_for(backend: str) -> Path:
    return BASE_OUTPUT / backend


def find_daily_files(backend: str, prefix: str) -> list[Path]:
    """Find daily netCDF files for a backend (dSURF_*.nc or dPLEV_*.nc)."""
    daily_dir = output_dir_for(backend) / "daily"
    if not daily_dir.exists():
        return []
    return sorted(daily_dir.glob(f"{prefix}_*.nc"))


def report_dataset(label: str, ds: xr.Dataset) -> None:
    """Print summary statistics for a dataset."""
    print(f"\n  {label}")
    print(f"    Dims:  {dict(ds.sizes)}")
    print(f"    Vars:  {list(ds.data_vars)}")
    if "time" in ds.dims:
        times = ds.time.values
        print(f"    Time steps: {len(times)}, "
              f"first={str(times[0])[:19]}, last={str(times[-1])[:19]}")
    if "level" in ds.dims:
        print(f"    Levels: {ds.level.values.tolist()}")
    for var in sorted(ds.data_vars):
        vals = ds[var].values
        print(f"    {var:>8s}: shape={vals.shape}  "
              f"min={np.nanmin(vals):12.4f}  "
              f"max={np.nanmax(vals):12.4f}  "
              f"mean={np.nanmean(vals):12.4f}")


# ---------------------------------------------------------------------------
# Phase 1: Download (each backend in its own subprocess)
# ---------------------------------------------------------------------------
DOWNLOAD_WORKER = textwrap.dedent("""\
    import json, sys, time
    cfg = json.loads(sys.argv[1])

    from topopyscale2.inputs.nwp_downloader import ERA5Loader

    kwargs = dict(
        backend=cfg["backend"],
        bbox=tuple(cfg["bbox"]),
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        pressure_levels=cfg["pressure_levels"],
        output_format=cfg["output_format"],
        output_dir=cfg["output_dir"],
        time_resolution=cfg["time_resolution"],
        max_workers=1,
        compute_rh=True,
        skip_existing=True,
    )
    if cfg.get("backend_kwargs"):
        kwargs["backend_kwargs"] = cfg["backend_kwargs"]

    loader = ERA5Loader(**kwargs)
    t0 = time.perf_counter()
    loader.download()
    elapsed = time.perf_counter() - t0
    print(f"__ELAPSED__={elapsed:.2f}")
""")


def download_backend(backend: str, extra_env: dict[str, str]) -> tuple[float, int]:
    """Run a single backend download in a subprocess.

    Returns (elapsed_seconds, return_code).  Elapsed is always measured from
    the parent side so it's available even if the subprocess crashes.
    """
    cfg = dict(
        backend=backend,
        bbox=list(BBOX),
        start_date=START_DATE,
        end_date=END_DATE,
        pressure_levels=PRESSURE_LEVELS,
        output_format=OUTPUT_FORMAT,
        output_dir=str(output_dir_for(backend)),
        time_resolution=TIME_RESOLUTION,
    )

    if backend == "google":
        cfg["backend_kwargs"] = {"plev_vars": GOOGLE_PLEV_VARS}
    elif backend == "s3zarr":
        cfg["backend_kwargs"] = {"zarr_url": S3_ZARR_URL}
    elif backend == "openmeteo":
        cfg["backend_kwargs"] = {
            "start_date": START_DATE,
            "end_date": END_DATE,
        }

    env = {**os.environ, **extra_env}
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", DOWNLOAD_WORKER, json.dumps(cfg)],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.perf_counter() - t0

    # Print subprocess output
    if proc.stdout:
        for line in proc.stdout.splitlines():
            if not line.startswith("__ELAPSED__"):
                print(f"  {line}")
    if proc.stderr:
        for line in proc.stderr.splitlines():
            print(f"  {line}")

    return elapsed, proc.returncode


def download_all() -> dict[str, float]:
    """Download from each backend in subprocesses."""
    s3_env = load_s3_env(S3_ENV_FILE) if S3_ENV_FILE else {}
    if s3_env:
        print(f"  Loaded S3 credentials from {S3_ENV_FILE}")

    timings: dict[str, float] = {}

    for backend in BACKENDS:
        out_dir = output_dir_for(backend)
        print(f"\n{'='*60}")
        print(f"Backend: {backend}")
        print(f"Output:  {out_dir}")
        print(f"{'='*60}")

        try:
            elapsed, rc = download_backend(backend, s3_env)
        except subprocess.TimeoutExpired:
            print("  TIMEOUT (600s)")
            timings[backend] = float("nan")
            continue

        if rc == 0:
            timings[backend] = elapsed
            print(f"\n  Completed in {elapsed:.1f}s")
        else:
            # Check if daily files were written despite crash (e.g. segfault in finalize)
            n_files = len(find_daily_files(backend, "dSURF"))
            if n_files > 0:
                print(f"\n  Process exited with code {rc} (likely crash in finalize), "
                      f"but {n_files} daily files written")
                timings[backend] = elapsed
            else:
                print(f"\n  FAILED (exit code {rc})")
                timings[backend] = float("nan")

    return timings


# ---------------------------------------------------------------------------
# Phase 2: Report
# ---------------------------------------------------------------------------
def report_all() -> dict[str, dict[str, xr.Dataset]]:
    """Read back daily files and print per-backend summaries.

    Returns {backend: {"surf": merged_ds, "plev": merged_ds}} for backends
    that have data.
    """
    datasets: dict[str, dict[str, xr.Dataset]] = {}

    for backend in BACKENDS:
        print(f"\n{'='*60}")
        print(f"Report: {backend}")
        print(f"{'='*60}")

        surf_files = find_daily_files(backend, "dSURF")
        plev_files = find_daily_files(backend, "dPLEV")

        if not surf_files and not plev_files:
            print("  No output files found — skipping.")
            continue

        entry: dict[str, xr.Dataset] = {}

        if surf_files:
            ds_surf = xr.open_mfdataset(
                [str(f) for f in surf_files], combine="by_coords",
                parallel=False,
            ).load()
            report_dataset(f"SURF ({len(surf_files)} files)", ds_surf)
            entry["surf"] = ds_surf

        if plev_files:
            ds_plev = xr.open_mfdataset(
                [str(f) for f in plev_files], combine="by_coords",
                parallel=False,
            ).load()
            report_dataset(f"PLEV ({len(plev_files)} files)", ds_plev)
            entry["plev"] = ds_plev

        datasets[backend] = entry

    return datasets


# ---------------------------------------------------------------------------
# Phase 3: Timing summary
# ---------------------------------------------------------------------------
def print_timing_summary(timings: dict[str, float]) -> None:
    n_days = 3  # 2020-01-01 to 2020-01-03 inclusive

    print(f"\n{'='*60}")
    print("Timing Summary")
    print(f"{'='*60}")
    print(f"  {'Backend':<12s} {'Total (s)':>10s} {'Per day (s)':>12s}")
    print(f"  {'-'*12} {'-'*10} {'-'*12}")
    for backend in BACKENDS:
        t = timings.get(backend, float("nan"))
        if np.isnan(t):
            print(f"  {backend:<12s} {'FAILED':>10s} {'':>12s}")
        else:
            print(f"  {backend:<12s} {t:10.1f} {t / n_days:12.1f}")


# ---------------------------------------------------------------------------
# Phase 4: Cross-comparison
# ---------------------------------------------------------------------------
def cross_compare(datasets: dict[str, dict[str, xr.Dataset]]) -> None:
    """Compare every pair of backends variable-by-variable."""
    available = [b for b in BACKENDS if b in datasets]

    if len(available) < 2:
        print("\n  Need at least 2 backends with data to compare — skipping.")
        return

    print(f"\n{'='*60}")
    print("Cross-Backend Comparison")
    print(f"{'='*60}")

    for kind in ("surf", "plev"):
        print(f"\n--- {kind.upper()} variables ---")
        print(f"  {'Pair':<22s} {'Variable':>10s} "
              f"{'MaxAbsDiff':>12s} {'MeanAbsDiff':>12s}")
        print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12}")

        for b1, b2 in combinations(available, 2):
            ds1 = datasets[b1].get(kind)
            ds2 = datasets[b2].get(kind)

            if ds1 is None or ds2 is None:
                print(f"  {b1+' vs '+b2:<22s}  ** missing dataset **")
                continue

            vars1 = set(ds1.data_vars)
            vars2 = set(ds2.data_vars)
            shared = sorted(vars1 & vars2)
            only1 = vars1 - vars2
            only2 = vars2 - vars1

            if only1:
                print(f"  {b1+' vs '+b2:<22s}  only in {b1}: {only1}")
            if only2:
                print(f"  {b1+' vs '+b2:<22s}  only in {b2}: {only2}")

            for var in shared:
                a1 = ds1[var]
                a2 = ds2[var]

                # Check dimension compatibility (names, not order)
                if set(a1.dims) != set(a2.dims):
                    print(f"  {b1+' vs '+b2:<22s} {var:>10s}  "
                          f"DIM MISMATCH: {dict(a1.sizes)} vs {dict(a2.sizes)}")
                    continue

                # Align on shared coordinates (handles minor grid differences)
                try:
                    a1_aligned, a2_aligned = xr.align(a1, a2, join="inner")
                except Exception as exc:
                    print(f"  {b1+' vs '+b2:<22s} {var:>10s}  "
                          f"ALIGN ERROR: {exc}")
                    continue

                if a1_aligned.size == 0:
                    print(f"  {b1+' vs '+b2:<22s} {var:>10s}  "
                          f"NO OVERLAPPING COORDS")
                    continue

                # Use xarray diff (handles dimension order differences)
                abs_diff = abs(a1_aligned - a2_aligned)
                max_diff = float(abs_diff.max().values)
                mean_diff = float(abs_diff.mean().values)

                print(f"  {b1+' vs '+b2:<22s} {var:>10s} "
                      f"{max_diff:12.6f} {mean_diff:12.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Backend Comparison Test")
    print(f"  BBox:       {BBOX}")
    print(f"  Dates:      {START_DATE} to {END_DATE}")
    print(f"  Levels:     {PRESSURE_LEVELS}")
    print(f"  Resolution: {TIME_RESOLUTION}")
    print(f"  Format:     {OUTPUT_FORMAT}")
    print(f"  Output:     {BASE_OUTPUT.resolve()}")

    # Phase 1: Download
    timings = download_all()

    # Phase 2: Report
    datasets = report_all()

    # Phase 3: Timing
    print_timing_summary(timings)

    # Phase 4: Cross-compare
    cross_compare(datasets)

    print(f"\nFiles left in {BASE_OUTPUT.resolve()} for manual inspection.")


if __name__ == "__main__":
    main()
