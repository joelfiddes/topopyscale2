"""Verify ZarrWriter.finalize() handles overlapping time ranges.

When a day is re-downloaded (e.g. incremental cron runs), the existing
ERA5.zarr and the new daily stores may share timestamps. combine="by_coords"
raised "not monotonic" on any overlap; combine="nested" + dedup handles it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.inputs.nwp_downloader.writer import ZarrWriter


def _make_daily_ds(date: str, n_hours: int = 24, value: float = 1.0) -> xr.Dataset:
    times = pd.date_range(date, periods=n_hours, freq="h")
    data = np.full((n_hours, 2, 2), value, dtype=np.float32)
    return xr.Dataset(
        {"t2m": (["time", "latitude", "longitude"], data)},
        coords={
            "time": times,
            "latitude": [40.0, 41.0],
            "longitude": [60.0, 61.0],
        },
    )


def test_finalize_handles_overlapping_days(tmp_path: Path):
    writer = ZarrWriter(output_dir=tmp_path, merge=True, cleanup_daily=False)

    # Existing ERA5.zarr covers Jan 1-2
    ds1 = xr.concat(
        [_make_daily_ds("2024-01-01", value=1.0), _make_daily_ds("2024-01-02", value=1.0)],
        dim="time",
    )
    ds1.to_zarr(writer.zarr_path, mode="w", zarr_format=3)

    # New daily stores: Jan 2 (OVERLAP, different values) + Jan 3
    writer.daily_dir.mkdir(parents=True, exist_ok=True)
    _make_daily_ds("2024-01-02", value=2.0).to_zarr(
        writer.daily_dir / "day_20240102.zarr", mode="w", zarr_format=3
    )
    _make_daily_ds("2024-01-03", value=2.0).to_zarr(
        writer.daily_dir / "day_20240103.zarr", mode="w", zarr_format=3
    )

    # Should not raise "not monotonic"
    writer.finalize()

    result = xr.open_zarr(writer.zarr_path)
    # 3 days × 24 hours, no duplicate timestamps
    assert result.sizes["time"] == 72
    # Times are sorted and unique
    times = result["time"].values
    assert np.all(np.diff(times).astype("timedelta64[s]").astype(int) > 0)
    # First occurrence kept (existing ERA5 store wins for Jan 2)
    jan2_mask = pd.DatetimeIndex(result["time"].values).date == pd.Timestamp("2024-01-02").date()
    assert float(result["t2m"].values[jan2_mask].mean()) == 1.0


def test_finalize_no_overlap_still_works(tmp_path: Path):
    writer = ZarrWriter(output_dir=tmp_path, merge=True, cleanup_daily=False)

    # Existing ERA5.zarr covers Jan 1
    _make_daily_ds("2024-01-01", value=1.0).to_zarr(
        writer.zarr_path, mode="w", zarr_format=3
    )

    # New daily stores: Jan 2, Jan 3 (no overlap)
    writer.daily_dir.mkdir(parents=True, exist_ok=True)
    _make_daily_ds("2024-01-02", value=2.0).to_zarr(
        writer.daily_dir / "day_20240102.zarr", mode="w", zarr_format=3
    )
    _make_daily_ds("2024-01-03", value=2.0).to_zarr(
        writer.daily_dir / "day_20240103.zarr", mode="w", zarr_format=3
    )

    writer.finalize()

    result = xr.open_zarr(writer.zarr_path)
    assert result.sizes["time"] == 72
    times = result["time"].values
    assert np.all(np.diff(times).astype("timedelta64[s]").astype(int) > 0)
