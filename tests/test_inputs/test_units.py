"""Tests for topopyscale2.inputs.units — unit conversion and input QC."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.inputs.qc import qc_era5_surface
from topopyscale2.inputs.units import (
    convert_era5,
    convert_era5_surface,
    detect_time_step,
)


def _make_surface_ds(
    n_time: int = 24,
    n_lat: int = 3,
    n_lon: int = 3,
    freq: str = "3h",
) -> xr.Dataset:
    """Create a minimal ERA5 surface dataset for testing."""
    times = pd.date_range("2020-06-01", periods=n_time, freq=freq)
    lats = np.linspace(38.0, 39.0, n_lat)
    lons = np.linspace(68.0, 69.0, n_lon)

    rng = np.random.default_rng(42)
    shape = (n_time, n_lat, n_lon)

    ds = xr.Dataset(
        {
            "t2m": (["time", "latitude", "longitude"], 290.0 + rng.normal(0, 2, shape)),
            "d2m": (["time", "latitude", "longitude"], 280.0 + rng.normal(0, 2, shape)),
            "sp": (["time", "latitude", "longitude"], 80000.0 + rng.normal(0, 200, shape)),
            "ssrd": (["time", "latitude", "longitude"], rng.uniform(0, 2e6, shape)),
            "strd": (["time", "latitude", "longitude"], rng.uniform(8e5, 1.2e6, shape)),
            "tp": (["time", "latitude", "longitude"], rng.uniform(0, 0.005, shape)),
        },
        coords={
            "time": times,
            "latitude": lats,
            "longitude": lons,
        },
    )
    return ds


class TestQCEra5Surface:
    """Tests for qc_era5_surface spike detection and repair."""

    def test_clean_data_unchanged(self):
        """Clean data should pass through unmodified."""
        ds = _make_surface_ds()
        original = ds["t2m"].values.copy()
        ds_qc = qc_era5_surface(ds)
        np.testing.assert_array_equal(ds_qc["t2m"].values, original)

    def test_single_spike_detected_and_fixed(self):
        """A single-timestep temperature spike should be interpolated."""
        ds = _make_surface_ds(n_time=10)

        # Inject a spike at t=5, lat=1, lon=1: drop 40K for one timestep
        original_val = ds["t2m"].values[5, 1, 1]
        ds["t2m"].values[5, 1, 1] = original_val - 60.0  # -60K spike

        ds_qc = qc_era5_surface(ds)

        # The spike should be replaced with interpolation of neighbours
        expected = 0.5 * (ds["t2m"].values[4, 1, 1] + ds["t2m"].values[6, 1, 1])
        assert abs(ds_qc["t2m"].values[5, 1, 1] - expected) < 1e-10

    def test_spike_at_different_gridcells(self):
        """Spikes at different grid cells in different timesteps."""
        ds = _make_surface_ds(n_time=20)

        # Spike 1: t=3, cell (0,0)
        ds["t2m"].values[3, 0, 0] -= 50.0
        # Spike 2: t=10, cell (2,2)
        ds["t2m"].values[10, 2, 2] += 45.0

        ds_qc = qc_era5_surface(ds)

        # Both should be fixed
        expected_1 = 0.5 * (ds["t2m"].values[2, 0, 0] + ds["t2m"].values[4, 0, 0])
        expected_2 = 0.5 * (ds["t2m"].values[9, 2, 2] + ds["t2m"].values[11, 2, 2])
        assert abs(ds_qc["t2m"].values[3, 0, 0] - expected_1) < 1e-10
        assert abs(ds_qc["t2m"].values[10, 2, 2] - expected_2) < 1e-10

    def test_gradual_change_not_flagged(self):
        """A sustained temperature change (e.g. cold front) should NOT be flagged."""
        ds = _make_surface_ds(n_time=20)

        # Insert a ramp: temperature drops 5K per step for 3 steps
        for i in range(10, 13):
            ds["t2m"].values[i, 1, 1] -= 5.0 * (i - 9)

        original = ds["t2m"].values.copy()
        ds_qc = qc_era5_surface(ds)

        # Should be unchanged — the change is gradual, not a spike
        np.testing.assert_array_equal(ds_qc["t2m"].values, original)

    def test_boundary_timesteps_not_modified(self):
        """Spikes at t=0 or t=T-1 can't be detected (no two neighbours)."""
        ds = _make_surface_ds(n_time=10)

        # Spike at first timestep — cannot be detected
        ds["t2m"].values[0, 1, 1] -= 80.0
        original = ds["t2m"].values.copy()

        ds_qc = qc_era5_surface(ds)
        assert ds_qc["t2m"].values[0, 1, 1] == original[0, 1, 1]

    def test_pressure_spike_detected(self):
        """A spike in surface pressure should also be detected."""
        ds = _make_surface_ds(n_time=10)

        ds["sp"].values[5, 1, 1] -= 8000.0  # -80 hPa spike

        ds_qc = qc_era5_surface(ds)

        expected = 0.5 * (ds["sp"].values[4, 1, 1] + ds["sp"].values[6, 1, 1])
        assert abs(ds_qc["sp"].values[5, 1, 1] - expected) < 1e-10

    def test_custom_thresholds(self):
        """Custom thresholds should override defaults."""
        ds = _make_surface_ds(n_time=10)

        # Insert a 25K spike — above default 20K threshold
        ds["t2m"].values[5, 0, 0] -= 25.0

        # With stricter threshold (10K), it should be caught
        ds_qc_strict = qc_era5_surface(ds, thresholds={"t2m": 10.0})
        expected = 0.5 * (ds["t2m"].values[4, 0, 0] + ds["t2m"].values[6, 0, 0])
        assert abs(ds_qc_strict["t2m"].values[5, 0, 0] - expected) < 1e-10

        # With relaxed threshold (30K), it should NOT be caught
        ds2 = _make_surface_ds(n_time=10)
        ds2["t2m"].values[5, 0, 0] -= 25.0
        original = ds2["t2m"].values[5, 0, 0]
        ds_qc_relaxed = qc_era5_surface(ds2, thresholds={"t2m": 30.0})
        assert ds_qc_relaxed["t2m"].values[5, 0, 0] == original

    def test_short_timeseries_passthrough(self):
        """Datasets with < 3 timesteps should pass through unchanged."""
        ds = _make_surface_ds(n_time=2)
        original = ds["t2m"].values.copy()
        ds_qc = qc_era5_surface(ds)
        np.testing.assert_array_equal(ds_qc["t2m"].values, original)

    def test_multiple_variables_fixed(self):
        """Spikes in multiple variables should all be fixed."""
        ds = _make_surface_ds(n_time=10)

        ds["t2m"].values[5, 1, 1] -= 40.0
        ds["d2m"].values[7, 0, 2] += 35.0

        ds_qc = qc_era5_surface(ds)

        # Both should be interpolated
        exp_t2m = 0.5 * (ds["t2m"].values[4, 1, 1] + ds["t2m"].values[6, 1, 1])
        exp_d2m = 0.5 * (ds["d2m"].values[6, 0, 2] + ds["d2m"].values[8, 0, 2])
        assert abs(ds_qc["t2m"].values[5, 1, 1] - exp_t2m) < 1e-10
        assert abs(ds_qc["d2m"].values[7, 0, 2] - exp_d2m) < 1e-10


class TestConvertEra5Surface:
    """Tests for convert_era5_surface unit conversion."""

    def test_radiation_converted_to_watts(self):
        ds = _make_surface_ds(n_time=5)
        ssrd_raw = ds["ssrd"].values.copy()
        ds_conv = convert_era5_surface(ds)
        np.testing.assert_allclose(ds_conv["ssrd"].values, ssrd_raw / 3600.0)
        assert ds_conv["ssrd"].attrs["units"] == "W m-2"

    def test_precip_converted_to_mm(self):
        ds = _make_surface_ds(n_time=5)
        tp_raw = ds["tp"].values.copy()
        ds_conv = convert_era5_surface(ds)
        np.testing.assert_allclose(ds_conv["tp"].values, tp_raw * 1000.0)
        assert ds_conv["tp"].attrs["units"] == "mm"


class TestConvertEra5WithQC:
    """Test that convert_era5 integrates QC correctly."""

    def test_qc_runs_by_default(self):
        ds_surf = _make_surface_ds(n_time=10)
        ds_plev = xr.Dataset()  # Empty pressure dataset

        # Inject spike
        ds_surf["t2m"].values[5, 1, 1] -= 50.0

        surf, _ = convert_era5(ds_surf, ds_plev)

        # The spike should have been fixed before conversion
        # Value should be reasonable (near 290K, not 240K)
        assert surf["t2m"].values[5, 1, 1] > 280.0

    def test_qc_can_be_disabled(self):
        ds_surf = _make_surface_ds(n_time=10)
        ds_plev = xr.Dataset()

        # Inject spike
        ds_surf["t2m"].values[5, 1, 1] -= 50.0

        surf, _ = convert_era5(ds_surf, ds_plev, run_qc=False)

        # Spike should still be there
        assert surf["t2m"].values[5, 1, 1] < 250.0


class TestDetectTimeStep:
    """Tests for detect_time_step."""

    def test_3hourly(self):
        ds = _make_surface_ds(n_time=24, freq="3h")
        assert detect_time_step(ds) == 3.0

    def test_hourly(self):
        ds = _make_surface_ds(n_time=24, freq="1h")
        assert detect_time_step(ds) == 1.0

    def test_single_timestep_default(self):
        ds = _make_surface_ds(n_time=1)
        assert detect_time_step(ds) == 1.0
