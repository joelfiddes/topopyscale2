"""Tests for flexible variable handling in the downscaler.

Verifies that the downscaler can operate with partial inputs,
only computing the groups that are resolvable from available data
and requested outputs.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.core.downscale import (
    VARIABLE_GROUPS,
    Downscaler,
    resolve_groups,
)
from topopyscale2.spatial.units import SpatialUnit

# ── Fixtures ─────────────────────────────────────────────────────────────


def _make_unit(elevation=3000.0):
    return SpatialUnit(
        id="test_unit",
        centroid=(76.5, 42.5, elevation),
        attributes={"slope": 15.0, "aspect": 180.0, "svf": 0.9, "elevation": elevation},
        surface_type="open",
        area_m2=900.0,
    )


def _make_times(nt=24):
    return pd.date_range("2020-01-01", periods=nt, freq="h").values


def _make_surf_temp_only(nt=24):
    """Minimal surface forcing: only t2m + sp + z_surf (temperature group)."""
    times = _make_times(nt)
    return xr.Dataset(
        {
            "t2m": ("time", np.full(nt, 280.0)),
            "sp": ("time", np.full(nt, 85000.0)),
            "z_surf": 2549.0,
        },
        coords={"time": times},
    )


def _make_surf_temp_precip(nt=24):
    """Surface forcing with t2m + sp + tp + z_surf (temperature + precipitation)."""
    times = _make_times(nt)
    return xr.Dataset(
        {
            "t2m": ("time", np.full(nt, 280.0)),
            "sp": ("time", np.full(nt, 85000.0)),
            "tp": ("time", np.full(nt, 0.001)),
            "z_surf": 2549.0,
        },
        coords={"time": times},
    )


def _make_surf_full_simple(nt=24):
    """Full surface forcing for simple mode (all vars)."""
    times = _make_times(nt)
    return xr.Dataset(
        {
            "t2m": ("time", np.full(nt, 280.0)),
            "d2m": ("time", np.full(nt, 275.0)),
            "sp": ("time", np.full(nt, 85000.0)),
            "ssrd": ("time", np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, nt)) * 800)),
            "strd": ("time", np.full(nt, 280.0)),
            "tp": ("time", np.full(nt, 0.001)),
            "z_surf": 2549.0,
            "u10": ("time", np.full(nt, 5.0)),
            "v10": ("time", np.full(nt, 0.0)),
        },
        coords={"time": times},
    )


def _make_derived(nt=24):
    times = _make_times(nt)
    sol_elev = xr.DataArray(
        np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, nt)) * 60),
        dims=["time"], coords={"time": times},
    )
    sol_az = xr.DataArray(np.linspace(90, 270, nt), dims=["time"], coords={"time": times})
    kt = xr.DataArray(np.full(nt, 0.6), dims=["time"], coords={"time": times})
    return sol_elev, sol_az, kt


# ── resolve_groups() tests ───────────────────────────────────────────────


class TestResolveGroups:
    def test_temp_only_from_inputs(self):
        """Only t2m + sp available → only temperature group."""
        groups = resolve_groups(
            available_surface={"t2m", "sp", "z_surf"},
            available_plev=set(),
            requested_outputs=[],
            mode="simple",
        )
        assert "temperature" in groups
        assert "precipitation" not in groups
        assert "radiation" not in groups
        assert "wind" not in groups
        assert "humidity" not in groups

    def test_temp_precip(self):
        """t2m + sp + tp → temperature + precipitation."""
        groups = resolve_groups(
            available_surface={"t2m", "sp", "tp", "z_surf"},
            available_plev=set(),
            requested_outputs=[],
            mode="simple",
        )
        assert "temperature" in groups
        assert "precipitation" in groups
        assert "radiation" not in groups

    def test_all_simple(self):
        """All simple-mode vars → all groups."""
        groups = resolve_groups(
            available_surface={"t2m", "d2m", "sp", "ssrd", "strd", "tp", "u10", "v10", "z_surf"},
            available_plev=set(),
            requested_outputs=[],
            mode="simple",
        )
        assert set(groups) == set(VARIABLE_GROUPS.keys())

    def test_all_full(self):
        """All vars (full mode) → all groups."""
        groups = resolve_groups(
            available_surface={"t2m", "d2m", "sp", "ssrd", "strd", "tp", "z_surf"},
            available_plev={"t", "z", "u", "v", "q"},
            requested_outputs=[],
            mode="full",
        )
        assert set(groups) == set(VARIABLE_GROUPS.keys())

    def test_dependency_auto_includes_temperature(self):
        """Requesting precipitation auto-includes temperature (dependency)."""
        groups = resolve_groups(
            available_surface={"t2m", "sp", "tp", "z_surf"},
            available_plev=set(),
            requested_outputs=["precipitation"],
            mode="simple",
        )
        assert "temperature" in groups
        assert "precipitation" in groups

    def test_missing_dependency_drops_both(self):
        """Request precipitation but no t2m → neither group computes."""
        groups = resolve_groups(
            available_surface={"tp", "z_surf"},
            available_plev=set(),
            requested_outputs=["precipitation"],
            mode="simple",
        )
        assert "temperature" not in groups
        assert "precipitation" not in groups

    def test_empty_requested_means_all_computable(self):
        """Empty requested_outputs means all computable groups."""
        groups = resolve_groups(
            available_surface={"t2m", "sp", "tp", "z_surf"},
            available_plev=set(),
            requested_outputs=[],
            mode="simple",
        )
        # Should include temperature and precipitation (both computable)
        assert "temperature" in groups
        assert "precipitation" in groups

    def test_requested_specific_output(self):
        """Request only 'temperature' → only temperature group."""
        groups = resolve_groups(
            available_surface={"t2m", "d2m", "sp", "ssrd", "strd", "tp", "u10", "v10", "z_surf"},
            available_plev=set(),
            requested_outputs=["temperature"],
            mode="simple",
        )
        assert groups == ["temperature"]

    def test_wind_simple_needs_u10_v10(self):
        """Wind in simple mode needs u10/v10."""
        groups = resolve_groups(
            available_surface={"t2m", "sp", "z_surf"},
            available_plev=set(),
            requested_outputs=["wind_speed"],
            mode="simple",
        )
        assert "wind" not in groups

        groups2 = resolve_groups(
            available_surface={"t2m", "sp", "u10", "v10", "z_surf"},
            available_plev=set(),
            requested_outputs=["wind_speed"],
            mode="simple",
        )
        assert "wind" in groups2

    def test_full_mode_wind_needs_plev(self):
        """Wind in full mode needs pressure-level u, v, z."""
        groups = resolve_groups(
            available_surface={"t2m", "sp", "z_surf"},
            available_plev={"t", "z"},
            requested_outputs=["wind_speed"],
            mode="full",
        )
        assert "wind" not in groups

        groups2 = resolve_groups(
            available_surface={"t2m", "sp", "z_surf"},
            available_plev={"t", "z", "u", "v", "q"},
            requested_outputs=["wind_speed"],
            mode="full",
        )
        assert "wind" in groups2

    def test_order_temperature_first(self):
        """Temperature always comes first in the resolved order."""
        groups = resolve_groups(
            available_surface={"t2m", "d2m", "sp", "ssrd", "strd", "tp", "u10", "v10", "z_surf"},
            available_plev=set(),
            requested_outputs=[],
            mode="simple",
        )
        assert groups[0] == "temperature"


# ── process_unit_simple() with partial groups ────────────────────────────


class TestProcessUnitSimplePartial:
    def test_temp_only(self):
        """process_unit_simple with only temperature group."""
        ds = Downscaler("python", mode="simple").process_unit_simple(
            _make_unit(),
            _make_surf_temp_only(),
            groups={"temperature"},
        )
        assert "temperature" in ds
        assert "pressure" in ds
        assert "precipitation" not in ds
        assert "wind_speed" not in ds
        assert "longwave" not in ds

    def test_temp_precip(self):
        """process_unit_simple with temperature + precipitation groups."""
        ds = Downscaler("python", mode="simple").process_unit_simple(
            _make_unit(),
            _make_surf_temp_precip(),
            groups={"temperature", "precipitation"},
        )
        assert "temperature" in ds
        assert "pressure" in ds
        assert "precipitation" in ds
        assert "rainfall" in ds
        assert "snowfall" in ds
        assert "wind_speed" not in ds
        assert "longwave" not in ds

    def test_rain_snow_conservation_partial(self):
        """Rain + snow = total precip even in partial mode."""
        ds = Downscaler("python", mode="simple").process_unit_simple(
            _make_unit(),
            _make_surf_temp_precip(),
            groups={"temperature", "precipitation"},
        )
        total = ds["rainfall"].values + ds["snowfall"].values
        np.testing.assert_allclose(total, ds["precipitation"].values, rtol=1e-10)

    def test_temp_only_values_reasonable(self):
        """Temperature values should be physically reasonable."""
        ds = Downscaler("python", mode="simple").process_unit_simple(
            _make_unit(elevation=3000.0),
            _make_surf_temp_only(),
            groups={"temperature"},
        )
        t = ds["temperature"].values
        # Source t2m=280K at 2549m, target at 3000m → should be ~277K
        assert np.all(t > 250.0) and np.all(t < 300.0)
        # Should be cooler than source due to higher elevation
        assert np.all(t < 280.0)


# ── Backward compatibility ───────────────────────────────────────────────


class TestBackwardCompatibility:
    def test_all_inputs_empty_requested_gives_all_vars(self):
        """Full inputs + empty requested_outputs → all 12 output variables."""
        sol_elev, sol_az, kt = _make_derived()
        ds = Downscaler("python", mode="simple").process_unit_simple(
            _make_unit(),
            _make_surf_full_simple(),
            solar_elevation=sol_elev,
            solar_azimuth=sol_az,
            clearness_index=kt,
            groups=None,  # None = backward compatible = all groups
        )
        expected = {
            "temperature", "pressure", "precipitation", "rainfall", "snowfall",
            "shortwave_direct", "shortwave_diffuse", "longwave",
            "humidity_specific", "humidity_relative",
            "wind_speed", "wind_direction",
        }
        assert set(ds.data_vars) == expected

    def test_process_unit_simple_no_groups_param(self):
        """Calling without groups param should still work (all groups)."""
        sol_elev, sol_az, kt = _make_derived()
        ds = Downscaler("python", mode="simple").process_unit_simple(
            _make_unit(),
            _make_surf_full_simple(),
            sol_elev, sol_az, kt,
        )
        assert len(ds.data_vars) == 12


# ── Config schema validation ─────────────────────────────────────────────


class TestOutputConfigValidation:
    def test_empty_variables_is_default(self):
        from topopyscale2.config.schema import OutputConfig
        cfg = OutputConfig()
        assert cfg.variables == []

    def test_valid_variables_accepted(self):
        from topopyscale2.config.schema import OutputConfig
        cfg = OutputConfig(variables=["temperature", "precipitation"])
        assert cfg.variables == ["temperature", "precipitation"]

    def test_invalid_variable_rejected(self):
        from topopyscale2.config.schema import OutputConfig
        with pytest.raises(ValueError, match="Invalid output variable"):
            OutputConfig(variables=["temperature", "not_a_real_variable"])
