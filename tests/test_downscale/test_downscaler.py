"""Tests for the downscaling engine."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.config.schema import WindConfig
from topopyscale2.core.downscale import Downscaler
from topopyscale2.spatial.units import SpatialUnit


def make_unit(
    elevation=3000.0,
    slope=15.0,
    aspect=180.0,
    svf=0.9,
    surface_type="open",
    sx=None,
):
    """Create a spatial unit for testing.

    Parameters
    ----------
    elevation : float
        Unit elevation in meters.
    slope : float
        Slope angle in degrees.
    aspect : float
        Aspect angle in degrees (0=N, 90=E, 180=S, 270=W).
    svf : float
        Sky view factor [0-1].
    surface_type : str
        Surface type key (open, forest, glacier, rock).
    sx : float, optional
        Winstral Sx terrain parameter [degrees].
    """
    attrs = {"slope": slope, "aspect": aspect, "svf": svf, "elevation": elevation}
    if sx is not None:
        attrs["sx"] = sx
    return SpatialUnit(
        id="test_unit",
        centroid=(76.5, 42.5, elevation),
        attributes=attrs,
        surface_type=surface_type,
        area_m2=900.0,
    )


def make_surf_forcing(
    nt=24,
    t2m_base=280.0,
    z_surf_geopot=25000.0,
    u10=5.0,
    v10=0.0,
):
    """Synthetic ERA5 surface forcing.

    Parameters
    ----------
    nt : int
        Number of timesteps.
    t2m_base : float
        2m temperature [K].
    z_surf_geopot : float
        Surface geopotential [m2/s2].
    u10 : float or array
        10m u-wind component [m/s]. Positive = eastward.
    v10 : float or array
        10m v-wind component [m/s]. Positive = northward.
    """
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values

    # Handle scalar vs array wind inputs
    if np.isscalar(u10):
        u10 = np.full(nt, u10)
    if np.isscalar(v10):
        v10 = np.full(nt, v10)

    return xr.Dataset(
        {
            "t2m": ("time", np.full(nt, t2m_base)),
            "d2m": ("time", np.full(nt, t2m_base - 5.0)),
            "sp": ("time", np.full(nt, 85000.0)),
            "ssrd": ("time", np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, nt)) * 800)),
            "strd": ("time", np.full(nt, 280.0)),
            "tp": ("time", np.full(nt, 0.001)),
            "z_surf": z_surf_geopot,  # Surface geopotential [m2/s2]
            "u10": ("time", u10),
            "v10": ("time", v10),
        },
        coords={"time": times},
    )


def make_plev_forcing(nt=24, z_surf_m=2549.0):
    """Synthetic ERA5 pressure-level forcing.

    Parameters
    ----------
    nt : int
        Number of timesteps.
    z_surf_m : float
        Surface elevation [m] for reference.

    Creates 4 pressure levels (700, 800, 850, 900 hPa) with realistic
    temperature, geopotential, wind, and humidity profiles.
    """
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    levels = [700, 800, 850, 900]  # hPa
    n_levels = len(levels)

    # Geopotential heights [m] - typical values for these pressure levels
    # 700 hPa ~ 3000m, 800 hPa ~ 1950m, 850 hPa ~ 1500m, 900 hPa ~ 1000m
    z_heights = np.array([3000.0, 1950.0, 1500.0, 1000.0])

    # Temperature [K] - decreasing with altitude (lapse rate ~ -6.5 K/km)
    t_base = 280.0  # At surface
    t_levels = t_base + (-0.0065) * (z_heights - z_surf_m)

    # Create 2D arrays (level, time)
    t_plev = np.tile(t_levels[:, np.newaxis], (1, nt))
    z_plev = np.tile(z_heights[:, np.newaxis], (1, nt)) * 9.80665  # Convert to geopotential [m2/s2]

    # Wind components - slight increase with altitude
    u_plev = np.tile(np.array([8.0, 6.0, 5.0, 4.0])[:, np.newaxis], (1, nt))
    v_plev = np.tile(np.array([2.0, 1.5, 1.0, 0.5])[:, np.newaxis], (1, nt))

    # Specific humidity [kg/kg] - decreasing with altitude
    q_plev = np.tile(np.array([0.002, 0.003, 0.004, 0.005])[:, np.newaxis], (1, nt))

    return xr.Dataset(
        {
            "t": (["level", "time"], t_plev),
            "z": (["level", "time"], z_plev),
            "u": (["level", "time"], u_plev),
            "v": (["level", "time"], v_plev),
            "q": (["level", "time"], q_plev),
        },
        coords={"time": times, "level": levels},
    )


def make_derived(nt=24):
    """Synthetic derived fields (solar geometry, clearness index)."""
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    sol_elev = xr.DataArray(
        np.maximum(0, np.sin(np.linspace(0, 2 * np.pi, nt)) * 60),
        dims=["time"],
        coords={"time": times},
    )
    sol_az = xr.DataArray(np.linspace(90, 270, nt), dims=["time"], coords={"time": times})
    kt = xr.DataArray(np.full(nt, 0.6), dims=["time"], coords={"time": times})
    return sol_elev, sol_az, kt


class TestProcessUnit:
    def test_output_variables(self):
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(), surf, plev, sol_elev, sol_az, kt
        )
        expected_vars = [
            "temperature", "precipitation", "rainfall", "snowfall",
            "shortwave_direct", "shortwave_diffuse", "longwave",
            "humidity_specific", "humidity_relative", "wind_speed",
            "wind_direction", "pressure",
        ]
        for var in expected_vars:
            assert var in ds, f"Missing variable: {var}"

    def test_temperature_decreases_with_elevation(self):
        """Higher unit → lower temperature (from pressure-level interpolation)."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        low = Downscaler("python").process_unit(
            make_unit(elevation=1500), surf, plev, sol_elev, sol_az, kt
        )
        high = Downscaler("python").process_unit(
            make_unit(elevation=2800), surf, plev, sol_elev, sol_az, kt
        )
        assert np.all(high["temperature"].values <= low["temperature"].values)

    def test_precipitation_increases_with_elevation(self):
        """With an opt-in positive gradient, higher unit → more precipitation."""
        # z_surf passed in metres here so dz is positive for both units.
        surf = make_surf_forcing(z_surf_geopot=2549.0)
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        # Gradient is off by default; enable it explicitly to test the behaviour.
        low = Downscaler("python", precip_gradient=0.0003).process_unit(
            make_unit(elevation=2600), surf, plev, sol_elev, sol_az, kt
        )
        high = Downscaler("python", precip_gradient=0.0003).process_unit(
            make_unit(elevation=4000), surf, plev, sol_elev, sol_az, kt
        )
        assert np.all(high["precipitation"].values > low["precipitation"].values)

    def test_wetbulb_default_gives_at_least_as_much_snow(self):
        """Default wet-bulb partitioning yields >= snowfall vs air-temperature.

        Wet-bulb temperature is <= air temperature, so the rain fraction can only
        shrink — snowfall is >= the air-temperature result, strictly so when the
        partition temperature lands in the mixed-phase ramp in unsaturated air.
        """
        # z_surf in metres; high elevation puts t_target in the mixed-phase ramp.
        surf = make_surf_forcing(z_surf_geopot=2549.0, t2m_base=280.0)
        sol_elev, sol_az, kt = make_derived()
        unit = make_unit(elevation=3500)

        wb = Downscaler("python", mode="simple")  # wet_bulb is the default
        air = Downscaler("python", mode="simple", phase_method="air_temperature")

        ds_wb = wb.process_unit_simple(unit, surf, sol_elev, sol_az, kt)
        ds_air = air.process_unit_simple(unit, surf, sol_elev, sol_az, kt)

        assert np.all(ds_wb["snowfall"].values >= ds_air["snowfall"].values - 1e-12)
        assert ds_wb["snowfall"].values.sum() > ds_air["snowfall"].values.sum()
        # Conservation still holds under wet-bulb.
        np.testing.assert_allclose(
            ds_wb["rainfall"].values + ds_wb["snowfall"].values,
            ds_wb["precipitation"].values, rtol=1e-10,
        )

    def test_air_temp_guard_forces_rain_in_warm_dry_air(self):
        """Guard rail forces all-rain above t_air_max even when wet-bulb says snow."""
        nt = 24
        times = pd.date_range("2020-01-01", periods=nt, freq="h").values
        # Warm air (+6 °C) but very dry (dewpoint −8 °C) at ~3500 m: wet-bulb sits
        # near freezing, so the wet-bulb ramp alone would produce snow.
        surf = xr.Dataset(
            {
                "t2m": ("time", np.full(nt, 279.15)),   # +6 °C
                "d2m": ("time", np.full(nt, 265.15)),   # very dry
                "sp": ("time", np.full(nt, 65000.0)),   # ~3500 m
                "ssrd": ("time", np.zeros(nt)),
                "strd": ("time", np.full(nt, 250.0)),
                "tp": ("time", np.full(nt, 0.001)),
                "z_surf": 3500.0,
                "u10": ("time", np.full(nt, 1.0)),
                "v10": ("time", np.zeros(nt)),
            },
            coords={"time": times},
        )
        sol_elev, sol_az, kt = make_derived(nt)
        unit = make_unit(elevation=3500)  # dz=0 → t_target ≈ +6 °C air temp

        guarded = Downscaler("python", mode="simple")            # t_air_max = +4 °C default
        no_guard = Downscaler("python", mode="simple", t_air_max=350.0)  # disabled

        dg = guarded.process_unit_simple(unit, surf, sol_elev, sol_az, kt)
        dn = no_guard.process_unit_simple(unit, surf, sol_elev, sol_az, kt)

        # Air temp (+6 °C) exceeds the +4 °C guard → all rain.
        np.testing.assert_allclose(dg["snowfall"].values, 0.0, atol=1e-12)
        np.testing.assert_allclose(
            dg["rainfall"].values, dg["precipitation"].values, rtol=1e-10
        )
        # Without the guard, the wet-bulb ramp produces snow at the same warm air temp.
        assert dn["snowfall"].values.sum() > 0.0

    def test_precipitation_passthrough_by_default(self):
        """Default (no gradient) → downscaled precip equals source precip."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        ds = Downscaler("python").process_unit(
            make_unit(elevation=4000), surf, plev, sol_elev, sol_az, kt
        )
        np.testing.assert_allclose(
            ds["precipitation"].values, surf["tp"].values, rtol=1e-10
        )

    def test_rain_snow_conservation(self):
        """Rainfall + snowfall = total precipitation."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(), surf, plev, sol_elev, sol_az, kt
        )
        total = ds["rainfall"].values + ds["snowfall"].values
        np.testing.assert_allclose(total, ds["precipitation"].values, rtol=1e-10)

    def test_svf_affects_diffuse(self):
        """Lower SVF → less diffuse radiation."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        open_unit = make_unit(svf=1.0)
        shaded_unit = make_unit(svf=0.5)

        ds_open = Downscaler("python").process_unit(
            open_unit, surf, plev, sol_elev, sol_az, kt
        )
        ds_shaded = Downscaler("python").process_unit(
            shaded_unit, surf, plev, sol_elev, sol_az, kt
        )

        # At least some timesteps should differ
        assert np.any(ds_shaded["shortwave_diffuse"].values < ds_open["shortwave_diffuse"].values)


class TestMapUnitsToERA5:
    def test_nearest_grid_cell(self):
        units = [
            SpatialUnit("u1", (76.5, 42.1, 3000)),
            SpatialUnit("u2", (77.2, 42.8, 3500)),
        ]
        lat = np.array([42.0, 42.25, 42.5, 42.75, 43.0])
        lon = np.array([76.0, 76.25, 76.5, 76.75, 77.0, 77.25])

        mapping = Downscaler.map_units_to_era5(units, lat, lon)

        assert mapping["u1"] == (0, 2)  # lat~42.0, lon~76.5
        assert mapping["u2"][1] == 5     # lon~77.25


class TestWindDownscaling:
    """Tests for wind downscaling with pressure-level interpolation + log-profile and Winstral corrections."""

    def test_nonzero_wind_output(self):
        """Wind speed should be non-zero when pressure-level data has wind."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(), surf, plev, sol_elev, sol_az, kt
        )
        assert np.all(ds["wind_speed"].values > 0), "Wind speed should be positive"

    def test_wind_speed_magnitude(self):
        """Wind speed should be interpolated from pressure levels and corrected."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(surface_type="open", elevation=2000),
            surf, plev, sol_elev, sol_az, kt,
        )
        # Wind is interpolated from pressure levels then corrected
        # At 2000m, interpolated wind should be roughly 5-6 m/s before correction
        # After log profile correction from 50m to 10m, it should be reduced
        assert ds["wind_speed"].values.mean() > 0.5, "Wind should be positive after correction"
        assert ds["wind_speed"].values.mean() < 10.0, "Wind should be reasonable magnitude"

    def test_forest_reduces_wind(self):
        """Forest surface type should reduce wind speed (higher z0)."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        open_unit = make_unit(surface_type="open")
        forest_unit = make_unit(surface_type="forest")

        ds_open = Downscaler("python").process_unit(
            open_unit, surf, plev, sol_elev, sol_az, kt
        )
        ds_forest = Downscaler("python").process_unit(
            forest_unit, surf, plev, sol_elev, sol_az, kt
        )

        # Forest should have lower wind speed due to higher roughness
        assert np.all(ds_forest["wind_speed"].values < ds_open["wind_speed"].values), (
            "Forest wind should be lower than open terrain"
        )

    def test_glacier_different_from_open(self):
        """Glacier (very low z0) should have higher wind than open terrain."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        open_unit = make_unit(surface_type="open")
        glacier_unit = make_unit(surface_type="glacier")

        ds_open = Downscaler("python").process_unit(
            open_unit, surf, plev, sol_elev, sol_az, kt
        )
        ds_glacier = Downscaler("python").process_unit(
            glacier_unit, surf, plev, sol_elev, sol_az, kt
        )

        # Glacier (z0=0.001) has lower roughness than open (z0=0.03)
        # So wind speed should be higher over glacier
        assert np.all(ds_glacier["wind_speed"].values > ds_open["wind_speed"].values), (
            "Glacier wind should be higher than open terrain"
        )

    def test_wind_direction_east_wind(self):
        """Wind direction should be computed from interpolated u, v components."""
        # Create pressure levels with eastward wind (u>0, v=0)
        surf = make_surf_forcing()
        nt = 24
        times = pd.date_range("2020-01-01", periods=nt, freq="h").values
        levels = [700, 800, 850, 900]
        n_levels = len(levels)

        # Pure eastward wind at all levels
        plev = xr.Dataset(
            {
                "t": (["level", "time"], np.tile(np.array([265, 270, 273, 276])[:, np.newaxis], (1, nt))),
                "z": (["level", "time"], np.tile(np.array([3000, 1950, 1500, 1000])[:, np.newaxis] * 9.80665, (1, nt))),
                "u": (["level", "time"], np.full((n_levels, nt), 5.0)),  # Eastward
                "v": (["level", "time"], np.full((n_levels, nt), 0.0)),  # No north/south
                "q": (["level", "time"], np.full((n_levels, nt), 0.003)),
            },
            coords={"time": times, "level": levels},
        )
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(elevation=2000), surf, plev, sol_elev, sol_az, kt
        )
        # Wind FROM west = 270 degrees
        np.testing.assert_allclose(
            ds["wind_direction"].values, 270.0, atol=1.0,
            err_msg="Eastward wind should be from west (270)",
        )

    def test_wind_direction_north_wind(self):
        """Pure northward wind (u=0, v>0) should give direction ~180 (from south)."""
        surf = make_surf_forcing()
        nt = 24
        times = pd.date_range("2020-01-01", periods=nt, freq="h").values
        levels = [700, 800, 850, 900]
        n_levels = len(levels)

        plev = xr.Dataset(
            {
                "t": (["level", "time"], np.tile(np.array([265, 270, 273, 276])[:, np.newaxis], (1, nt))),
                "z": (["level", "time"], np.tile(np.array([3000, 1950, 1500, 1000])[:, np.newaxis] * 9.80665, (1, nt))),
                "u": (["level", "time"], np.full((n_levels, nt), 0.0)),
                "v": (["level", "time"], np.full((n_levels, nt), 5.0)),  # Northward
                "q": (["level", "time"], np.full((n_levels, nt), 0.003)),
            },
            coords={"time": times, "level": levels},
        )
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(elevation=2000), surf, plev, sol_elev, sol_az, kt
        )
        # Wind FROM south = 180 degrees
        np.testing.assert_allclose(
            ds["wind_direction"].values, 180.0, atol=1.0,
            err_msg="Northward wind should be from south (180)",
        )

    def test_wind_direction_south_wind(self):
        """Pure southward wind (u=0, v<0) should give direction ~0/360 (from north)."""
        surf = make_surf_forcing()
        nt = 24
        times = pd.date_range("2020-01-01", periods=nt, freq="h").values
        levels = [700, 800, 850, 900]
        n_levels = len(levels)

        plev = xr.Dataset(
            {
                "t": (["level", "time"], np.tile(np.array([265, 270, 273, 276])[:, np.newaxis], (1, nt))),
                "z": (["level", "time"], np.tile(np.array([3000, 1950, 1500, 1000])[:, np.newaxis] * 9.80665, (1, nt))),
                "u": (["level", "time"], np.full((n_levels, nt), 0.0)),
                "v": (["level", "time"], np.full((n_levels, nt), -5.0)),  # Southward
                "q": (["level", "time"], np.full((n_levels, nt), 0.003)),
            },
            coords={"time": times, "level": levels},
        )
        sol_elev, sol_az, kt = make_derived()
        ds = Downscaler("python").process_unit(
            make_unit(elevation=2000), surf, plev, sol_elev, sol_az, kt
        )
        # Wind FROM north = 0 degrees (or 360)
        direction = ds["wind_direction"].values
        # Accept either 0 or 360
        assert np.allclose(direction, 0.0, atol=1.0) or np.allclose(direction, 360.0, atol=1.0), (
            f"Southward wind should be from north (0 or 360), got {direction[0]}"
        )

    def test_wind_direction_preserved_across_surface_types(self):
        """Wind direction should be identical regardless of surface type."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        ds_open = Downscaler("python").process_unit(
            make_unit(surface_type="open"), surf, plev, sol_elev, sol_az, kt
        )
        ds_forest = Downscaler("python").process_unit(
            make_unit(surface_type="forest"), surf, plev, sol_elev, sol_az, kt
        )

        np.testing.assert_array_equal(
            ds_open["wind_direction"].values,
            ds_forest["wind_direction"].values,
            err_msg="Wind direction should be same for all surface types",
        )

    def test_winstral_correction_exposed(self):
        """Positive Sx (exposed terrain) should increase wind speed."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()
        wind_config = WindConfig(method="winstral")

        unit_flat = make_unit(sx=0.0)  # No exposure
        unit_exposed = make_unit(sx=20.0)  # Exposed ridge

        ds_flat = Downscaler("python", wind_config=wind_config).process_unit(
            unit_flat, surf, plev, sol_elev, sol_az, kt
        )
        ds_exposed = Downscaler("python", wind_config=wind_config).process_unit(
            unit_exposed, surf, plev, sol_elev, sol_az, kt
        )

        assert np.all(ds_exposed["wind_speed"].values > ds_flat["wind_speed"].values), (
            "Exposed terrain (Sx>0) should have higher wind"
        )

    def test_winstral_correction_sheltered(self):
        """Negative Sx (sheltered terrain) should decrease wind speed."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()
        wind_config = WindConfig(method="winstral")

        unit_flat = make_unit(sx=0.0)  # No exposure
        unit_sheltered = make_unit(sx=-20.0)  # Sheltered lee

        ds_flat = Downscaler("python", wind_config=wind_config).process_unit(
            unit_flat, surf, plev, sol_elev, sol_az, kt
        )
        ds_sheltered = Downscaler("python", wind_config=wind_config).process_unit(
            unit_sheltered, surf, plev, sol_elev, sol_az, kt
        )

        assert np.all(ds_sheltered["wind_speed"].values < ds_flat["wind_speed"].values), (
            "Sheltered terrain (Sx<0) should have lower wind"
        )

    def test_winstral_not_applied_without_config(self):
        """Winstral correction should NOT be applied when method is log_profile."""
        surf = make_surf_forcing()
        plev = make_plev_forcing()
        sol_elev, sol_az, kt = make_derived()

        # Default method is log_profile, not winstral
        unit_exposed = make_unit(sx=20.0)
        unit_flat = make_unit(sx=0.0)

        ds_exposed = Downscaler("python").process_unit(
            unit_exposed, surf, plev, sol_elev, sol_az, kt
        )
        ds_flat = Downscaler("python").process_unit(
            unit_flat, surf, plev, sol_elev, sol_az, kt
        )

        # With log_profile method, Sx should be ignored
        np.testing.assert_array_equal(
            ds_exposed["wind_speed"].values,
            ds_flat["wind_speed"].values,
            err_msg="Sx should be ignored when method is log_profile",
        )


class TestInitValidation:
    """Downscaler constructor parameter validation."""

    def test_defaults_valid(self):
        Downscaler("python")

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            Downscaler("python", mode="hybrid")

    def test_invalid_phase_method(self):
        with pytest.raises(ValueError, match="phase_method"):
            Downscaler("python", phase_method="dewpoint")

    def test_lapse_rate_out_of_range(self):
        with pytest.raises(ValueError, match="lapse_rate"):
            Downscaler("python", lapse_rate=0.5)

    def test_lapse_rate_nan(self):
        with pytest.raises(ValueError, match="lapse_rate"):
            Downscaler("python", lapse_rate=float("nan"))

    def test_negative_lapse_rate_allowed(self):
        # Inversions are physical — a negative lapse rate is allowed
        Downscaler("python", lapse_rate=-0.005)

    def test_precip_gradient_out_of_range(self):
        with pytest.raises(ValueError, match="precip_gradient"):
            Downscaler("python", precip_gradient=0.1)

    def test_phase_thresholds_misordered(self):
        with pytest.raises(ValueError, match="t_snow"):
            Downscaler("python", t_snow=276.0, t_rain=274.0)

    def test_phase_thresholds_equal_rejected(self):
        with pytest.raises(ValueError, match="t_snow"):
            Downscaler("python", t_snow=274.0, t_rain=274.0)
