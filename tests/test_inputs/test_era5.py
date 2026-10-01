"""Tests for ERA5 input source."""

import numpy as np
import xarray as xr

from topopyscale2.config.schema import InputConfig
from topopyscale2.inputs.era5 import ERA5Source


class TestComputeLapseRate:
    def test_linear_profile(self, synthetic_era5_pressure):
        """Known linear T profile → expected gamma."""
        config = InputConfig()
        source = ERA5Source(config)

        # Create a dataset with perfectly linear T vs Z
        g = 9.80665
        levels = np.array([300, 500, 700, 850, 1000])
        z_m = np.array([9500.0, 5500.0, 3000.0, 1500.0, 100.0])

        # Linear lapse: T = 288.0 - 0.0065 * (z - 100)
        gamma_expected = -0.0065
        t_linear = 288.0 + gamma_expected * (z_m - 100.0)

        ds = xr.Dataset(
            {
                "t": (["time", "level", "latitude", "longitude"],
                       t_linear[np.newaxis, :, np.newaxis, np.newaxis] * np.ones((1, 5, 1, 1))),
                "z": (["time", "level", "latitude", "longitude"],
                       (z_m * g)[np.newaxis, :, np.newaxis, np.newaxis] * np.ones((1, 5, 1, 1))),
            },
            coords={
                "time": [np.datetime64("2020-01-01")],
                "level": levels,
                "latitude": [42.0],
                "longitude": [76.0],
            },
        )

        z_surface = xr.DataArray([[100.0]], dims=["latitude", "longitude"])
        gamma = source.compute_lapse_rate(ds, z_surface)

        np.testing.assert_allclose(gamma.values[0, 0, 0], gamma_expected, rtol=1e-3)


class TestComputeSolarGeometry:
    def test_equinox_noon_equator(self):
        """Spring equinox, solar noon at equator → elevation ≈ 90°."""
        config = InputConfig()
        source = ERA5Source(config)

        # March 20 (day 80), noon UTC at longitude 0
        time = np.array([np.datetime64("2020-03-20T12:00")])
        lat = np.array([0.0])
        lon = np.array([0.0])

        elev, az = source.compute_solar_geometry(time, lat, lon)
        # Solar elevation should be close to 90° at equinox noon at equator
        assert float(elev.values[0, 0, 0]) > 80.0

    def test_night_negative_elevation(self):
        """Midnight at equator → negative solar elevation."""
        config = InputConfig()
        source = ERA5Source(config)

        time = np.array([np.datetime64("2020-06-21T00:00")])
        lat = np.array([0.0])
        lon = np.array([0.0])

        elev, az = source.compute_solar_geometry(time, lat, lon)
        assert float(elev.values[0, 0, 0]) < 0.0

    def test_output_shape(self):
        config = InputConfig()
        source = ERA5Source(config)

        time = np.array([np.datetime64("2020-01-01T00:00"), np.datetime64("2020-01-01T12:00")])
        lat = np.array([40.0, 42.0])
        lon = np.array([76.0, 78.0])

        elev, az = source.compute_solar_geometry(time, lat, lon)
        assert elev.shape == (2, 2, 2)
        assert az.shape == (2, 2, 2)


class TestComputeClearnessIndex:
    def test_clear_sky_high_kt(self):
        """High radiation relative to TOA → high kt."""
        config = InputConfig()
        source = ERA5Source(config)

        solar_elev = xr.DataArray(
            [[[60.0]]],
            dims=["time", "latitude", "longitude"],
        )
        ds_surf = xr.Dataset({
            "ssrd": xr.DataArray(
                [[[1000.0]]],
                dims=["time", "latitude", "longitude"],
            )
        })

        kt = source.compute_clearness_index(ds_surf, solar_elev)
        assert float(kt.values[0, 0, 0]) > 0.5

    def test_overcast_low_kt(self):
        """Low radiation → low kt."""
        config = InputConfig()
        source = ERA5Source(config)

        solar_elev = xr.DataArray([[[60.0]]], dims=["time", "latitude", "longitude"])
        ds_surf = xr.Dataset({
            "ssrd": xr.DataArray([[[50.0]]], dims=["time", "latitude", "longitude"])
        })

        kt = source.compute_clearness_index(ds_surf, solar_elev)
        assert float(kt.values[0, 0, 0]) < 0.2

    def test_clamping(self):
        """kt should be clamped to [0, 1]."""
        config = InputConfig()
        source = ERA5Source(config)

        solar_elev = xr.DataArray([[[60.0]]], dims=["time", "latitude", "longitude"])
        ds_surf = xr.Dataset({
            "ssrd": xr.DataArray([[[99999.0]]], dims=["time", "latitude", "longitude"])
        })

        kt = source.compute_clearness_index(ds_surf, solar_elev)
        assert float(kt.values[0, 0, 0]) <= 1.0

    def test_night_zero(self):
        """Night (negative solar elevation) → kt = 0."""
        config = InputConfig()
        source = ERA5Source(config)

        solar_elev = xr.DataArray([[[-10.0]]], dims=["time", "latitude", "longitude"])
        ds_surf = xr.Dataset({
            "ssrd": xr.DataArray([[[0.0]]], dims=["time", "latitude", "longitude"])
        })

        kt = source.compute_clearness_index(ds_surf, solar_elev)
        np.testing.assert_allclose(float(kt.values[0, 0, 0]), 0.0, atol=1e-15)
