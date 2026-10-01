"""Cross-backend temperature kernel tests."""

import numpy as np
import pytest

from tests.conftest import ATOL, RTOL


class TestLapseRateCorrection:
    def test_scalar_known_value(self, temperature_mod):
        """290.25 = 300 + (-0.0065) * 1500."""
        result = temperature_mod.lapse_rate_correction(
            np.float64(300.0),
            np.float64(-0.0065),
            np.float64(1500.0),
            np.float64(0.0),
        )
        np.testing.assert_allclose(result, 290.25, rtol=RTOL, atol=ATOL)

    def test_no_elevation_difference(self, temperature_mod):
        result = temperature_mod.lapse_rate_correction(
            np.float64(280.0),
            np.float64(-0.0065),
            np.float64(500.0),
            np.float64(500.0),
        )
        np.testing.assert_allclose(result, 280.0, rtol=RTOL, atol=ATOL)

    def test_arrays(self, temperature_mod):
        t_source = np.array([300.0, 280.0, 290.0])
        gamma = np.array([-0.0065, -0.0065, -0.005])
        z_unit = np.array([2000.0, 1000.0, 3000.0])
        z_source = np.array([500.0, 500.0, 1000.0])

        result = temperature_mod.lapse_rate_correction(t_source, gamma, z_unit, z_source)

        expected = t_source + gamma * (z_unit - z_source)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_positive_lapse_rate(self, temperature_mod):
        """Inversion: temperature increases with altitude."""
        result = temperature_mod.lapse_rate_correction(
            np.float64(270.0),
            np.float64(0.005),
            np.float64(1000.0),
            np.float64(500.0),
        )
        np.testing.assert_allclose(result, 272.5, rtol=RTOL, atol=ATOL)

    def test_2d_arrays(self, temperature_mod):
        t = np.full((3, 4), 290.0)
        gamma = np.full((3, 4), -0.0065)
        z_unit = np.full((3, 4), 2000.0)
        z_source = np.full((3, 4), 1000.0)
        result = temperature_mod.lapse_rate_correction(t, gamma, z_unit, z_source)
        expected = 290.0 + (-0.0065) * 1000.0
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestSimpleLapseRate:
    """Tests for simple lapse rate correction (ERA5-Land compatible mode)."""

    def test_basic_correction(self, temperature_mod):
        """Temperature decreases going uphill with default lapse rate."""
        t_surface = np.array([280.0, 290.0, 300.0])
        z_surface = np.array([500.0, 1000.0, 200.0])
        z_target = 2000.0
        lapse_rate = 0.0065

        result = temperature_mod.simple_lapse_rate(t_surface, z_surface, z_target, lapse_rate)

        # T_target = T_surface - lapse_rate * (z_target - z_surface)
        expected = t_surface - lapse_rate * (z_target - z_surface)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_known_value(self, temperature_mod):
        """Going from 500m to 2000m with 6.5 K/km lapse rate."""
        t_surface = np.array([290.0])
        z_surface = np.array([500.0])
        z_target = 2000.0

        result = temperature_mod.simple_lapse_rate(t_surface, z_surface, z_target, 0.0065)

        # 290 - 0.0065 * 1500 = 290 - 9.75 = 280.25
        np.testing.assert_allclose(result, 280.25, rtol=RTOL, atol=ATOL)

    def test_downhill_warms(self, temperature_mod):
        """Temperature increases going downhill."""
        t_surface = np.array([280.0])
        z_surface = np.array([2000.0])
        z_target = 500.0

        result = temperature_mod.simple_lapse_rate(t_surface, z_surface, z_target, 0.0065)

        # 280 - 0.0065 * (500 - 2000) = 280 - 0.0065 * (-1500) = 280 + 9.75 = 289.75
        np.testing.assert_allclose(result, 289.75, rtol=RTOL, atol=ATOL)

    def test_no_elevation_change(self, temperature_mod):
        """No change when target equals source elevation."""
        t_surface = np.array([285.0])
        z_surface = np.array([1500.0])
        z_target = 1500.0

        result = temperature_mod.simple_lapse_rate(t_surface, z_surface, z_target, 0.0065)

        np.testing.assert_allclose(result, 285.0, rtol=RTOL, atol=ATOL)

    def test_default_lapse_rate(self, temperature_mod):
        """Default lapse rate is 0.0065 K/m (6.5 K/km)."""
        t_surface = np.array([290.0])
        z_surface = np.array([1000.0])
        z_target = 2000.0

        # With default lapse rate
        result = temperature_mod.simple_lapse_rate(t_surface, z_surface, z_target)

        # Should use 0.0065 as default
        expected = 290.0 - 0.0065 * 1000.0
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_custom_lapse_rate(self, temperature_mod):
        """Can use custom lapse rate (e.g., moist adiabatic ~5 K/km)."""
        t_surface = np.array([290.0])
        z_surface = np.array([1000.0])
        z_target = 2000.0
        lapse_rate = 0.005  # 5 K/km

        result = temperature_mod.simple_lapse_rate(t_surface, z_surface, z_target, lapse_rate)

        expected = 290.0 - 0.005 * 1000.0  # 285.0
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestCrossBackendTemperature:
    """Verify all backends produce identical results."""

    def test_agreement(self):
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        t_source = np.array([300.0, 280.0, 290.0, 260.0])
        gamma = np.array([-0.0065, -0.005, -0.007, -0.006])
        z_unit = np.array([2000.0, 1000.0, 3000.0, 4000.0])
        z_source = np.array([500.0, 300.0, 1000.0, 800.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["temperature"].lapse_rate_correction(t_source, gamma, z_unit, z_source)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"Mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_simple_lapse_rate(self):
        """Cross-backend agreement for simple_lapse_rate."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        t_surface = np.array([300.0, 280.0, 290.0, 260.0])
        z_surface = np.array([500.0, 1000.0, 200.0, 1500.0])
        z_target = 2500.0
        lapse_rate = 0.0065

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["temperature"].simple_lapse_rate(t_surface, z_surface, z_target, lapse_rate)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"Mismatch between {names[0]} and {names[i]}",
            )
