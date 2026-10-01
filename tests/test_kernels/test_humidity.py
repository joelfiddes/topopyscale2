"""Cross-backend humidity kernel tests."""

import numpy as np

from tests.conftest import ATOL, RTOL


class TestAdjustHumidity:
    def test_q_conserved(self, humidity_mod):
        """Specific humidity should be unchanged."""
        q = np.float64(0.005)
        q_adj, rh = humidity_mod.adjust_humidity(q, np.float64(280.0), np.float64(80000.0))
        np.testing.assert_allclose(float(q_adj), float(q), rtol=RTOL, atol=ATOL)

    def test_rh_rises_with_cooling(self, humidity_mod):
        """Cooler temperature → higher RH (same q, same pressure)."""
        q = np.float64(0.005)
        p = np.float64(80000.0)

        _, rh_warm = humidity_mod.adjust_humidity(q, np.float64(290.0), p)
        _, rh_cold = humidity_mod.adjust_humidity(q, np.float64(270.0), p)
        assert float(rh_cold) > float(rh_warm)

    def test_rh_capped_at_1(self, humidity_mod):
        """Very cold, high q → RH should be capped at 1.0."""
        q = np.float64(0.01)
        _, rh = humidity_mod.adjust_humidity(q, np.float64(250.0), np.float64(100000.0))
        assert float(rh) <= 1.0 + 1e-15

    def test_rh_non_negative(self, humidity_mod):
        """RH should never be negative."""
        q = np.float64(0.0001)
        _, rh = humidity_mod.adjust_humidity(q, np.float64(310.0), np.float64(100000.0))
        assert float(rh) >= 0.0

    def test_arrays(self, humidity_mod):
        q = np.array([0.005, 0.003, 0.008])
        t = np.array([280.0, 260.0, 290.0])
        p = np.array([80000.0, 70000.0, 90000.0])

        q_adj, rh = humidity_mod.adjust_humidity(q, t, p)
        np.testing.assert_allclose(np.asarray(q_adj), q, rtol=RTOL)
        assert np.all(np.asarray(rh) >= 0.0)
        assert np.all(np.asarray(rh) <= 1.0 + 1e-15)

    def test_bolton_formula(self, humidity_mod):
        """Verify RH against manual Bolton (1980) calculation."""
        q = np.float64(0.005)
        t = np.float64(280.0)
        p = np.float64(85000.0)

        _, rh = humidity_mod.adjust_humidity(q, t, p)

        # Manual calculation
        mr = 0.005 / (1.0 - 0.005)
        e = mr * 85000.0 / (0.62197 + mr)
        e_sat = 611.2 * np.exp(17.67 * (280.0 - 273.15) / (280.0 - 29.65))
        expected_rh = np.clip(e / e_sat, 0.0, 1.0)

        np.testing.assert_allclose(float(rh), expected_rh, rtol=RTOL)
