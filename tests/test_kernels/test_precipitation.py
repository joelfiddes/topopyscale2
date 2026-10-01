"""Cross-backend precipitation kernel tests."""

import numpy as np

from tests.conftest import ATOL, RTOL

# Must match the constants used inside the wet-bulb kernels.
_CP_AIR = 1004.0
_EPSILON = 0.622
_LV = 2.501e6


def _es(t_k):
    """Saturation vapour pressure [Pa] (same form as the kernels)."""
    return 611.2 * np.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))


def _q_from_rh(t_k, p_pa, rh):
    """Specific humidity [kg/kg] for given air temp, pressure and RH (0-1)."""
    e = rh * _es(t_k)
    return _EPSILON * e / (p_pa - (1.0 - _EPSILON) * e)


class TestElevationGradient:
    def test_positive_gradient(self, precipitation_mod):
        """Higher elevation → more precipitation."""
        result = precipitation_mod.elevation_gradient(
            np.float64(0.01), np.float64(2000.0), np.float64(1000.0), np.float64(0.0003)
        )
        expected = 0.01 * (1.0 + 0.0003 * 1000.0)
        np.testing.assert_allclose(float(result), expected, rtol=RTOL)

    def test_zero_gradient(self, precipitation_mod):
        """Zero gradient → unchanged."""
        result = precipitation_mod.elevation_gradient(
            np.float64(0.01), np.float64(2000.0), np.float64(1000.0), np.float64(0.0)
        )
        np.testing.assert_allclose(float(result), 0.01, rtol=RTOL)

    def test_clamp_negative(self, precipitation_mod):
        """Strong negative gradient shouldn't produce negative precip."""
        result = precipitation_mod.elevation_gradient(
            np.float64(0.001), np.float64(0.0), np.float64(5000.0), np.float64(0.001)
        )
        assert float(result) >= 0.0

    def test_arrays(self, precipitation_mod):
        p = np.array([0.01, 0.005, 0.02])
        z_unit = np.array([2000.0, 1500.0, 3000.0])
        z_source = np.array([1000.0, 1000.0, 1000.0])
        gradient = np.array([0.0003, 0.0003, 0.0003])

        result = precipitation_mod.elevation_gradient(p, z_unit, z_source, gradient)
        expected = np.maximum(p * (1.0 + gradient * (z_unit - z_source)), 0.0)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=RTOL)


class TestPhasePartition:
    def test_all_rain(self, precipitation_mod):
        """Temperature above t_rain → all rain."""
        rain, snow = precipitation_mod.phase_partition(
            np.float64(0.01), np.float64(278.15), np.float64(275.15), np.float64(271.15)
        )
        np.testing.assert_allclose(float(rain), 0.01, rtol=RTOL)
        np.testing.assert_allclose(float(snow), 0.0, atol=ATOL)

    def test_all_snow(self, precipitation_mod):
        """Temperature below t_snow → all snow."""
        rain, snow = precipitation_mod.phase_partition(
            np.float64(0.01), np.float64(268.15), np.float64(275.15), np.float64(271.15)
        )
        np.testing.assert_allclose(float(rain), 0.0, atol=ATOL)
        np.testing.assert_allclose(float(snow), 0.01, rtol=RTOL)

    def test_mixed(self, precipitation_mod):
        """Temperature between thresholds → linear mix."""
        t_rain = np.float64(275.15)
        t_snow = np.float64(271.15)
        temperature = np.float64(273.15)  # Midpoint
        p = np.float64(0.01)

        rain, snow = precipitation_mod.phase_partition(p, temperature, t_rain, t_snow)
        rain_fraction = (273.15 - 271.15) / (275.15 - 271.15)  # 0.5
        np.testing.assert_allclose(float(rain), p * rain_fraction, rtol=RTOL)
        np.testing.assert_allclose(float(snow), p * (1 - rain_fraction), rtol=RTOL)

    def test_exact_threshold_rain(self, precipitation_mod):
        """Temperature exactly at t_rain → all rain."""
        rain, snow = precipitation_mod.phase_partition(
            np.float64(0.01), np.float64(275.15), np.float64(275.15), np.float64(271.15)
        )
        np.testing.assert_allclose(float(rain), 0.01, rtol=RTOL)
        np.testing.assert_allclose(float(snow), 0.0, atol=ATOL)

    def test_exact_threshold_snow(self, precipitation_mod):
        """Temperature exactly at t_snow → all snow."""
        rain, snow = precipitation_mod.phase_partition(
            np.float64(0.01), np.float64(271.15), np.float64(275.15), np.float64(271.15)
        )
        np.testing.assert_allclose(float(rain), 0.0, atol=ATOL)
        np.testing.assert_allclose(float(snow), 0.01, rtol=RTOL)

    def test_conservation(self, precipitation_mod):
        """Rain + snow = total."""
        p = np.array([0.01, 0.005, 0.02, 0.001])
        t = np.array([280.0, 270.0, 273.0, 275.0])
        t_rain = np.full(4, 275.15)
        t_snow = np.full(4, 271.15)

        rain, snow = precipitation_mod.phase_partition(p, t, t_rain, t_snow)
        np.testing.assert_allclose(
            np.asarray(rain) + np.asarray(snow), p, rtol=RTOL
        )


class TestWetBulbTemperature:
    def test_saturated_equals_air_temp(self, precipitation_mod):
        """At saturation (RH=100%) the wet-bulb equals the air temperature."""
        t, p = 283.15, 90000.0
        q = _q_from_rh(t, p, 1.0)
        tw = precipitation_mod.wet_bulb_temperature(
            np.float64(t), np.float64(p), np.float64(q)
        )
        np.testing.assert_allclose(float(tw), t, atol=1e-3)

    def test_dry_below_air_temp(self, precipitation_mod):
        """Dry air → wet-bulb meaningfully below air temperature."""
        t, p = 280.0, 90000.0
        q = _q_from_rh(t, p, 0.30)
        tw = float(
            precipitation_mod.wet_bulb_temperature(
                np.float64(t), np.float64(p), np.float64(q)
            )
        )
        assert t - 20.0 < tw < t - 1.0

    def test_never_above_air_temp(self, precipitation_mod):
        """Supersaturated input is clamped to <= air temperature."""
        t, p = 275.0, 90000.0
        q = _q_from_rh(t, p, 1.2)  # RH > 100%
        tw = float(
            precipitation_mod.wet_bulb_temperature(
                np.float64(t), np.float64(p), np.float64(q)
            )
        )
        assert tw <= t + 1e-9

    def test_satisfies_psychrometer_equation(self, precipitation_mod):
        """Solved Tw satisfies e = e_s(Tw) - gamma*(T - Tw) across p and RH."""
        t = np.array([285.0, 278.0, 271.0, 290.0])
        p = np.array([95000.0, 80000.0, 60000.0, 101325.0])
        rh = np.array([0.4, 0.6, 0.8, 0.5])
        q = _q_from_rh(t, p, rh)
        e = rh * _es(t)

        tw = np.asarray(
            precipitation_mod.wet_bulb_temperature(t, p, q)
        )
        gamma = _CP_AIR * p / (_EPSILON * _LV)
        e_recovered = _es(tw) - gamma * (t - tw)
        np.testing.assert_allclose(e_recovered, e, rtol=1e-6)
