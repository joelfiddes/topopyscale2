"""Cross-backend wind kernel tests."""

import numpy as np
import pytest

from tests.conftest import ATOL, RTOL


class TestLogProfileCorrection:
    """Tests for the log-profile wind correction kernel."""

    def test_scalar_known_value(self, wind_mod):
        """Test with known analytical result.

        U_target = U_source * ln((z_target - d_target) / z0_target) / ln((z_source - d_source) / z0_source)

        U_source = 10 m/s
        z_source = 10 m (standard 10m wind measurement height)
        z_target = 2 m (target 2m height)
        z0_source = 0.01 m (open terrain)
        z0_target = 0.01 m (same roughness)
        d_source = 0 m (no displacement)
        d_target = 0 m (no displacement)

        ln(10/0.01) = ln(1000) = 6.9078
        ln(2/0.01) = ln(200) = 5.2983
        U_target = 10 * 5.2983 / 6.9078 = 7.672
        """
        result = wind_mod.log_profile_correction(
            np.float64(10.0),  # u_source
            np.float64(10.0),  # z_source
            np.float64(2.0),   # z_target
            np.float64(0.01),  # z0_source
            np.float64(0.01),  # z0_target
            np.float64(0.0),   # d_source
            np.float64(0.0),   # d_target
        )
        expected = 10.0 * np.log(2.0 / 0.01) / np.log(10.0 / 0.01)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_same_height_same_roughness(self, wind_mod):
        """Same heights and roughness should return same wind speed."""
        result = wind_mod.log_profile_correction(
            np.float64(5.0),   # u_source
            np.float64(10.0),  # z_source
            np.float64(10.0),  # z_target (same as source)
            np.float64(0.03),  # z0_source
            np.float64(0.03),  # z0_target (same)
            np.float64(0.0),   # d_source
            np.float64(0.0),   # d_target
        )
        np.testing.assert_allclose(result, 5.0, rtol=RTOL, atol=ATOL)

    def test_higher_target_increases_speed(self, wind_mod):
        """Wind should increase at higher heights (same roughness, no displacement)."""
        u_source = 5.0
        z_source = 2.0
        z_target = 10.0  # higher than source
        z0 = 0.01
        d = 0.0

        result = wind_mod.log_profile_correction(
            np.float64(u_source),
            np.float64(z_source),
            np.float64(z_target),
            np.float64(z0),
            np.float64(z0),
            np.float64(d),
            np.float64(d),
        )
        # Higher target should give higher wind speed
        assert result > u_source

    def test_lower_target_decreases_speed(self, wind_mod):
        """Wind should decrease at lower heights (same roughness, no displacement)."""
        u_source = 10.0
        z_source = 10.0
        z_target = 2.0  # lower than source
        z0 = 0.01
        d = 0.0

        result = wind_mod.log_profile_correction(
            np.float64(u_source),
            np.float64(z_source),
            np.float64(z_target),
            np.float64(z0),
            np.float64(z0),
            np.float64(d),
            np.float64(d),
        )
        # Lower target should give lower wind speed
        assert result < u_source

    def test_rougher_target_reduces_speed(self, wind_mod):
        """Rougher target surface should reduce wind speed at same height."""
        u_source = 10.0
        z = 10.0  # same height
        d = 0.0

        # Smooth source, rough target
        result = wind_mod.log_profile_correction(
            np.float64(u_source),
            np.float64(z),
            np.float64(z),
            np.float64(0.01),   # smooth source (z0=1cm)
            np.float64(1.0),    # rough target (z0=1m, forest)
            np.float64(d),
            np.float64(d),
        )
        # Rougher surface = lower wind
        assert result < u_source

    def test_smoother_target_increases_speed(self, wind_mod):
        """Smoother target surface should increase wind speed at same height."""
        u_source = 5.0
        z = 10.0  # same height
        d = 0.0

        # Rough source, smooth target
        result = wind_mod.log_profile_correction(
            np.float64(u_source),
            np.float64(z),
            np.float64(z),
            np.float64(1.0),    # rough source (z0=1m, forest)
            np.float64(0.01),   # smooth target (z0=1cm)
            np.float64(d),
            np.float64(d),
        )
        # Smoother surface = higher wind
        assert result > u_source

    def test_displacement_reduces_effective_height(self, wind_mod):
        """Displacement height should reduce effective measurement height."""
        u_source = 10.0
        z0 = 0.1
        z_source = 20.0
        z_target = 20.0  # same height

        # With displacement at target, effective height is lower -> lower wind
        result_with_d = wind_mod.log_profile_correction(
            np.float64(u_source),
            np.float64(z_source),
            np.float64(z_target),
            np.float64(z0),
            np.float64(z0),
            np.float64(0.0),   # no displacement at source
            np.float64(10.0),  # displacement at target (forest canopy)
        )

        result_no_d = wind_mod.log_profile_correction(
            np.float64(u_source),
            np.float64(z_source),
            np.float64(z_target),
            np.float64(z0),
            np.float64(z0),
            np.float64(0.0),   # no displacement at source
            np.float64(0.0),   # no displacement at target
        )

        # With displacement, effective target height is lower -> lower wind
        assert result_with_d < result_no_d

    def test_edge_case_z_equals_d_plus_z0(self, wind_mod):
        """When z == d + z0, should return 0 (invalid log argument)."""
        result = wind_mod.log_profile_correction(
            np.float64(10.0),  # u_source
            np.float64(10.0),  # z_source
            np.float64(1.1),   # z_target = d_target + z0_target = 1.0 + 0.1
            np.float64(0.1),   # z0_source
            np.float64(0.1),   # z0_target
            np.float64(0.0),   # d_source
            np.float64(1.0),   # d_target
        )
        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_edge_case_z_below_d_plus_z0(self, wind_mod):
        """When z < d + z0, should return 0 (non-physical)."""
        result = wind_mod.log_profile_correction(
            np.float64(10.0),  # u_source
            np.float64(10.0),  # z_source
            np.float64(0.5),   # z_target < d_target + z0_target
            np.float64(0.1),   # z0_source
            np.float64(0.1),   # z0_target
            np.float64(0.0),   # d_source
            np.float64(1.0),   # d_target
        )
        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_edge_case_source_invalid(self, wind_mod):
        """When source height is invalid, should return 0."""
        result = wind_mod.log_profile_correction(
            np.float64(10.0),  # u_source
            np.float64(0.05),  # z_source < z0_source (invalid)
            np.float64(10.0),  # z_target (valid)
            np.float64(0.1),   # z0_source
            np.float64(0.1),   # z0_target
            np.float64(0.0),   # d_source
            np.float64(0.0),   # d_target
        )
        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_zero_wind_stays_zero(self, wind_mod):
        """Zero wind speed should remain zero."""
        result = wind_mod.log_profile_correction(
            np.float64(0.0),   # u_source = 0
            np.float64(10.0),  # z_source
            np.float64(2.0),   # z_target
            np.float64(0.01),  # z0_source
            np.float64(0.01),  # z0_target
            np.float64(0.0),   # d_source
            np.float64(0.0),   # d_target
        )
        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_arrays_1d(self, wind_mod):
        """Test with 1D arrays."""
        u_source = np.array([5.0, 10.0, 15.0])
        z_source = np.array([10.0, 10.0, 10.0])
        z_target = np.array([2.0, 10.0, 20.0])
        z0_source = np.array([0.01, 0.01, 0.01])
        z0_target = np.array([0.01, 0.01, 0.01])
        d_source = np.array([0.0, 0.0, 0.0])
        d_target = np.array([0.0, 0.0, 0.0])

        result = wind_mod.log_profile_correction(
            u_source, z_source, z_target, z0_source, z0_target, d_source, d_target
        )

        # Manual calculation for expected values
        h_source = z_source - d_source
        h_target = z_target - d_target
        expected = u_source * np.log(h_target / z0_target) / np.log(h_source / z0_source)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_arrays_2d(self, wind_mod):
        """Test with 2D arrays (spatial grid)."""
        shape = (3, 4)
        u_source = np.full(shape, 10.0)
        z_source = np.full(shape, 10.0)
        z_target = np.full(shape, 2.0)
        z0_source = np.full(shape, 0.01)
        z0_target = np.full(shape, 0.01)
        d_source = np.full(shape, 0.0)
        d_target = np.full(shape, 0.0)

        result = wind_mod.log_profile_correction(
            u_source, z_source, z_target, z0_source, z0_target, d_source, d_target
        )

        expected = 10.0 * np.log(2.0 / 0.01) / np.log(10.0 / 0.01)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_mixed_valid_invalid_array(self, wind_mod):
        """Test array with mix of valid and invalid points."""
        u_source = np.array([10.0, 10.0, 10.0])
        z_source = np.array([10.0, 10.0, 10.0])
        z_target = np.array([5.0, 1.0, 0.5])  # first two valid, last invalid
        z0_source = np.array([0.1, 0.1, 0.1])
        z0_target = np.array([0.1, 0.1, 0.1])
        d_source = np.array([0.0, 0.0, 0.0])
        d_target = np.array([0.0, 0.5, 0.5])  # last point: z_target - d_target = 0 <= z0

        result = wind_mod.log_profile_correction(
            u_source, z_source, z_target, z0_source, z0_target, d_source, d_target
        )

        # First point is valid
        expected_0 = 10.0 * np.log(5.0 / 0.1) / np.log(10.0 / 0.1)
        np.testing.assert_allclose(result[0], expected_0, rtol=RTOL, atol=ATOL)

        # Second point is valid
        expected_1 = 10.0 * np.log(0.5 / 0.1) / np.log(10.0 / 0.1)
        np.testing.assert_allclose(result[1], expected_1, rtol=RTOL, atol=ATOL)

        # Third point is invalid (z_target - d_target = 0.0, not > z0)
        np.testing.assert_allclose(result[2], 0.0, rtol=RTOL, atol=ATOL)

    def test_typical_era5_to_2m(self, wind_mod):
        """Realistic test: ERA5 10m wind to 2m height."""
        # ERA5 uses 10m reference height, open terrain
        u_10m = np.float64(8.0)  # 8 m/s at 10m
        z_era5 = np.float64(10.0)
        z_target = np.float64(2.0)
        z0_open = np.float64(0.03)  # short grass

        result = wind_mod.log_profile_correction(
            u_10m, z_era5, z_target, z0_open, z0_open,
            np.float64(0.0), np.float64(0.0)
        )

        # Should be less than 10m wind
        assert result < u_10m
        # ln(2/0.03)/ln(10/0.03) = 4.199/5.809 = 0.723
        expected = 8.0 * np.log(2.0 / 0.03) / np.log(10.0 / 0.03)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_forest_canopy_adjustment(self, wind_mod):
        """Test wind adjustment for forest with displacement height."""
        u_source = np.float64(10.0)
        z_source = np.float64(50.0)  # measurement above canopy
        z_target = np.float64(30.0)  # target within/just above canopy
        z0_forest = np.float64(1.5)   # forest roughness
        d_forest = np.float64(20.0)   # displacement height (2/3 of tree height)

        result = wind_mod.log_profile_correction(
            u_source, z_source, z_target,
            z0_forest, z0_forest,
            d_forest, d_forest
        )

        # h_source = 50 - 20 = 30m
        # h_target = 30 - 20 = 10m
        # ln(10/1.5)/ln(30/1.5) = 1.897/2.996 = 0.633
        expected = 10.0 * np.log(10.0 / 1.5) / np.log(30.0 / 1.5)
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestWinstralWindCorrection:
    """Tests for the Winstral Sx-based wind exposure/sheltering correction."""

    def test_positive_sx_increases_wind(self, wind_mod):
        """Positive Sx (exposed terrain) should increase wind speed."""
        u_log = np.float64(10.0)
        sx = np.float64(15.0)  # exposed

        result = wind_mod.winstral_wind_correction(u_log, sx)

        # With default sx_scale=0.5, sx_ref=15.0:
        # factor = 1 + 0.5 * tanh(15/15) = 1 + 0.5 * tanh(1) ≈ 1 + 0.5 * 0.762 = 1.381
        assert result > u_log

    def test_negative_sx_decreases_wind(self, wind_mod):
        """Negative Sx (sheltered terrain) should decrease wind speed."""
        u_log = np.float64(10.0)
        sx = np.float64(-15.0)  # sheltered

        result = wind_mod.winstral_wind_correction(u_log, sx)

        # With default sx_scale=0.5, sx_ref=15.0:
        # factor = 1 + 0.5 * tanh(-15/15) = 1 - 0.5 * tanh(1) ≈ 1 - 0.381 = 0.619
        assert result < u_log

    def test_zero_sx_unchanged(self, wind_mod):
        """Zero Sx should leave wind unchanged."""
        u_log = np.float64(10.0)
        sx = np.float64(0.0)

        result = wind_mod.winstral_wind_correction(u_log, sx)

        # factor = 1 + 0.5 * tanh(0) = 1 + 0 = 1.0
        np.testing.assert_allclose(result, u_log, rtol=RTOL, atol=ATOL)

    def test_known_value(self, wind_mod):
        """Test with known analytical result."""
        u_log = np.float64(10.0)
        sx = np.float64(15.0)
        sx_scale = 0.5
        sx_ref = 15.0

        result = wind_mod.winstral_wind_correction(u_log, sx, sx_scale, sx_ref)

        # factor = 1 + 0.5 * tanh(1) = 1 + 0.5 * 0.76159416 = 1.38079708
        expected = 10.0 * (1.0 + 0.5 * np.tanh(1.0))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_sx_scale_effect(self, wind_mod):
        """Larger sx_scale should amplify the correction."""
        u_log = np.float64(10.0)
        sx = np.float64(15.0)
        sx_ref = 15.0

        result_low_scale = wind_mod.winstral_wind_correction(u_log, sx, sx_scale=0.2, sx_ref=sx_ref)
        result_high_scale = wind_mod.winstral_wind_correction(u_log, sx, sx_scale=0.8, sx_ref=sx_ref)

        # Higher scale should give larger correction
        # Both should be > u_log since Sx > 0
        assert u_log < result_low_scale < result_high_scale

    def test_sx_ref_effect(self, wind_mod):
        """Larger sx_ref should reduce sensitivity to Sx."""
        u_log = np.float64(10.0)
        sx = np.float64(15.0)
        sx_scale = 0.5

        result_low_ref = wind_mod.winstral_wind_correction(u_log, sx, sx_scale, sx_ref=10.0)
        result_high_ref = wind_mod.winstral_wind_correction(u_log, sx, sx_scale, sx_ref=30.0)

        # Lower sx_ref -> stronger response (closer to saturation)
        # Both should be > u_log since Sx > 0
        # tanh(15/10) > tanh(15/30), so low_ref gives bigger enhancement
        assert result_low_ref > result_high_ref > u_log

    def test_saturation_large_positive_sx(self, wind_mod):
        """Very large positive Sx should saturate at (1 + sx_scale) factor."""
        u_log = np.float64(10.0)
        sx = np.float64(1000.0)  # very exposed
        sx_scale = 0.5
        sx_ref = 15.0

        result = wind_mod.winstral_wind_correction(u_log, sx, sx_scale, sx_ref)

        # tanh(1000/15) ≈ 1.0 (saturated)
        # factor ≈ 1 + 0.5 * 1.0 = 1.5
        expected_max = u_log * (1.0 + sx_scale)
        np.testing.assert_allclose(result, expected_max, rtol=1e-6, atol=1e-6)

    def test_saturation_large_negative_sx(self, wind_mod):
        """Very large negative Sx should saturate at (1 - sx_scale) factor."""
        u_log = np.float64(10.0)
        sx = np.float64(-1000.0)  # very sheltered
        sx_scale = 0.5
        sx_ref = 15.0

        result = wind_mod.winstral_wind_correction(u_log, sx, sx_scale, sx_ref)

        # tanh(-1000/15) ≈ -1.0 (saturated)
        # factor ≈ 1 + 0.5 * (-1.0) = 0.5
        expected_min = u_log * (1.0 - sx_scale)
        np.testing.assert_allclose(result, expected_min, rtol=1e-6, atol=1e-6)

    def test_zero_wind_stays_zero(self, wind_mod):
        """Zero wind speed should remain zero regardless of Sx."""
        u_log = np.float64(0.0)
        sx = np.float64(30.0)

        result = wind_mod.winstral_wind_correction(u_log, sx)

        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_arrays_1d(self, wind_mod):
        """Test with 1D arrays."""
        u_log = np.array([5.0, 10.0, 15.0])
        sx = np.array([-15.0, 0.0, 15.0])

        result = wind_mod.winstral_wind_correction(u_log, sx)

        expected = u_log * (1.0 + 0.5 * np.tanh(sx / 15.0))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_arrays_2d(self, wind_mod):
        """Test with 2D arrays (spatial grid)."""
        shape = (3, 4)
        u_log = np.full(shape, 10.0)
        sx = np.linspace(-30, 30, 12).reshape(shape)

        result = wind_mod.winstral_wind_correction(u_log, sx)

        expected = u_log * (1.0 + 0.5 * np.tanh(sx / 15.0))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_custom_parameters(self, wind_mod):
        """Test with custom sx_scale and sx_ref."""
        u_log = np.array([10.0, 10.0, 10.0])
        sx = np.array([10.0, 20.0, 30.0])
        sx_scale = 0.3
        sx_ref = 20.0

        result = wind_mod.winstral_wind_correction(u_log, sx, sx_scale, sx_ref)

        expected = u_log * (1.0 + sx_scale * np.tanh(sx / sx_ref))
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_symmetric_response(self, wind_mod):
        """Positive and negative Sx of same magnitude should give symmetric response."""
        u_log = np.float64(10.0)
        sx_pos = np.float64(20.0)
        sx_neg = np.float64(-20.0)

        result_pos = wind_mod.winstral_wind_correction(u_log, sx_pos)
        result_neg = wind_mod.winstral_wind_correction(u_log, sx_neg)

        # result_pos - u_log should equal u_log - result_neg
        enhancement = result_pos - u_log
        reduction = u_log - result_neg
        np.testing.assert_allclose(enhancement, reduction, rtol=RTOL, atol=ATOL)


class TestCrossBackendWind:
    """Verify all backends produce identical results."""

    def test_agreement_log_profile(self):
        """All backends should produce identical log_profile_correction results."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        # Test data covering various scenarios
        u_source = np.array([5.0, 10.0, 15.0, 8.0, 0.0, 12.0])
        z_source = np.array([10.0, 10.0, 20.0, 50.0, 10.0, 10.0])
        z_target = np.array([2.0, 10.0, 10.0, 30.0, 2.0, 0.5])
        z0_source = np.array([0.01, 0.03, 0.1, 1.5, 0.01, 0.1])
        z0_target = np.array([0.01, 0.03, 0.03, 1.5, 0.01, 0.1])
        d_source = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.0])
        d_target = np.array([0.0, 0.0, 0.0, 20.0, 0.0, 0.5])  # last is invalid

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].log_profile_correction(
                    u_source, z_source, z_target,
                    z0_source, z0_target, d_source, d_target
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"log_profile_correction mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_edge_cases(self):
        """All backends should handle edge cases identically."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        # Edge cases: boundary conditions, zeros, invalid heights
        u_source = np.array([10.0, 10.0, 10.0, 0.0])
        z_source = np.array([10.0, 0.05, 10.0, 10.0])  # second is invalid
        z_target = np.array([0.5, 10.0, 1.1, 2.0])     # first/third are edge
        z0_source = np.array([0.1, 0.1, 0.1, 0.01])
        z0_target = np.array([0.1, 0.1, 0.1, 0.01])
        d_source = np.array([0.0, 0.0, 0.0, 0.0])
        d_target = np.array([0.5, 0.0, 1.0, 0.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].log_profile_correction(
                    u_source, z_source, z_target,
                    z0_source, z0_target, d_source, d_target
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"Edge cases mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_2d_arrays(self):
        """All backends should handle 2D arrays identically."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        shape = (5, 7)
        rng = np.random.default_rng(42)

        u_source = rng.uniform(1.0, 20.0, shape)
        z_source = np.full(shape, 10.0)
        z_target = rng.uniform(2.0, 50.0, shape)
        z0_source = np.full(shape, 0.03)
        z0_target = rng.uniform(0.01, 0.5, shape)
        d_source = np.full(shape, 0.0)
        d_target = np.full(shape, 0.0)

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].log_profile_correction(
                    u_source, z_source, z_target,
                    z0_source, z0_target, d_source, d_target
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"2D array mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_winstral_basic(self):
        """All backends should produce identical winstral_wind_correction results."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        # Test data covering various Sx values
        u_log = np.array([5.0, 10.0, 15.0, 8.0, 0.0, 12.0])
        sx = np.array([-30.0, -15.0, 0.0, 15.0, 30.0, 45.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].winstral_wind_correction(u_log, sx)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"winstral_wind_correction mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_winstral_custom_params(self):
        """All backends should handle custom sx_scale and sx_ref identically."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        u_log = np.array([5.0, 10.0, 15.0, 20.0])
        sx = np.array([-20.0, -5.0, 10.0, 25.0])
        sx_scale = 0.3
        sx_ref = 20.0

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].winstral_wind_correction(u_log, sx, sx_scale, sx_ref)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"winstral custom params mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_winstral_2d_arrays(self):
        """All backends should handle 2D arrays identically for Winstral."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        shape = (5, 7)
        rng = np.random.default_rng(42)

        u_log = rng.uniform(1.0, 20.0, shape)
        sx = rng.uniform(-45.0, 45.0, shape)

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].winstral_wind_correction(u_log, sx)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"Winstral 2D array mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_winstral_saturation(self):
        """All backends should handle saturation cases identically."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        # Test extreme Sx values that should saturate tanh
        u_log = np.array([10.0, 10.0, 10.0, 10.0])
        sx = np.array([-1000.0, -100.0, 100.0, 1000.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["wind"].winstral_wind_correction(u_log, sx)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"Winstral saturation mismatch between {names[0]} and {names[i]}",
            )
