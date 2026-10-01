"""Cross-backend redistribution kernel tests."""

import numpy as np
import pytest

from tests.conftest import ATOL, RTOL


class TestWindTransport:
    """Tests for the wind_transport kernel."""

    def test_no_erosion_when_no_exposure(self, redistribution_mod):
        """No erosion should occur when all cells are sheltered (Sx <= 0)."""
        swe = np.array([100.0, 100.0, 100.0, 100.0])
        sx = np.array([-10.0, -5.0, 0.0, -15.0])  # All sheltered or neutral
        wind_speed = np.array([10.0, 10.0, 10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        # Simple ring topology: each cell connects to its neighbors
        neighbor_indices = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        # No change - no exposure means no erosion
        np.testing.assert_allclose(result, swe, rtol=RTOL, atol=ATOL)

    def test_erosion_from_exposed_cells(self, redistribution_mod):
        """Exposed cells (Sx > 0) should have snow eroded."""
        swe = np.array([100.0, 100.0, 100.0, 100.0])
        sx = np.array([20.0, -10.0, 0.0, -5.0])  # Only first cell is exposed
        wind_speed = np.array([10.0, 10.0, 10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        neighbor_indices = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        # First cell should have less snow
        assert result[0] < swe[0]
        # Other cells should have equal or more
        assert result[1] >= swe[1]
        assert result[3] >= swe[3]

    def test_mass_conservation(self, redistribution_mod):
        """Total mass (SWE * area) should be conserved."""
        swe = np.array([150.0, 50.0, 100.0, 200.0])
        sx = np.array([20.0, -10.0, 5.0, -15.0])  # Mix of exposed and sheltered
        wind_speed = np.array([8.0, 12.0, 6.0, 10.0])
        transport_coeff = 0.005
        area = np.array([1000.0, 1500.0, 800.0, 1200.0])
        neighbor_indices = np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64)
        neighbor_weights = np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        mass_before = np.sum(swe * area)
        mass_after = np.sum(result * area)

        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-10, atol=1e-10)

    def test_deposition_prefers_sheltered(self, redistribution_mod):
        """Eroded snow should preferentially deposit on sheltered cells."""
        swe = np.array([200.0, 50.0, 50.0])
        sx = np.array([30.0, -20.0, -5.0])  # Cell 0 exposed, cells 1 and 2 sheltered
        wind_speed = np.array([10.0, 10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0, 1000.0])
        # Cell 0 has both sheltered neighbors
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        # Cell 1 (more sheltered, Sx=-20) should receive more than cell 2 (Sx=-5)
        deposit_1 = result[1] - swe[1]
        deposit_2 = result[2] - swe[2]
        assert deposit_1 > deposit_2

    def test_no_snow_no_erosion(self, redistribution_mod):
        """Cells with no snow should not contribute to erosion."""
        swe = np.array([0.0, 100.0, 50.0])
        sx = np.array([30.0, -10.0, 20.0])  # Cells 0 and 2 are exposed
        wind_speed = np.array([10.0, 10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0, 1000.0])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        # Cell 0 stays at 0
        np.testing.assert_allclose(result[0], 0.0, rtol=RTOL, atol=ATOL)

    def test_zero_wind_no_transport(self, redistribution_mod):
        """No transport should occur with zero wind speed."""
        swe = np.array([100.0, 100.0, 100.0])
        sx = np.array([30.0, -10.0, 20.0])
        wind_speed = np.array([0.0, 0.0, 0.0])  # No wind
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0, 1000.0])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        np.testing.assert_allclose(result, swe, rtol=RTOL, atol=ATOL)

    def test_zero_transport_coeff(self, redistribution_mod):
        """Zero transport coefficient means no transport."""
        swe = np.array([100.0, 100.0, 100.0])
        sx = np.array([30.0, -10.0, 20.0])
        wind_speed = np.array([10.0, 10.0, 10.0])
        transport_coeff = 0.0  # No transport
        area = np.array([1000.0, 1000.0, 1000.0])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        np.testing.assert_allclose(result, swe, rtol=RTOL, atol=ATOL)

    def test_varying_cell_areas(self, redistribution_mod):
        """Mass conservation should work with varying cell areas."""
        swe = np.array([100.0, 100.0, 100.0])
        sx = np.array([20.0, -15.0, -5.0])
        wind_speed = np.array([10.0, 10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([500.0, 1000.0, 2000.0])  # Different areas
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        mass_before = np.sum(swe * area)
        mass_after = np.sum(result * area)
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-10, atol=1e-10)

    def test_no_neighbors_edge_case(self, redistribution_mod):
        """Cells with no valid neighbors should still work."""
        swe = np.array([100.0, 100.0])
        sx = np.array([20.0, -10.0])
        wind_speed = np.array([10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0])
        # Cell 0 has neighbor, cell 1 has no valid neighbors
        neighbor_indices = np.array([[1, -1], [-1, -1]], dtype=np.int64)
        neighbor_weights = np.array([[1.0, 0.0], [0.0, 0.0]])

        result = redistribution_mod.wind_transport(
            swe, sx, wind_speed, transport_coeff, area, neighbor_indices, neighbor_weights
        )

        # Should not crash, mass should be conserved
        mass_before = np.sum(swe * area)
        mass_after = np.sum(result * area)
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-10, atol=1e-10)


class TestAvalancheRedistribute:
    """Tests for the avalanche_redistribute kernel."""

    def test_no_avalanche_below_threshold(self, redistribution_mod):
        """No avalanche should occur when all slopes are below threshold."""
        swe = np.array([100.0, 100.0, 100.0])
        slope = np.array([20.0, 25.0, 28.0])  # All below 30 degrees
        slope_threshold = 30.0
        flow_fractions = np.array([[0.7, 0.3], [0.5, 0.5], [0.6, 0.4]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0, 1000.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        np.testing.assert_allclose(result, swe, rtol=RTOL, atol=ATOL)

    def test_avalanche_above_threshold(self, redistribution_mod):
        """Cells above slope threshold should redistribute snow."""
        swe = np.array([100.0, 50.0, 50.0])
        slope = np.array([45.0, 20.0, 25.0])  # First cell steep
        slope_threshold = 30.0
        flow_fractions = np.array([[0.7, 0.3], [0.5, 0.5], [0.6, 0.4]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0, 1000.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        # Cell 0 should have less snow
        assert result[0] < swe[0]
        # Cells 1 and 2 should have more
        assert result[1] > swe[1]
        assert result[2] > swe[2]

    def test_mass_conservation(self, redistribution_mod):
        """Total mass should be conserved during avalanche."""
        swe = np.array([200.0, 50.0, 100.0, 75.0])
        slope = np.array([50.0, 40.0, 20.0, 55.0])
        slope_threshold = 35.0
        flow_fractions = np.array([
            [0.6, 0.4, 0.0],
            [0.5, 0.5, 0.0],
            [0.3, 0.7, 0.0],
            [0.4, 0.3, 0.3]
        ])
        neighbor_indices = np.array([
            [1, 2, -1],
            [0, 2, -1],
            [0, 1, -1],
            [0, 1, 2]
        ], dtype=np.int64)
        area = np.array([1000.0, 1200.0, 800.0, 1500.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        mass_before = np.sum(swe * area)
        mass_after = np.sum(result * area)
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-10, atol=1e-10)

    def test_redistribute_fraction_increases_with_slope(self, redistribution_mod):
        """Steeper slopes should redistribute more snow."""
        swe = np.array([100.0, 100.0, 50.0])
        slope1 = np.array([35.0, 20.0, 20.0])  # Just above threshold
        slope2 = np.array([60.0, 20.0, 20.0])  # Much above threshold (2x)
        slope_threshold = 30.0
        flow_fractions = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0, 1000.0])

        result1 = redistribution_mod.avalanche_redistribute(
            swe, slope1, slope_threshold, flow_fractions, neighbor_indices, area
        )
        result2 = redistribution_mod.avalanche_redistribute(
            swe, slope2, slope_threshold, flow_fractions, neighbor_indices, area
        )

        # More snow removed from cell 0 with steeper slope
        removed1 = swe[0] - result1[0]
        removed2 = swe[0] - result2[0]
        assert removed2 > removed1

    def test_flow_fractions_respected(self, redistribution_mod):
        """Snow should be distributed according to flow fractions."""
        swe = np.array([100.0, 0.0, 0.0])
        slope = np.array([60.0, 10.0, 10.0])  # Cell 0 steep
        slope_threshold = 30.0
        # 70% to cell 1, 30% to cell 2
        flow_fractions = np.array([[0.7, 0.3], [0.5, 0.5], [0.5, 0.5]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0, 1000.0])  # Same areas

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        # Cell 1 should receive ~70% of redistributed snow
        total_deposited = result[1] + result[2]
        if total_deposited > 0:
            ratio = result[1] / total_deposited
            np.testing.assert_allclose(ratio, 0.7, rtol=0.01, atol=0.01)

    def test_no_snow_no_avalanche(self, redistribution_mod):
        """Cells with no snow should not contribute to avalanche."""
        swe = np.array([0.0, 100.0, 50.0])
        slope = np.array([60.0, 20.0, 45.0])  # Cells 0 and 2 steep
        slope_threshold = 30.0
        # Cell 2 deposits to cells 0 and 1 (both are its neighbors)
        flow_fractions = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0, 1000.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        # Cell 0 started with 0 but receives snow from cell 2 (which is steep)
        # Cell 2 slope excess = 45 - 30 = 15, fraction = 15/30 = 0.5
        # Cell 2 moves 0.5 * 50 = 25 mm, split 50/50 to cells 0 and 1
        # So cell 0 receives 12.5 mm
        expected_cell_0 = 0.5 * 50.0 * 0.5  # 12.5
        np.testing.assert_allclose(result[0], expected_cell_0, rtol=RTOL, atol=ATOL)

        # Mass should be conserved
        mass_before = np.sum(swe * area)
        mass_after = np.sum(result * area)
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-10, atol=1e-10)

    def test_exactly_at_threshold(self, redistribution_mod):
        """Cells exactly at threshold should not trigger avalanche."""
        swe = np.array([100.0, 50.0])
        slope = np.array([30.0, 20.0])  # Exactly at threshold
        slope_threshold = 30.0
        flow_fractions = np.array([[1.0], [1.0]])
        neighbor_indices = np.array([[1], [0]], dtype=np.int64)
        area = np.array([1000.0, 1000.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        np.testing.assert_allclose(result, swe, rtol=RTOL, atol=ATOL)

    def test_no_neighbors_stays_in_place(self, redistribution_mod):
        """Steep cells with no outflow neighbors should retain snow."""
        swe = np.array([100.0, 50.0])
        slope = np.array([50.0, 20.0])
        slope_threshold = 30.0
        # Cell 0 has no valid neighbors
        flow_fractions = np.array([[0.0, 0.0], [1.0, 0.0]])
        neighbor_indices = np.array([[-1, -1], [0, -1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        # Cell 0 retains its snow (no valid neighbors to deposit to)
        np.testing.assert_allclose(result[0], swe[0], rtol=RTOL, atol=ATOL)

    def test_varying_areas(self, redistribution_mod):
        """Mass conservation should work with varying cell areas."""
        swe = np.array([100.0, 50.0, 75.0])
        slope = np.array([50.0, 20.0, 45.0])
        slope_threshold = 30.0
        flow_fractions = np.array([[0.6, 0.4], [0.5, 0.5], [0.5, 0.5]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([500.0, 1000.0, 2000.0])

        result = redistribution_mod.avalanche_redistribute(
            swe, slope, slope_threshold, flow_fractions, neighbor_indices, area
        )

        mass_before = np.sum(swe * area)
        mass_after = np.sum(result * area)
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-10, atol=1e-10)


class TestComputeTransportRate:
    """Tests for the compute_transport_rate kernel."""

    def test_below_wind_threshold(self, redistribution_mod):
        """Transport rate should be 0 when wind is below threshold."""
        wind_speed = np.array([3.0, 4.0, 4.9])  # All below 5.0
        sx = np.array([20.0, 30.0, 40.0])
        threshold_wind = 5.0

        result = redistribution_mod.compute_transport_rate(
            wind_speed, sx, threshold_wind=threshold_wind
        )

        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_sheltered_terrain_no_transport(self, redistribution_mod):
        """Transport rate should be 0 for sheltered terrain (Sx <= 0)."""
        wind_speed = np.array([10.0, 15.0, 20.0])
        sx = np.array([-10.0, 0.0, -5.0])  # Sheltered

        result = redistribution_mod.compute_transport_rate(wind_speed, sx)

        np.testing.assert_allclose(result, 0.0, rtol=RTOL, atol=ATOL)

    def test_rate_increases_with_wind(self, redistribution_mod):
        """Transport rate should increase with wind speed."""
        wind_speed = np.array([6.0, 10.0, 15.0])
        sx = np.array([15.0, 15.0, 15.0])  # Same exposure

        result = redistribution_mod.compute_transport_rate(wind_speed, sx)

        assert result[0] < result[1] < result[2]

    def test_rate_increases_with_exposure(self, redistribution_mod):
        """Transport rate should increase with Sx (exposure)."""
        wind_speed = np.array([10.0, 10.0, 10.0])
        sx = np.array([5.0, 15.0, 30.0])  # Increasing exposure

        result = redistribution_mod.compute_transport_rate(wind_speed, sx)

        assert result[0] < result[1] < result[2]

    def test_max_rate_capped(self, redistribution_mod):
        """Transport rate should be capped at max_rate."""
        wind_speed = np.array([100.0, 200.0])  # Very high wind
        sx = np.array([45.0, 45.0])  # Very exposed
        max_rate = 0.1

        result = redistribution_mod.compute_transport_rate(
            wind_speed, sx, max_rate=max_rate
        )

        assert np.all(result <= max_rate + 1e-10)

    def test_known_value(self, redistribution_mod):
        """Test with analytically computed value."""
        wind_speed = np.array([10.0])  # 5 m/s excess above threshold
        sx = np.array([15.0])  # exposure_factor = 15/15 = 1.0
        threshold_wind = 5.0
        max_rate = 0.1

        # rate = 0.01 * (10-5) * (15/15) = 0.01 * 5 * 1 = 0.05
        expected = 0.05

        result = redistribution_mod.compute_transport_rate(
            wind_speed, sx, threshold_wind=threshold_wind, max_rate=max_rate
        )

        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_scalar_input(self, redistribution_mod):
        """Test with scalar input values."""
        wind_speed = np.float64(10.0)
        sx = np.float64(15.0)

        result = redistribution_mod.compute_transport_rate(wind_speed, sx)

        expected = 0.01 * 5.0 * 1.0  # 0.05
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)

    def test_2d_arrays(self, redistribution_mod):
        """Test with 2D arrays."""
        shape = (3, 4)
        wind_speed = np.full(shape, 10.0)
        sx = np.full(shape, 15.0)

        result = redistribution_mod.compute_transport_rate(wind_speed, sx)

        expected = 0.05
        np.testing.assert_allclose(result, expected, rtol=RTOL, atol=ATOL)


class TestCrossBackendRedistribution:
    """Verify all backends produce identical results for redistribution kernels."""

    def test_agreement_compute_transport_rate(self):
        """All backends should produce identical compute_transport_rate results."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        wind_speed = np.array([3.0, 5.0, 8.0, 12.0, 20.0, 50.0])
        sx = np.array([-10.0, 0.0, 5.0, 15.0, 30.0, 45.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["redistribution"].compute_transport_rate(wind_speed, sx)
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"compute_transport_rate mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_compute_transport_rate_custom_params(self):
        """All backends should handle custom params identically."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        wind_speed = np.array([8.0, 15.0, 25.0])
        sx = np.array([10.0, 20.0, 30.0])
        threshold_wind = 3.0
        max_rate = 0.2

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["redistribution"].compute_transport_rate(
                    wind_speed, sx, threshold_wind=threshold_wind, max_rate=max_rate
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"compute_transport_rate custom params mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_wind_transport(self):
        """All backends should produce identical wind_transport results."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        swe = np.array([100.0, 150.0, 50.0, 200.0])
        sx = np.array([20.0, -15.0, 10.0, -5.0])
        wind_speed = np.array([8.0, 12.0, 6.0, 10.0])
        transport_coeff = 0.005
        area = np.array([1000.0, 1200.0, 800.0, 1500.0])
        neighbor_indices = np.array([[1, 2], [0, 3], [0, 3], [1, 2]], dtype=np.int64)
        neighbor_weights = np.array([[0.6, 0.4], [0.5, 0.5], [0.3, 0.7], [0.4, 0.6]])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["redistribution"].wind_transport(
                    swe, sx, wind_speed, transport_coeff, area,
                    neighbor_indices, neighbor_weights
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"wind_transport mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_wind_transport_edge_cases(self):
        """All backends should handle edge cases identically for wind_transport."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        # Edge cases: zero snow, zero wind, all sheltered
        swe = np.array([0.0, 100.0, 50.0, 100.0])
        sx = np.array([30.0, -10.0, 0.0, -5.0])
        wind_speed = np.array([0.0, 10.0, 10.0, 10.0])
        transport_coeff = 0.01
        area = np.array([1000.0, 1000.0, 1000.0, 1000.0])
        neighbor_indices = np.array([[1, 3], [0, 2], [1, 3], [0, 2]], dtype=np.int64)
        neighbor_weights = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["redistribution"].wind_transport(
                    swe, sx, wind_speed, transport_coeff, area,
                    neighbor_indices, neighbor_weights
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"wind_transport edge cases mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_avalanche_redistribute(self):
        """All backends should produce identical avalanche_redistribute results."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        swe = np.array([200.0, 50.0, 100.0, 75.0])
        slope = np.array([50.0, 40.0, 20.0, 55.0])
        slope_threshold = 35.0
        flow_fractions = np.array([
            [0.6, 0.4, 0.0],
            [0.5, 0.5, 0.0],
            [0.3, 0.7, 0.0],
            [0.4, 0.3, 0.3]
        ])
        neighbor_indices = np.array([
            [1, 2, -1],
            [0, 2, -1],
            [0, 1, -1],
            [0, 1, 2]
        ], dtype=np.int64)
        area = np.array([1000.0, 1200.0, 800.0, 1500.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["redistribution"].avalanche_redistribute(
                    swe, slope, slope_threshold, flow_fractions,
                    neighbor_indices, area
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"avalanche_redistribute mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_avalanche_edge_cases(self):
        """All backends should handle edge cases identically for avalanche."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        # Edge cases: at threshold, no snow, varying slopes
        swe = np.array([0.0, 100.0, 100.0])
        slope = np.array([50.0, 30.0, 35.0])  # steep with no snow, at threshold, just above
        slope_threshold = 30.0
        flow_fractions = np.array([[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
        neighbor_indices = np.array([[1, 2], [0, 2], [0, 1]], dtype=np.int64)
        area = np.array([1000.0, 1000.0, 1000.0])

        results = {}
        for name, modules in BACKENDS.items():
            results[name] = np.asarray(
                modules["redistribution"].avalanche_redistribute(
                    swe, slope, slope_threshold, flow_fractions,
                    neighbor_indices, area
                )
            )

        names = list(results.keys())
        for i in range(1, len(names)):
            np.testing.assert_allclose(
                results[names[0]], results[names[i]],
                rtol=RTOL, atol=ATOL,
                err_msg=f"avalanche edge cases mismatch between {names[0]} and {names[i]}",
            )

    def test_agreement_mass_conservation(self):
        """All backends should conserve mass identically."""
        from tests.test_kernels.conftest import BACKENDS

        if len(BACKENDS) < 2:
            pytest.skip("Need at least 2 backends for cross-backend test")

        rng = np.random.default_rng(42)
        n_units = 10

        swe = rng.uniform(50.0, 200.0, n_units)
        sx = rng.uniform(-30.0, 30.0, n_units)
        wind_speed = rng.uniform(5.0, 20.0, n_units)
        transport_coeff = 0.01
        area = rng.uniform(500.0, 2000.0, n_units)

        # Create random connectivity
        neighbor_indices = np.zeros((n_units, 3), dtype=np.int64)
        neighbor_weights = np.zeros((n_units, 3), dtype=np.float64)
        for i in range(n_units):
            neighbors = list(range(n_units))
            neighbors.remove(i)
            rng.shuffle(neighbors)
            for j in range(3):
                neighbor_indices[i, j] = neighbors[j]
            weights = rng.uniform(0.1, 1.0, 3)
            neighbor_weights[i] = weights / weights.sum()

        mass_before = np.sum(swe * area)

        for name, modules in BACKENDS.items():
            result = np.asarray(
                modules["redistribution"].wind_transport(
                    swe, sx, wind_speed, transport_coeff, area,
                    neighbor_indices, neighbor_weights
                )
            )
            mass_after = np.sum(result * area)
            np.testing.assert_allclose(
                mass_after, mass_before,
                rtol=1e-10, atol=1e-10,
                err_msg=f"Mass not conserved in {name} backend",
            )
