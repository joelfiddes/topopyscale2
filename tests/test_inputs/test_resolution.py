"""Tests for resolution scaling utilities."""

import pytest

from topopyscale2.inputs.resolution import (
    adaptive_lapse_rate_weight,
    adaptive_precipitation_gradient_weight,
    adaptive_radiation_partitioning_weight,
    adaptive_wind_correction_weight,
    compute_correction_weights,
    estimate_effective_resolution,
    resolution_scaling_factor,
)


class TestResolutionScalingFactor:
    """Test resolution_scaling_factor function."""

    def test_coarse_to_fine_high_factor(self):
        """Coarse source to fine target gives high factor."""
        # ERA5 (31km) to 100m: need lots of correction
        factor = resolution_scaling_factor(31000, 100)
        assert factor > 0.9

    def test_fine_to_fine_low_factor(self):
        """Fine source to fine target gives low factor."""
        # COSMO-1 (1km) to 100m: less correction needed
        factor = resolution_scaling_factor(1000, 100)
        assert factor < 0.95

    def test_same_resolution_min_factor(self):
        """Same resolution gives minimum factor."""
        factor = resolution_scaling_factor(100, 100)
        assert factor == 0.1  # min_factor

    def test_finer_source_than_target(self):
        """Source finer than target gives minimum factor."""
        factor = resolution_scaling_factor(50, 100)
        assert factor == 0.1

    def test_linear_method(self):
        """Linear method produces expected scaling."""
        factor = resolution_scaling_factor(10000, 100, method="linear")
        assert 0.1 <= factor <= 1.0

    def test_sqrt_method(self):
        """Sqrt method produces expected scaling."""
        factor = resolution_scaling_factor(10000, 100, method="sqrt")
        assert 0.1 <= factor <= 1.0

    def test_log_method(self):
        """Log method produces expected scaling."""
        factor = resolution_scaling_factor(10000, 100, method="log")
        assert 0.1 <= factor <= 1.0

    def test_invalid_method_raises(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown method"):
            resolution_scaling_factor(10000, 100, method="invalid")

    def test_negative_resolution_raises(self):
        """Negative resolution raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            resolution_scaling_factor(-1000, 100)

    def test_zero_resolution_raises(self):
        """Zero resolution raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            resolution_scaling_factor(1000, 0)

    def test_custom_min_max(self):
        """Custom min/max factors are respected."""
        factor = resolution_scaling_factor(
            100, 100, min_factor=0.5, max_factor=0.8
        )
        assert factor == 0.5


class TestComputeCorrectionWeights:
    """Test compute_correction_weights function."""

    def test_multiple_sources(self):
        """Computes weights for multiple sources."""
        resolutions = {"era5": 31000, "hres": 9000, "cosmo": 1000}
        weights = compute_correction_weights(resolutions, target_res_m=100)

        assert len(weights) == 3
        # ERA5 (coarsest) should have highest weight
        assert weights["era5"] > weights["hres"]
        assert weights["hres"] > weights["cosmo"]

    def test_empty_dict(self):
        """Empty dict returns empty dict."""
        weights = compute_correction_weights({}, target_res_m=100)
        assert weights == {}


class TestEstimateEffectiveResolution:
    """Test estimate_effective_resolution function."""

    def test_single_source(self):
        """Single source gives that source's resolution."""
        weights = {"era5": 1.0}
        resolutions = {"era5": 31000.0}
        eff_res = estimate_effective_resolution(weights, resolutions)
        assert eff_res == 31000.0

    def test_weighted_average(self):
        """Multiple sources give weighted average."""
        weights = {"era5": 0.5, "cosmo": 0.5}
        resolutions = {"era5": 31000.0, "cosmo": 1000.0}
        eff_res = estimate_effective_resolution(weights, resolutions)
        # Should be (0.5 * 31000 + 0.5 * 1000) = 16000
        assert eff_res == 16000.0

    def test_unequal_weights(self):
        """Unequal weights are handled correctly."""
        weights = {"era5": 0.25, "cosmo": 0.75}
        resolutions = {"era5": 31000.0, "cosmo": 1000.0}
        eff_res = estimate_effective_resolution(weights, resolutions)
        # Should be (0.25 * 31000 + 0.75 * 1000) = 8500
        assert eff_res == 8500.0

    def test_zero_weights_raises(self):
        """Zero total weight raises ValueError."""
        weights = {"era5": 0, "cosmo": 0}
        resolutions = {"era5": 31000.0, "cosmo": 1000.0}
        with pytest.raises(ValueError, match="sum to zero"):
            estimate_effective_resolution(weights, resolutions)


class TestAdaptiveLapseRateWeight:
    """Test adaptive_lapse_rate_weight function."""

    def test_coarse_source_high_weight(self):
        """Coarse source needs more local correction."""
        weight = adaptive_lapse_rate_weight(31000, 100)
        assert weight > 0.5

    def test_fine_source_lower_weight(self):
        """Fine source needs less local correction."""
        weight_coarse = adaptive_lapse_rate_weight(31000, 100)
        weight_fine = adaptive_lapse_rate_weight(1000, 100)
        assert weight_fine < weight_coarse

    def test_terrain_complexity_increases_weight(self):
        """Complex terrain increases correction weight."""
        weight_simple = adaptive_lapse_rate_weight(10000, 100, terrain_complexity=0.0)
        weight_complex = adaptive_lapse_rate_weight(10000, 100, terrain_complexity=1.0)
        assert weight_complex > weight_simple


class TestAdaptiveRadiationWeight:
    """Test adaptive_radiation_partitioning_weight function."""

    def test_coarse_source_high_weight(self):
        """Coarse source needs more topographic correction."""
        weight = adaptive_radiation_partitioning_weight(31000, 100)
        assert weight > 0.9

    def test_fine_source_lower_weight(self):
        """Fine source needs less correction."""
        weight_coarse = adaptive_radiation_partitioning_weight(31000, 100)
        weight_fine = adaptive_radiation_partitioning_weight(1000, 100)
        assert weight_fine < weight_coarse


class TestAdaptiveWindWeight:
    """Test adaptive_wind_correction_weight function."""

    def test_coarse_source_high_weight(self):
        """Coarse source needs wind correction."""
        weight = adaptive_wind_correction_weight(31000, 100)
        assert weight > 0.5

    def test_range(self):
        """Weight is in valid range."""
        weight = adaptive_wind_correction_weight(10000, 100)
        assert 0.1 <= weight <= 1.0


class TestAdaptivePrecipitationWeight:
    """Test adaptive_precipitation_gradient_weight function."""

    def test_coarse_source_high_weight(self):
        """Coarse source underestimates orographic enhancement."""
        weight = adaptive_precipitation_gradient_weight(31000, 100)
        assert weight > 0.9

    def test_fine_source_lower_weight(self):
        """Fine source captures more orographic effects."""
        weight_coarse = adaptive_precipitation_gradient_weight(31000, 100)
        weight_fine = adaptive_precipitation_gradient_weight(1000, 100)
        assert weight_fine < weight_coarse
