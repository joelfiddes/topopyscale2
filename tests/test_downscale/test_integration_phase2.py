"""Integration tests for Phase 2 features.

Tests end-to-end functionality of Phase 2 additions:
- Polygon mode domain setup
- Winstral wind configuration
- Redistribution configuration
- Multi-NWP source configuration
- Storage tier configuration
"""

from pathlib import Path

import pytest

from topopyscale2.config.schema import (
    ApplicationConfig,
    BlendingConfig,
    DomainConfig,
    DownscalingConfig,
    FeedbackConfig,
    InputConfig,
    NWPSourceConfig,
    PolygonConfig,
    RedistributionConfig,
    StateDAConfig,
    StorageConfig,
    TPS2Config,
    WindConfig,
)
from topopyscale2.core.downscale import Downscaler
from topopyscale2.domain import Domain


class TestPolygonModeIntegration:
    """Tests for polygon (HRU) spatial mode."""

    def test_polygon_mode_config_parsing(self):
        """Config parses polygon mode settings correctly."""
        config = TPS2Config(
            domain=DomainConfig(bbox=[0, 0, 1, 1], spatial_mode="polygons"),
            polygons=PolygonConfig(
                method="hru",
                elevation_bands=200.0,
                aspect_classes=4,
                respect_catchments=True,
                min_area_m2=50000.0,
            ),
        )
        assert config.domain.spatial_mode == "polygons"
        assert config.polygons.method == "hru"
        assert config.polygons.elevation_bands == 200.0
        assert config.polygons.aspect_classes == 4

    def test_polygon_config_aspect_classes_validation(self):
        """Aspect classes must be 4 or 8."""
        config = TPS2Config(
            domain=DomainConfig(bbox=[0, 0, 1, 1], spatial_mode="polygons"),
            polygons=PolygonConfig(aspect_classes=8),
        )
        assert config.polygons.aspect_classes == 8

    def test_domain_polygon_mode_info(self):
        """Domain info reports polygon mode correctly."""
        config = TPS2Config(
            domain=DomainConfig(bbox=[0, 0, 1, 1], spatial_mode="polygons"),
            polygons=PolygonConfig(elevation_bands=100.0),
        )
        domain = Domain(config)
        info = domain.info()
        assert "polygons" in info.lower()


class TestWinstralWindIntegration:
    """Tests for Winstral wind downscaling configuration."""

    def test_wind_config_defaults(self):
        """WindConfig has sensible defaults."""
        config = WindConfig()
        assert config.method == "log_profile"
        assert config.sx_search_distance_m == 300.0
        assert config.sx_n_directions == 36
        assert "glacier" in config.roughness_lengths
        assert "forest" in config.displacement_heights

    def test_wind_config_winstral_method(self):
        """WindConfig accepts winstral method."""
        config = WindConfig(method="winstral")
        assert config.method == "winstral"

    def test_downscaler_wind_config(self):
        """Downscaler accepts wind config."""
        wind_config = WindConfig(
            method="winstral",
            sx_search_distance_m=500.0,
            roughness_lengths={"glacier": 0.002},
        )
        downscaler = Downscaler(backend="python", wind_config=wind_config)
        assert downscaler.wind_config.method == "winstral"
        assert downscaler.roughness_lengths["glacier"] == 0.002

    def test_sx_n_directions_validation(self):
        """sx_n_directions must be in valid range."""
        with pytest.raises(ValueError):
            WindConfig(sx_n_directions=3)  # Too small
        with pytest.raises(ValueError):
            WindConfig(sx_n_directions=500)  # Too large


class TestRedistributionIntegration:
    """Tests for snow redistribution configuration."""

    def test_redistribution_config_defaults(self):
        """RedistributionConfig defaults are reasonable."""
        config = RedistributionConfig()
        assert config.wind_transport is False
        assert config.avalanche is False
        assert config.avalanche_slope_threshold == 35.0
        assert config.transport_coefficient == 0.5

    def test_redistribution_wind_transport_enabled(self):
        """Wind transport can be enabled."""
        config = RedistributionConfig(wind_transport=True)
        assert config.wind_transport is True

    def test_redistribution_avalanche_enabled(self):
        """Avalanche redistribution can be enabled."""
        config = RedistributionConfig(avalanche=True, avalanche_slope_threshold=40.0)
        assert config.avalanche is True
        assert config.avalanche_slope_threshold == 40.0

    def test_avalanche_slope_threshold_validation(self):
        """Slope threshold must be 0-90 degrees."""
        with pytest.raises(ValueError):
            RedistributionConfig(avalanche_slope_threshold=-5.0)
        with pytest.raises(ValueError):
            RedistributionConfig(avalanche_slope_threshold=100.0)

    def test_transport_coefficient_validation(self):
        """Transport coefficient must be 0-1."""
        with pytest.raises(ValueError):
            RedistributionConfig(transport_coefficient=-0.1)
        with pytest.raises(ValueError):
            RedistributionConfig(transport_coefficient=1.5)


class TestMultiNWPIntegration:
    """Tests for multi-NWP source configuration."""

    def test_nwp_source_config(self):
        """NWP source configuration parses correctly."""
        source = NWPSourceConfig(
            name="era5_primary",
            type="era5",
            priority=1,
            resolution_m=31000.0,
        )
        assert source.name == "era5_primary"
        assert source.type == "era5"
        assert source.priority == 1

    def test_blending_config_priority(self):
        """Blending config with priority method."""
        blending = BlendingConfig(
            enabled=True,
            method="priority",
            gap_fill=True,
        )
        assert blending.enabled is True
        assert blending.method == "priority"

    def test_input_config_with_sources(self):
        """Input config with multiple NWP sources."""
        config = InputConfig(
            primary="era5",
            backend="google",
            time_range=["2020-01-01", "2020-12-31"],
            sources=[
                NWPSourceConfig(name="era5", type="era5", priority=1),
                NWPSourceConfig(name="hres", type="hres", priority=2),
            ],
            blending=BlendingConfig(enabled=True, method="priority"),
        )
        assert len(config.sources) == 2
        assert config.blending.enabled is True


class TestStorageTierIntegration:
    """Tests for storage tier configuration."""

    def test_storage_config_defaults(self):
        """StorageConfig has sensible defaults."""
        config = StorageConfig()
        assert config.input_cache_format == "zarr"
        assert config.working_format == "zarr"
        assert config.compression == "zstd"
        assert config.max_cache_gb is None  # None = no limit

    def test_storage_config_netcdf(self):
        """Storage config can use netcdf format."""
        config = StorageConfig(
            input_cache_format="netcdf",
            working_format="netcdf",
        )
        assert config.input_cache_format == "netcdf"
        assert config.working_format == "netcdf"

    def test_storage_compression_options(self):
        """All compression options are valid."""
        for compression in ["zstd", "lz4", "gzip", "none"]:
            config = StorageConfig(compression=compression)
            assert config.compression == compression


class TestApplicationConfigIntegration:
    """Tests for application layer configuration."""

    def test_application_config_fsm(self):
        """Application config for FSM model."""
        config = ApplicationConfig(
            model="fsm",
            feedback=FeedbackConfig(enabled=True, method="sdc", observation_type="fsca"),
        )
        assert config.model == "fsm"
        assert config.feedback.enabled is True
        assert config.feedback.method == "sdc"

    def test_application_config_state_da(self):
        """Application config with state DA."""
        config = ApplicationConfig(
            model="fsm",
            state_da=StateDAConfig(enabled=True, method="enkf", ensemble_size=100),
        )
        assert config.state_da.enabled is True
        assert config.state_da.ensemble_size == 100


class TestFullConfigIntegration:
    """Tests for complete Phase 2 configuration."""

    def test_switzerland_snow_config_parsing(self):
        """Switzerland snow example config parses correctly."""
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        if yaml_path.exists():
            config = TPS2Config.from_yaml(yaml_path)
            assert config.domain.spatial_mode == "polygons"
            assert config.downscaling.wind_config.method == "winstral"
            assert config.downscaling.redistribution.wind_transport is True
            assert len(config.inputs.sources) >= 1
            assert config.application.model == "fsm"

    def test_phase2_config_round_trip(self):
        """Full Phase 2 config creates valid domain."""
        config = TPS2Config(
            domain=DomainConfig(bbox=[0, 0, 1, 1], spatial_mode="polygons"),
            polygons=PolygonConfig(elevation_bands=200.0, aspect_classes=4),
            downscaling=DownscalingConfig(
                wind_config=WindConfig(method="winstral"),
                redistribution=RedistributionConfig(wind_transport=True),
            ),
            storage=StorageConfig(compression="lz4"),
            application=ApplicationConfig(model="fsm"),
        )
        domain = Domain(config)
        assert domain.config.domain.spatial_mode == "polygons"
        assert domain.config.downscaling.wind_config.method == "winstral"
