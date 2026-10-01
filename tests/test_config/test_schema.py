"""Tests for configuration schema."""

from pathlib import Path

import pytest

from topopyscale2.config.schema import (
    ApplicationConfig,
    BlendingConfig,
    CatchmentConfig,
    ClusteringConfig,
    DomainConfig,
    DownscalingConfig,
    ExecutionConfig,
    FeedbackConfig,
    GlacierConfig,
    InputConfig,
    NWPSourceConfig,
    OutputConfig,
    PolygonConfig,
    QCConfig,
    RedistributionConfig,
    SLURMConfig,
    StateDAConfig,
    StorageConfig,
    TPS2Config,
    WindConfig,
)


class TestDomainConfig:
    def test_valid_bbox(self):
        cfg = DomainConfig(bbox=[68.0, 38.0, 78.0, 43.0])
        assert cfg.bbox == [68.0, 38.0, 78.0, 43.0]
        assert cfg.dem_source == "glo_90"

    def test_valid_dem_path(self, tmp_path):
        dem_path = tmp_path / "test.tif"
        dem_path.touch()
        cfg = DomainConfig(dem=dem_path)
        assert cfg.dem == dem_path

    def test_no_dem_or_bbox_raises(self):
        with pytest.raises(ValueError, match="Either 'dem'"):
            DomainConfig()

    def test_invalid_spatial_mode(self):
        with pytest.raises(ValueError, match="spatial_mode"):
            DomainConfig(bbox=[0, 0, 1, 1], spatial_mode="invalid")

    def test_invalid_bbox_length(self):
        with pytest.raises(ValueError, match="bbox must be"):
            DomainConfig(bbox=[0, 0, 1])

    def test_invalid_bbox_south_north(self):
        with pytest.raises(ValueError, match="south must be less than north"):
            DomainConfig(bbox=[0, 5, 1, 3])

    def test_catchments_spatial_mode(self):
        cfg = DomainConfig(bbox=[0, 0, 1, 1], spatial_mode="catchments")
        assert cfg.spatial_mode == "catchments"


class TestCatchmentConfig:
    def test_minimal(self, tmp_path):
        shp = tmp_path / "catchments.shp"
        shp.touch()
        cfg = CatchmentConfig(shapefile=shp)
        assert cfg.id_column == "HYBAS_ID"
        assert cfg.downstream_column == "NEXT_DOWN"
        assert cfg.area_column == "SUB_AREA"

    def test_custom_columns(self, tmp_path):
        shp = tmp_path / "catchments.shp"
        shp.touch()
        cfg = CatchmentConfig(
            shapefile=shp,
            id_column="CATCHMENT_ID",
            downstream_column="DOWNSTREAM",
            area_column="AREA_KM2",
        )
        assert cfg.id_column == "CATCHMENT_ID"
        assert cfg.downstream_column == "DOWNSTREAM"
        assert cfg.area_column == "AREA_KM2"


class TestClusteringConfig:
    def test_defaults(self):
        cfg = ClusteringConfig()
        assert cfg.method == "kmeans"
        assert cfg.n_clusters == 100
        assert cfg.feature_standardization is True
        assert "elevation" in cfg.features

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            ClusteringConfig(method="invalid")


class TestInputConfig:
    def test_defaults(self):
        cfg = InputConfig()
        assert cfg.backend == "google"
        assert cfg.pressure_levels == [300, 500, 700, 850, 1000]

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            InputConfig(backend="invalid")

    def test_invalid_time_range(self):
        with pytest.raises(ValueError, match="time_range"):
            InputConfig(time_range=["2020-01-01"])


class TestOutputConfig:
    def test_defaults(self):
        cfg = OutputConfig()
        assert cfg.format == "netcdf"
        assert cfg.compression is True

    def test_invalid_format(self):
        with pytest.raises(ValueError, match="format"):
            OutputConfig(format="invalid")


class TestExecutionConfig:
    def test_defaults(self):
        cfg = ExecutionConfig()
        assert cfg.kernel_backend == "rust"
        assert cfg.n_workers == 4

    def test_invalid_kernel_backend(self):
        with pytest.raises(ValueError, match="kernel_backend"):
            ExecutionConfig(kernel_backend="invalid")


class TestTPS2Config:
    def test_minimal(self):
        cfg = TPS2Config(domain=DomainConfig(bbox=[0, 0, 1, 1]))
        assert cfg.domain.bbox == [0, 0, 1, 1]
        assert cfg.clustering.method == "kmeans"

    def test_from_yaml(self):
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "central_asia_basic.yaml"
        )
        cfg = TPS2Config.from_yaml(yaml_path)
        assert cfg.domain.bbox == [68.0, 38.0, 78.0, 43.0]
        assert cfg.clustering.n_clusters == 200
        assert cfg.inputs.backend == "google"
        assert cfg.execution.kernel_backend == "rust"

    def test_from_yaml_all_fields(self):
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "central_asia_basic.yaml"
        )
        cfg = TPS2Config.from_yaml(yaml_path)
        assert cfg.downscaling.temperature == "lapse_rate"
        assert cfg.output.compression is True
        assert "temperature" in cfg.output.variables

    def test_from_yaml_phase2_config(self):
        yaml_path = (
            Path(__file__).parent.parent.parent
            / "topopyscale2"
            / "config"
            / "examples"
            / "switzerland_snow.yaml"
        )
        cfg = TPS2Config.from_yaml(yaml_path)
        assert cfg.domain.spatial_mode == "polygons"
        assert cfg.polygons.elevation_bands == 200.0
        assert cfg.downscaling.wind_config.method == "winstral"
        assert cfg.application.model == "fsm"

    def test_phase2_defaults(self):
        cfg = TPS2Config(domain=DomainConfig(bbox=[0, 0, 1, 1]))
        assert cfg.polygons.method == "hru"
        assert cfg.qc.range_checks is True
        assert cfg.storage.compression == "zstd"
        assert cfg.application.model == "none"


# =============================================================================
# Phase 2 Config Tests
# =============================================================================


class TestWindConfig:
    def test_defaults(self):
        cfg = WindConfig()
        assert cfg.method == "log_profile"
        assert cfg.sx_search_distance_m == 300.0
        assert cfg.sx_n_directions == 36
        assert cfg.roughness_lengths["glacier"] == 0.001
        assert cfg.roughness_lengths["forest"] == 1.0

    def test_winstral_method(self):
        cfg = WindConfig(method="winstral")
        assert cfg.method == "winstral"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            WindConfig(method="invalid")

    def test_invalid_sx_n_directions_too_low(self):
        with pytest.raises(ValueError, match="sx_n_directions"):
            WindConfig(sx_n_directions=2)

    def test_invalid_sx_n_directions_too_high(self):
        with pytest.raises(ValueError, match="sx_n_directions"):
            WindConfig(sx_n_directions=500)

    def test_invalid_sx_search_distance(self):
        with pytest.raises(ValueError, match="sx_search_distance_m"):
            WindConfig(sx_search_distance_m=-100)

    def test_custom_roughness_lengths(self):
        cfg = WindConfig(roughness_lengths={"custom": 0.5})
        assert cfg.roughness_lengths["custom"] == 0.5


class TestRedistributionConfig:
    def test_defaults(self):
        cfg = RedistributionConfig()
        assert cfg.wind_transport is False
        assert cfg.avalanche is False
        assert cfg.avalanche_slope_threshold == 35.0
        assert cfg.transport_coefficient == 0.5

    def test_enable_transport(self):
        cfg = RedistributionConfig(wind_transport=True, avalanche=True)
        assert cfg.wind_transport is True
        assert cfg.avalanche is True

    def test_invalid_slope_threshold_negative(self):
        with pytest.raises(ValueError, match="avalanche_slope_threshold"):
            RedistributionConfig(avalanche_slope_threshold=-10)

    def test_invalid_slope_threshold_too_high(self):
        with pytest.raises(ValueError, match="avalanche_slope_threshold"):
            RedistributionConfig(avalanche_slope_threshold=100)

    def test_invalid_transport_coefficient_negative(self):
        with pytest.raises(ValueError, match="transport_coefficient"):
            RedistributionConfig(transport_coefficient=-0.1)

    def test_invalid_transport_coefficient_too_high(self):
        with pytest.raises(ValueError, match="transport_coefficient"):
            RedistributionConfig(transport_coefficient=1.5)


class TestPolygonConfig:
    def test_defaults(self):
        cfg = PolygonConfig()
        assert cfg.method == "hru"
        assert cfg.respect_catchments is True
        assert cfg.elevation_bands == 200.0
        assert cfg.aspect_classes == 4

    def test_quadtree_method(self):
        cfg = PolygonConfig(method="quadtree")
        assert cfg.method == "quadtree"

    def test_segmentation_method(self):
        cfg = PolygonConfig(method="segmentation")
        assert cfg.method == "segmentation"

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            PolygonConfig(method="invalid")

    def test_eight_aspect_classes(self):
        cfg = PolygonConfig(aspect_classes=8)
        assert cfg.aspect_classes == 8

    def test_invalid_aspect_classes(self):
        with pytest.raises(ValueError):
            PolygonConfig(aspect_classes=6)

    def test_invalid_elevation_bands(self):
        with pytest.raises(ValueError, match="elevation_bands"):
            PolygonConfig(elevation_bands=-100)

    def test_invalid_min_area(self):
        with pytest.raises(ValueError, match="min_area_m2"):
            PolygonConfig(min_area_m2=0)


class TestQCConfig:
    def test_defaults(self):
        cfg = QCConfig()
        assert cfg.range_checks is True
        assert cfg.temporal_consistency is True
        assert cfg.spatial_consistency is True
        assert cfg.flatline_hours == 24
        assert cfg.suspect_uncertainty_inflation == 3.0

    def test_disable_checks(self):
        cfg = QCConfig(range_checks=False, temporal_consistency=False)
        assert cfg.range_checks is False
        assert cfg.temporal_consistency is False

    def test_invalid_flatline_hours(self):
        with pytest.raises(ValueError, match="flatline_hours"):
            QCConfig(flatline_hours=0)

    def test_invalid_uncertainty_inflation(self):
        with pytest.raises(ValueError, match="suspect_uncertainty_inflation"):
            QCConfig(suspect_uncertainty_inflation=0.5)


class TestSLURMConfig:
    def test_defaults(self):
        cfg = SLURMConfig()
        assert cfg.partition == "standard"
        assert cfg.time == "24:00:00"
        assert cfg.mem_per_cpu == "4G"
        assert cfg.account is None

    def test_custom_settings(self):
        cfg = SLURMConfig(
            partition="gpu",
            time="48:00:00",
            mem_per_cpu="8G",
            account="myproject",
        )
        assert cfg.partition == "gpu"
        assert cfg.time == "48:00:00"
        assert cfg.account == "myproject"

    def test_invalid_time_format(self):
        with pytest.raises(ValueError, match="time must be in format"):
            SLURMConfig(time="24h")

    def test_valid_time_format_long(self):
        cfg = SLURMConfig(time="100:00:00")
        assert cfg.time == "100:00:00"


class TestStorageConfig:
    def test_defaults(self):
        cfg = StorageConfig()
        assert cfg.input_cache_format == "zarr"
        assert cfg.working_format == "zarr"
        assert cfg.compression == "zstd"
        assert cfg.max_cache_gb is None  # None = no limit

    def test_netcdf_format(self):
        cfg = StorageConfig(input_cache_format="netcdf", working_format="netcdf")
        assert cfg.input_cache_format == "netcdf"

    def test_different_compressions(self):
        for comp in ["zstd", "lz4", "gzip", "none"]:
            cfg = StorageConfig(compression=comp)
            assert cfg.compression == comp

    def test_invalid_compression(self):
        with pytest.raises(ValueError):
            StorageConfig(compression="bz2")

    def test_invalid_max_cache(self):
        with pytest.raises(ValueError, match="max_cache_gb"):
            StorageConfig(max_cache_gb=0)


class TestFeedbackConfig:
    def test_defaults(self):
        cfg = FeedbackConfig()
        assert cfg.enabled is False
        assert cfg.method == "sdc"
        assert cfg.observation_type is None

    def test_enabled(self):
        cfg = FeedbackConfig(enabled=True, observation_type="fsca")
        assert cfg.enabled is True
        assert cfg.observation_type == "fsca"


class TestStateDAConfig:
    def test_defaults(self):
        cfg = StateDAConfig()
        assert cfg.enabled is False
        assert cfg.method == "enkf"
        assert cfg.ensemble_size == 50

    def test_particle_filter(self):
        cfg = StateDAConfig(method="pf", ensemble_size=100)
        assert cfg.method == "pf"
        assert cfg.ensemble_size == 100

    def test_invalid_ensemble_size(self):
        with pytest.raises(ValueError, match="ensemble_size"):
            StateDAConfig(ensemble_size=1)


class TestApplicationConfig:
    def test_defaults(self):
        cfg = ApplicationConfig()
        assert cfg.model == "none"
        assert cfg.feedback.enabled is False
        assert cfg.state_da.enabled is False

    def test_fsm_model(self):
        cfg = ApplicationConfig(model="fsm")
        assert cfg.model == "fsm"

    def test_all_models(self):
        for model in ["fsm", "hydrology", "cryogrid", "snowpack", "none"]:
            cfg = ApplicationConfig(model=model)
            assert cfg.model == model

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            ApplicationConfig(model="invalid")


class TestNWPSourceConfig:
    def test_minimal(self):
        cfg = NWPSourceConfig(name="era5_main")
        assert cfg.name == "era5_main"
        assert cfg.type == "era5"
        assert cfg.priority == 1

    def test_hres_source(self):
        cfg = NWPSourceConfig(name="hres", type="hres", priority=2, resolution_m=9000.0)
        assert cfg.type == "hres"
        assert cfg.resolution_m == 9000.0

    def test_custom_source(self):
        cfg = NWPSourceConfig(
            name="local_wrf",
            type="custom",
            path=Path("/data/wrf"),
            variable_mapping={"t2m": "T2"},
        )
        assert cfg.type == "custom"
        assert cfg.path == Path("/data/wrf")
        assert cfg.variable_mapping["t2m"] == "T2"


class TestBlendingConfig:
    def test_defaults(self):
        cfg = BlendingConfig()
        assert cfg.enabled is False
        assert cfg.method == "priority"
        assert cfg.gap_fill is True

    def test_weighted_method(self):
        cfg = BlendingConfig(enabled=True, method="weighted")
        assert cfg.method == "weighted"


class TestDownscalingConfig:
    def test_defaults(self):
        cfg = DownscalingConfig()
        assert cfg.wind_config.method == "log_profile"
        assert cfg.redistribution.wind_transport is False

    def test_with_wind_config(self):
        cfg = DownscalingConfig(
            wind="winstral",
            wind_config=WindConfig(method="winstral", sx_search_distance_m=500),
        )
        assert cfg.wind == "winstral"
        assert cfg.wind_config.sx_search_distance_m == 500


class TestExecutionConfigPhase2:
    def test_slurm_backend(self):
        cfg = ExecutionConfig(backend="slurm")
        assert cfg.backend == "slurm"
        assert cfg.slurm.partition == "standard"

    def test_slurm_with_custom_settings(self):
        cfg = ExecutionConfig(
            backend="slurm",
            slurm=SLURMConfig(partition="gpu", time="72:00:00"),
        )
        assert cfg.slurm.partition == "gpu"
        assert cfg.slurm.time == "72:00:00"

    def test_invalid_backend(self):
        with pytest.raises(ValueError, match="backend"):
            ExecutionConfig(backend="kubernetes")


class TestInputConfigPhase2:
    def test_with_sources(self):
        cfg = InputConfig(
            sources=[
                NWPSourceConfig(name="era5"),
                NWPSourceConfig(name="hres", type="hres", priority=2),
            ]
        )
        assert len(cfg.sources) == 2
        assert cfg.sources[0].name == "era5"
        assert cfg.sources[1].type == "hres"

    def test_with_blending(self):
        cfg = InputConfig(blending=BlendingConfig(enabled=True, method="weighted"))
        assert cfg.blending.enabled is True
        assert cfg.blending.method == "weighted"


class TestOutputConfigPhase2:
    def test_new_formats(self):
        for fmt in ["smet", "hbv", "cryogrid", "crocus", "zarr"]:
            cfg = OutputConfig(format=fmt)
            assert cfg.format == fmt


class TestGlacierConfig:
    def test_defaults(self):
        cfg = GlacierConfig()
        assert cfg.finite_ice is False
        assert cfg.ice_thickness_source == "none"
        assert cfg.ice_thickness_path is None

    def test_finite_ice_requires_source(self):
        with pytest.raises(ValueError, match="finite_ice"):
            GlacierConfig(finite_ice=True)

    def test_source_requires_path(self):
        with pytest.raises(ValueError, match="ice_thickness_path"):
            GlacierConfig(ice_thickness_source="farinotti")

    def test_valid_finite_ice(self, tmp_path):
        cfg = GlacierConfig(
            finite_ice=True,
            ice_thickness_source="custom",
            ice_thickness_path=tmp_path / "thickness.tif",
        )
        assert cfg.finite_ice is True

    def test_application_config_default_glacier(self):
        cfg = ApplicationConfig()
        assert cfg.glacier.finite_ice is False
