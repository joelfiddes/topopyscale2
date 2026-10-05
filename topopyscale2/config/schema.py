"""Pydantic configuration schema for TopoPyScale 2.0."""

import importlib.util
import logging
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Config blocks that configure features outside the core downscaling engine, and the
# module that implements each. A public release may ship without them: the block is
# then accepted, ignored, and named once in a warning, rather than failing the load.
FEATURE_MODULES = {
    "validation": "topopyscale2.validation",
    "validation_lab": "topopyscale2.validation",
    "application": "topopyscale2.applications",
    "forecast": "topopyscale2.inputs.forecast_source",
    "da": "topopyscale2.da",
    "calibration": "topopyscale2.da",
    "climatology": "topopyscale2.outputs.climatology",
    "climate": "topopyscale2.climate",
    "topoclim": "topopyscale2.climate.perunit",
}


def feature_installed(module: str) -> bool:
    """True if ``module`` (a dotted name) can be imported in this installation."""
    try:
        return importlib.util.find_spec(module) is not None
    except ModuleNotFoundError:  # a parent package is absent
        return False

# =============================================================================
# Phase 2 Config Sections
# =============================================================================


class WindConfig(BaseModel):
    """Wind downscaling configuration."""

    method: Literal["log_profile", "winstral"] = "log_profile"
    sx_search_distance_m: float = 300.0
    sx_n_directions: int = 36
    roughness_lengths: dict[str, float] = Field(
        default_factory=lambda: {
            "glacier": 0.001,
            "open": 0.03,
            "rock": 0.01,
            "forest": 1.0,
        }
    )
    displacement_heights: dict[str, float] = Field(
        default_factory=lambda: {
            "glacier": 0.0,
            "open": 0.0,
            "rock": 0.0,
            "forest": 10.0,
        }
    )

    @field_validator("sx_n_directions")
    @classmethod
    def validate_sx_n_directions(cls, v):
        if v < 4 or v > 360:
            raise ValueError("sx_n_directions must be between 4 and 360")
        return v

    @field_validator("sx_search_distance_m")
    @classmethod
    def validate_sx_search_distance(cls, v):
        if v <= 0:
            raise ValueError("sx_search_distance_m must be positive")
        return v


class RedistributionConfig(BaseModel):
    """Snow redistribution configuration."""

    wind_transport: bool = False
    avalanche: bool = False
    avalanche_slope_threshold: float = 35.0
    transport_coefficient: float = 0.5

    @field_validator("avalanche_slope_threshold")
    @classmethod
    def validate_slope_threshold(cls, v):
        if v < 0 or v > 90:
            raise ValueError("avalanche_slope_threshold must be between 0 and 90 degrees")
        return v

    @field_validator("transport_coefficient")
    @classmethod
    def validate_transport_coefficient(cls, v):
        if v < 0 or v > 1:
            raise ValueError("transport_coefficient must be between 0 and 1")
        return v


class PolygonConfig(BaseModel):
    """Variable-resolution polygon (HRU) generation configuration."""

    method: Literal["hru", "quadtree", "segmentation"] = "hru"
    respect_catchments: bool = True
    elevation_bands: float = 200.0
    aspect_classes: Literal[4, 8] = 4
    min_area_m2: float = 10000.0

    @field_validator("elevation_bands")
    @classmethod
    def validate_elevation_bands(cls, v):
        if v <= 0:
            raise ValueError("elevation_bands must be positive")
        return v

    @field_validator("min_area_m2")
    @classmethod
    def validate_min_area(cls, v):
        if v <= 0:
            raise ValueError("min_area_m2 must be positive")
        return v


class QCConfig(BaseModel):
    """Quality control configuration for station observations."""

    range_checks: bool = True
    temporal_consistency: bool = True
    spatial_consistency: bool = True
    flatline_hours: int = 24
    suspect_uncertainty_inflation: float = 3.0

    @field_validator("flatline_hours")
    @classmethod
    def validate_flatline_hours(cls, v):
        if v < 1:
            raise ValueError("flatline_hours must be at least 1")
        return v

    @field_validator("suspect_uncertainty_inflation")
    @classmethod
    def validate_uncertainty_inflation(cls, v):
        if v < 1:
            raise ValueError("suspect_uncertainty_inflation must be at least 1")
        return v


class SLURMConfig(BaseModel):
    """SLURM HPC job configuration."""

    partition: str = "standard"
    time: str = "24:00:00"
    mem_per_cpu: str = "4G"
    account: Optional[str] = None
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 4

    @field_validator("time")
    @classmethod
    def validate_time_format(cls, v):
        import re

        if not re.match(r"^\d{1,3}:\d{2}:\d{2}$", v):
            raise ValueError("time must be in format HH:MM:SS or HHH:MM:SS")
        return v


class StorageConfig(BaseModel):
    """Storage tier configuration."""

    input_cache_format: Literal["zarr", "netcdf"] = "zarr"
    working_format: Literal["zarr", "netcdf"] = "zarr"
    compression: Literal["zstd", "lz4", "gzip", "none"] = "zstd"
    compression_level: int = 3
    max_cache_gb: Optional[float] = None  # None = no limit
    dem_cache: Path = Path("./dem_cache")
    forcing_cache: Path = Path("./forcing_cache")
    # Zarr-specific options
    zarr_merge: bool = True  # Merge daily stores into single ERA5.zarr
    zarr_cleanup_daily: bool = True  # Delete daily stores after successful merge
    zarr_chunks: dict[str, int] = Field(
        default_factory=lambda: {
            "time": 24,
            "level": -1,
            "latitude": -1,
            "longitude": -1,
        }
    )

    @field_validator("max_cache_gb")
    @classmethod
    def validate_max_cache(cls, v):
        if v is not None and v <= 0:
            raise ValueError("max_cache_gb must be positive or None (no limit)")
        return v

    @field_validator("compression_level")
    @classmethod
    def validate_compression_level(cls, v):
        if v < 1 or v > 22:
            raise ValueError("compression_level must be between 1 and 22")
        return v


class FeedbackConfig(BaseModel):
    """Application forcing feedback configuration."""

    enabled: bool = False
    method: Literal["sdc", "enkf", "none"] = "sdc"
    observation_type: Optional[str] = None


class StateDAConfig(BaseModel):
    """State data assimilation configuration."""

    enabled: bool = False
    method: Literal["enkf", "pf", "none"] = "enkf"
    ensemble_size: int = 50
    localization_radius_m: float = 5000.0

    @field_validator("ensemble_size")
    @classmethod
    def validate_ensemble_size(cls, v):
        if v < 2:
            raise ValueError("ensemble_size must be at least 2")
        return v


class PerturbationConfig(BaseModel):
    """Ensemble forcing perturbation configuration."""

    temperature_std: float = 1.0  # K, additive
    precipitation_factor_std: float = 0.3  # multiplicative
    shortwave_factor_std: float = 0.1  # multiplicative
    longwave_std: float = 10.0  # W/m2, additive
    wind_factor_std: float = 0.1  # multiplicative
    snowfall_additive_std: float = 0.0  # mm/hr, additive snowfall (creates snow where none exists)
    temporal_correlation: float = 0.8  # AR(1) coefficient

    # Spatial perturbation fields
    spatial_correlation_length_m: float = 0.0  # meters, 0 = uniform (no spatial structure)

    # Physics parameter perturbation (per-member FSM1 overrides)
    hfsn_std: float = 0.0  # m, fSCA depth scale (base ~0.1)
    rhof_std: float = 0.0  # kg/m³, fresh snow density (base ~100)
    asmx_std: float = 0.0  # fresh snow albedo (base ~0.8)
    tmlt_std: float = 0.0  # h, melting snow albedo decay timescale (base ~100)
    # Snow roughness length (base ~0.01 m). MULTIPLICATIVE, log-normal: the std
    # is the spread of ln(z0sn), so 0.7 ≈ a factor-of-two 1-sigma spread.
    # Roughness spans orders of magnitude (~0.0005-0.05 m across sites), so a
    # Gaussian on the raw value is mis-shaped and can go negative.
    z0sn_std: float = 0.0

    @field_validator("temporal_correlation")
    @classmethod
    def validate_temporal_correlation(cls, v):
        if v < 0 or v > 1:
            raise ValueError("temporal_correlation must be between 0 and 1")
        return v


class SatelliteObsConfig(BaseModel):
    """Satellite observation source configuration for DA."""

    source: Literal[
        "modis_fsca", "viirs_fsca", "sentinel2_snow",
        "sentinel1_wetsnow", "none",
    ] = "none"
    product: str = "MOD10A1"
    error_std: float = 0.1
    max_cloud_cover: float = 50.0
    cache_dir: Path = Path("./satellite_cache/")

    @field_validator("max_cloud_cover")
    @classmethod
    def validate_max_cloud_cover(cls, v):
        if v < 0 or v > 100:
            raise ValueError("max_cloud_cover must be between 0 and 100")
        return v


class DAConfig(BaseModel):
    """Data assimilation configuration.

    Supports three modes:
    - reanalysis: PBS/EnKS over a historical period (satellite + station obs)
    - realtime: EnKF sequential updates (station snow depth)
    - calibration: Parameter optimization (LHS/grid search)
    """

    enabled: bool = False
    mode: Literal["reanalysis", "realtime", "calibration", "batch_reanalysis", "none"] = "none"
    ensemble_size: int = 50
    ensemble_method: Literal["forcing", "synthetic"] = "forcing"
    perturbations: PerturbationConfig = Field(default_factory=PerturbationConfig)
    method: Literal["enkf", "enks", "pbs", "pf", "none"] = "pbs"
    localization_radius_m: float = 5000.0
    inflation_factor: float = 1.05
    cycle_interval: str = "1D"
    lookback_days: int = 14
    window_step_days: int = 7
    n_eff: int = 5
    satellite_obs: list[SatelliteObsConfig] = Field(default_factory=list)
    # fSCA observation source for the reanalysis DA:
    #   modis -> satellite_cache/fsca_daily.nc  (built by the operational MODIS step)
    #   fused -> satellite_cache/fused_fsca.nc  (MODIS + Sentinel-2 fusion, built by the
    #            scripts/gee runbook; must be refreshed externally -- the daily cycle
    #            only updates the MODIS file. S2 lifts the MODIS high-elevation
    #            viewability ceiling: obs +0.22 fSCA where MODIS>0.5 on kaz/Alps.)
    obs_source: Literal["modis", "fused"] = "modis"
    use_station_obs: bool = True
    smoother: bool = False
    smoother_lag: Optional[int] = None
    # Batch reanalysis settings
    water_year_start: Optional[int] = None
    water_year_end: Optional[int] = None
    cache_members: bool = False
    # Memory-safe chunked PBS: when set, process clusters in chunks of this size.
    # None (default) preserves the existing behavior of loading all ensemble
    # members fully into memory. Recommended for low-memory servers (e.g. 1000).
    chunk_size_clusters: Optional[int] = None

    @field_validator("ensemble_size")
    @classmethod
    def validate_ensemble_size(cls, v):
        if v < 2:
            raise ValueError("ensemble_size must be at least 2")
        return v

    @field_validator("inflation_factor")
    @classmethod
    def validate_inflation_factor(cls, v):
        if v < 1.0 or v > 2.0:
            raise ValueError("inflation_factor must be between 1.0 and 2.0")
        return v

    @field_validator("cycle_interval")
    @classmethod
    def validate_cycle_interval(cls, v):
        allowed = {"1H", "3H", "6H", "12H", "1D", "7D"}
        if v not in allowed:
            raise ValueError(f"cycle_interval must be one of {allowed}")
        return v

    @field_validator("lookback_days")
    @classmethod
    def validate_lookback_days(cls, v):
        if v < 1:
            raise ValueError("lookback_days must be at least 1")
        return v

    @field_validator("window_step_days")
    @classmethod
    def validate_window_step_days(cls, v):
        if v < 1:
            raise ValueError("window_step_days must be at least 1")
        return v

    @field_validator("n_eff")
    @classmethod
    def validate_n_eff(cls, v):
        if v < 1:
            raise ValueError("n_eff must be at least 1")
        return v

    @field_validator("chunk_size_clusters")
    @classmethod
    def validate_chunk_size_clusters(cls, v):
        if v is not None and v < 1:
            raise ValueError("chunk_size_clusters must be at least 1 (or None to disable)")
        return v


class ClimatologyConfig(BaseModel):
    """Climatology reference configuration for operational forecast comparisons."""

    reference_path: Optional[Path] = None  # External climatology.zarr
    auto_compare: bool = False  # Run comparison after forecast pipeline
    export_mcass: bool = False  # Export MCASS-format TSV files
    variables: list[str] = Field(default_factory=list)  # Subset of variables (empty = all)


class ClimateModelsConfig(BaseModel):
    """Climate model source configuration."""

    source: Literal["cmip6", "cordex", "custom"] = "cmip6"
    scenarios: list[str] = Field(default_factory=lambda: ["ssp245", "ssp585"])
    chains: str | list[str] = "all"  # "all" or list of specific model chains
    variables: list[str] = Field(
        default_factory=lambda: ["tas", "pr", "huss", "rsds", "rlds", "ps", "uas", "vas"]
    )
    custom_path: Optional[Path] = None  # For source: custom

    @field_validator("scenarios")
    @classmethod
    def validate_scenarios(cls, v):
        allowed = {"ssp126", "ssp245", "ssp370", "ssp585", "rcp26", "rcp45", "rcp85", "historical"}
        invalid = [s for s in v if s not in allowed]
        if invalid:
            raise ValueError(f"Unknown scenario(s): {invalid}. Must be one of {sorted(allowed)}")
        return v


class BiasCorrectionConfig(BaseModel):
    """Bias correction configuration for climate projections."""

    method: Literal["quantile_mapping", "detrended_qm", "scaled_distribution"] = "quantile_mapping"
    stratification: Literal["monthly", "seasonal", "annual"] = "monthly"
    n_quantiles: int = 100
    training_period: list[str] = Field(default_factory=lambda: ["1981-01-01", "2005-12-31"])
    evaluation_period: list[str] = Field(default_factory=lambda: ["2006-01-01", "2010-12-31"])

    @field_validator("n_quantiles")
    @classmethod
    def validate_n_quantiles(cls, v):
        if v < 10 or v > 10000:
            raise ValueError("n_quantiles must be between 10 and 10000")
        return v

    @field_validator("training_period", "evaluation_period")
    @classmethod
    def validate_period(cls, v):
        if v and len(v) != 2:
            raise ValueError("Period must be [start_date, end_date]")
        return v


class DisaggregationConfig(BaseModel):
    """Temporal disaggregation configuration (daily → hourly)."""

    method: Literal["melodist", "uniform"] = "melodist"


class EnsembleOutputConfig(BaseModel):
    """Ensemble output configuration."""

    statistics: list[str] = Field(default_factory=lambda: ["mean", "p10", "p25", "p50", "p75", "p90"])
    output_per_chain: bool = True

    @field_validator("statistics")
    @classmethod
    def validate_statistics(cls, v):
        allowed = {"mean", "std", "min", "max", "p5", "p10", "p25", "p50", "p75", "p90", "p95"}
        invalid = [s for s in v if s not in allowed]
        if invalid:
            raise ValueError(f"Unknown statistic(s): {invalid}. Must be one of {sorted(allowed)}")
        return v


class ClimateConfig(BaseModel):
    """Climate projections (TopoCLIM) configuration."""

    enabled: bool = False
    reference_period: list[str] = Field(default_factory=lambda: ["1981-01-01", "2010-12-31"])
    reference_source: Literal["era5", "da_reanalysis"] = "era5"
    reference_path: Optional[Path] = None  # Reuse existing reference
    models: ClimateModelsConfig = Field(default_factory=ClimateModelsConfig)
    bias_correction: BiasCorrectionConfig = Field(default_factory=BiasCorrectionConfig)
    disaggregation: DisaggregationConfig = Field(default_factory=DisaggregationConfig)
    projection_period: list[str] = Field(default_factory=lambda: ["2020-01-01", "2100-12-31"])
    ensemble: EnsembleOutputConfig = Field(default_factory=EnsembleOutputConfig)
    output_dir: Path = Path("./climate/")

    @field_validator("reference_period", "projection_period")
    @classmethod
    def validate_period(cls, v):
        if v and len(v) != 2:
            raise ValueError("Period must be [start_date, end_date]")
        return v


class TopoclimConfig(BaseModel):
    """Per-unit TopoCLIM (Fiddes, Aalstad & Lehning 2022) climate-scenario configuration.

    Drives ``tps2 topoclim``: reference (T-MET) daily aggregation, per-unit quantile
    mapping of local CMIP6 daily fields, disaggregation and FSM per model chain.
    """

    enabled: bool = False
    # hourly TPS2 reference forcing (T-MET), e.g. the 25-yr reanalysis forcing.zarr
    reference_forcing: Optional[str] = None  # path or glob of per-WY stores
    # directory holding <model>_<scenario>_<variant>.nc daily CMIP6 files
    cmip6_root: Optional[Path] = None
    # "all" or list of "MODEL:scenario" (e.g. "MPI-ESM1-2-HR:ssp245")
    chains: str | list[str] = "all"
    scenarios: list[str] = Field(default_factory=lambda: ["ssp245", "ssp585"])
    # QM training period (reference and GCM overlap; GCM historical is extended with
    # the chain's scenario after 2014, standard practice for CMIP6)
    training_period: list[str] = Field(default_factory=lambda: ["2000-01-01", "2024-12-31"])
    # optional independent evaluation period (only used by `topoclim evaluate`)
    evaluation_period: Optional[list[str]] = None
    # GCM series to bias-correct/store
    gcm_period: list[str] = Field(default_factory=lambda: ["1970-01-01", "2100-12-31"])
    # FSM analysis windows: list of [start, end]; each preceded by `spinup_years`
    fsm_periods: list[list[str]] = Field(
        default_factory=lambda: [["1985-10-01", "2014-09-30"], ["2030-10-01", "2060-09-30"], ["2070-10-01", "2100-09-30"]]
    )
    spinup_years: int = 1
    qstep: float = 0.01
    # variables using 12 monthly transfer functions (QM_MONTH); others annual
    monthly_variables: list[str] = Field(default_factory=lambda: ["pr", "rsds", "rlds"])
    lower_tail: Literal["constant", "delta"] = "constant"
    wet_day_threshold: Optional[float] = None  # mm/day; None = qmap default (off)
    unit_chunk: int = 2000
    reference_unit_chunk: int = 500  # smaller chunk for streaming the hourly reference
    output_dir: Path = Path("./topoclim/")
    keep_hourly_forcing: bool = False  # persist disaggregated hourly forcing per chain (large)

    @field_validator("training_period", "gcm_period")
    @classmethod
    def validate_period(cls, v):
        if v and len(v) != 2:
            raise ValueError("Period must be [start_date, end_date]")
        return v


class CalibrationConfig(BaseModel):
    """Parameter calibration configuration."""

    enabled: bool = False
    method: Literal["grid", "lhs", "bayesian", "none"] = "lhs"
    budget: int = 100
    objective: str = "kge"
    parameters: list[dict] = Field(default_factory=list)

    @field_validator("budget")
    @classmethod
    def validate_budget(cls, v):
        if v < 1:
            raise ValueError("budget must be at least 1")
        return v

    @field_validator("objective")
    @classmethod
    def validate_objective(cls, v):
        allowed = {"kge", "rmse", "fsca_rmse", "composite"}
        if v not in allowed:
            raise ValueError(f"objective must be one of {allowed}")
        return v


class FSM2VegConfig(BaseModel):
    """Per-unit vegetation parameters for FSM2."""

    source: Literal["dem", "landcover", "constant"] = "constant"
    default_height: float = 0.0  # m, for open terrain
    default_vai: float = 0.0  # Vegetation Area Index
    forest_height: float = 15.0  # m, for forested areas
    forest_vai: float = 3.0  # VAI for forest

    @field_validator("default_height", "forest_height")
    @classmethod
    def validate_height(cls, v):
        if v < 0:
            raise ValueError("Vegetation height must be non-negative")
        return v

    @field_validator("default_vai", "forest_vai")
    @classmethod
    def validate_vai(cls, v):
        if v < 0:
            raise ValueError("Vegetation Area Index must be non-negative")
        return v


class FSM1Config(BaseModel):
    """FSM1-specific configuration.

    FSM1 (Factorial Snow Model) is simpler than FSM2:
    - Single-point simulations (run once per unit)
    - No canopy model
    - Output averaging via Nave parameter
    - Physics options selected at runtime
    """

    enabled: bool = False
    backend: str = Field(default="rust", pattern=r"^(rust|fortran)$",
                         description="FSM1 backend: 'rust' (native, fast) or 'fortran' (subprocess)")
    binary_path: Optional[Path] = None

    # Output averaging: Nave timesteps averaged per output
    # e.g., Nave=8 for daily output from 3-hourly input
    nave: int = Field(default=8, ge=1, description="Number of timesteps to average in output")

    # Measurement heights
    temp_height: float = 2.0  # m
    wind_height: float = 10.0  # m

    # FSM1 snow/surface parameters (passed via &params namelist).
    # None = use FSM1 Fortran defaults. Set values to override.
    asmx: Optional[float] = None   # Max fresh snow albedo (default 0.8)
    asmn: Optional[float] = None   # Min melting snow albedo (default 0.5)
    hfsn: Optional[float] = None   # Snow cover fraction depth scale, m (default 0.1)
    rhof: Optional[float] = None   # Fresh snow density, kg/m³ (default 100)
    rcld: Optional[float] = None   # Max cold snow density, kg/m³ (default 300)
    rmlt: Optional[float] = None   # Max melting snow density, kg/m³ (default 500)
    trho: Optional[float] = None   # Snow compaction timescale, h (default 200)
    Salb: Optional[float] = None   # Snowfall to refresh albedo, kg/m² (default 10)
    Talb: Optional[float] = None   # Albedo decay temp threshold, °C (default -2)
    tcld: Optional[float] = None   # Cold snow albedo decay timescale, h (default 1000)
    tmlt: Optional[float] = None   # Melting snow albedo decay timescale, h (default 100)
    Wirr: Optional[float] = None   # Irreducible liquid water content (default 0.03)
    z0sn: Optional[float] = None   # Snow roughness length, m (default 0.01)
    z0sf: Optional[float] = None   # Snow-free roughness length, m (default 0.1)
    alb0: Optional[float] = None   # Snow-free ground albedo (default 0.2)
    rho0: Optional[float] = None   # Fixed snow density, kg/m³ (default 300)
    kfix: Optional[float] = None   # Fixed snow thermal conductivity, W/m/K (default 0.24)

    # Memory-safe chunking: when set, the Rust FSM1 kernel is called in batches
    # of this many units to bound memory (full-domain batches can use 8-12 GB).
    # None (default) = single full-domain call. Recommended: ~5000 for 8 GB hosts.
    chunk_size_units: Optional[int] = Field(
        default=None, ge=1,
        description="Max units per Rust FSM1 batch call; None = single batch (legacy).",
    )


class FSM2Config(BaseModel):
    """FSM2-specific configuration.

    FSM2 differs from FSM1:
    - Multi-point simulations (Npnts) in a single run
    - Full canopy model (interception, unloading, radiation)
    - Physics options are compile-time (preprocessor defines)
    - Three output files (flux, state, subcanopy)
    """

    enabled: bool = False
    binary_path: Optional[Path] = None

    # Physics options (compile-time in FSM2, we pre-compile variants)
    albedo: Literal[1, 2] = 2  # 1=diagnostic, 2=prognostic
    conductivity: Literal[0, 1] = 1  # 0=fixed, 1=variable
    density: Literal[0, 1, 2] = 1  # 0=fixed, 1=prognostic, 2=layered
    exchange: Literal[0, 1] = 1  # 0=neutral, 1=stability-corrected
    hydrology: Literal[0, 1, 2] = 1  # 0=free drain, 1=bucket, 2=Richards

    # Canopy options
    canopy_model: Literal[1, 2] = 1  # 1=zero-layer, 2=two-layer
    canopy_interception: Literal[1, 2] = 1  # 1=simple, 2=full
    canopy_radiation: Literal[1, 2] = 1  # 1=Beer's law, 2=two-stream

    # Vegetation parameters
    vegetation: FSM2VegConfig = Field(default_factory=FSM2VegConfig)

    # Measurement heights
    temp_height: float = 2.0  # m
    wind_height: float = 10.0  # m

    # Snow layer thicknesses [m]
    snow_layer_thicknesses: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.4]
    )
    # Soil layer thicknesses [m]
    soil_layer_thicknesses: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.5, 0.7]
    )

    @field_validator("temp_height", "wind_height")
    @classmethod
    def validate_measurement_height(cls, v):
        if v <= 0:
            raise ValueError("Measurement height must be positive")
        return v


class HBVConfig(BaseModel):
    """HBV model parameters (named, not positional)."""

    sfcf: float = Field(1.0, ge=0.5, le=1.5, description="Snowfall correction factor")
    cwh: float = Field(0.1, ge=0.0, le=0.2, description="Water holding capacity of snowpack")
    cfr: float = Field(0.05, ge=0.0, le=1.0, description="Refreezing coefficient")
    tt: float = Field(0.0, ge=-2.0, le=2.0, description="Threshold temperature [C]")
    fc: float = Field(200.0, ge=50.0, le=550.0, description="Field capacity [mm]")
    beta: float = Field(2.0, ge=1.0, le=6.0, description="Shape coefficient")
    lp: float = Field(0.6, ge=0.3, le=1.0, description="ET reduction threshold (fraction of FC)")
    k0: float = Field(0.3, ge=0.05, le=0.9, description="Fast recession [1/d]")
    k1: float = Field(0.1, ge=0.01, le=0.5, description="Interflow recession [1/d]")
    k2: float = Field(0.01, ge=0.001, le=0.1, description="Baseflow recession [1/d]")
    uzl: float = Field(20.0, ge=0.0, le=100.0, description="Upper zone threshold [mm]")
    perc: float = Field(1.0, ge=0.0, le=6.0, description="Max percolation [mm/d]")
    maxbas: float = Field(2.5, ge=1.0, le=7.0, description="Unit hydrograph length [d]")
    p_bias: float = Field(1.0, ge=0.5, le=1.5, description="Precipitation multiplier")
    et_bias: float = Field(1.0, ge=0.5, le=1.5, description="PET multiplier")


class GR4JConfig(BaseModel):
    """GR4J + CemaNeige parameters."""

    x1: float = Field(350.0, ge=100.0, le=1200.0, description="Production store capacity [mm]")
    x2: float = Field(0.0, ge=-5.0, le=3.0, description="Groundwater exchange [mm/d]")
    x3: float = Field(90.0, ge=20.0, le=300.0, description="Routing store capacity [mm]")
    x4: float = Field(1.7, ge=1.1, le=2.9, description="Unit hydrograph time base [d]")
    ctg: float = Field(0.5, ge=0.0, le=1.0, description="CemaNeige degree-day factor")
    kf: float = Field(0.5, ge=0.0, le=1.0, description="CemaNeige snowpack inertia")


class HydrologyConfig(BaseModel):
    """Hydrology application configuration."""

    model: Literal["hbv", "gr4j", "degreeday"] = "hbv"
    pet_method: Literal[
        "penman_monteith", "priestley_taylor", "hargreaves", "hamon"
    ] = "penman_monteith"
    hbv: HBVConfig = Field(default_factory=HBVConfig)
    gr4j: GR4JConfig = Field(default_factory=GR4JConfig)


class GlacierConfig(BaseModel):
    """Glacier-specific configuration for GFSM finite-ice mode."""

    finite_ice: bool = Field(
        default=False,
        description="Enable finite ice tracking (ice depletes via melt over time).",
    )
    ice_thickness_source: Literal["farinotti", "oggm", "custom", "none"] = Field(
        default="none",
        description=(
            "Source for initial ice thickness data. "
            "'none' uses infinite ice (legacy behavior)."
        ),
    )
    ice_thickness_path: Optional[Path] = Field(
        default=None,
        description="Path to raster (custom) or directory (farinotti/oggm).",
    )
    ice_albedo: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description=(
            "Albedo of exposed glacier ice. 0.4 is clean ice; observed Central "
            "Asian tongues reach ~0.26 by late summer, worth roughly 40 W/m2 of "
            "absorbed shortwave at midday. The most sensitive glacier-melt "
            "parameter in the model."
        ),
    )
    ice_roughness: float = Field(
        default=0.002,
        gt=0.0,
        le=1.0,
        description=(
            "Aerodynamic roughness length of exposed glacier ice (m). Melting "
            "ice measures 0.001-0.005 m. FSM1's bare-ground default of 0.1 m "
            "applied to ice overstates the turbulent exchange coefficient "
            "threefold, worth ~80 W/m2 of spurious sensible heat at 4 m/s."
        ),
    )
    volume_area_scaling: bool = Field(
        default=False,
        description="Enable volume-area scaling to update glacier area (post-processing).",
    )
    va_c: float = Field(
        default=0.0340,
        description="Volume-area scaling coefficient c (km3 vs km2).",
    )
    va_gamma: float = Field(
        default=1.375,
        description="Volume-area scaling exponent gamma.",
    )

    @model_validator(mode="after")
    def check_ice_thickness_source(self) -> "GlacierConfig":
        if self.finite_ice and self.ice_thickness_source == "none":
            raise ValueError(
                "glacier.finite_ice=true requires ice_thickness_source to be "
                "'farinotti', 'oggm', or 'custom' — with source 'none' the run "
                "would silently fall back to infinite ice."
            )
        if self.ice_thickness_source != "none" and self.ice_thickness_path is None:
            raise ValueError(
                f"glacier.ice_thickness_source='{self.ice_thickness_source}' "
                "requires ice_thickness_path to be set."
            )
        return self


class ApplicationConfig(BaseModel):
    """Application layer configuration."""

    model: Literal["fsm", "fsm2", "hydrology", "cryogrid", "snowpack", "none"] = "none"
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)
    state_da: StateDAConfig = Field(default_factory=StateDAConfig)
    model_path: Optional[Path] = None
    fsm1: FSM1Config = Field(default_factory=FSM1Config)
    fsm2: FSM2Config = Field(default_factory=FSM2Config)
    hydrology: HydrologyConfig = Field(default_factory=HydrologyConfig)
    glacier: GlacierConfig = Field(default_factory=GlacierConfig)


class NWPSourceConfig(BaseModel):
    """Individual NWP source configuration."""

    name: str
    type: Literal["era5", "ifs", "hres", "icon", "cosmo", "gfs", "custom"] = "era5"
    priority: int = 1
    resolution_m: Optional[float] = None
    path: Optional[Path] = None
    variable_mapping: dict[str, str] = Field(default_factory=dict)


class BlendingConfig(BaseModel):
    """Multi-source blending configuration."""

    enabled: bool = False
    method: Literal["priority", "weighted", "optimal"] = "priority"
    gap_fill: bool = True


class ForecastBlendingConfig(BaseModel):
    """Forecast-reanalysis temporal blending configuration."""

    enabled: bool = True
    method: Literal["crossfade", "hard_switch"] = "crossfade"
    crossfade_hours: int = 6
    reanalysis_source: str = "era5"

    @field_validator("crossfade_hours")
    @classmethod
    def validate_crossfade_hours(cls, v):
        if v < 1 or v > 48:
            raise ValueError("crossfade_hours must be between 1 and 48")
        return v


class ForecastConfig(BaseModel):
    """Forecast data source configuration."""

    enabled: bool = False
    backend: Literal["ecmwf_opendata", "openmeteo_ifs", "auto"] = "ecmwf_opendata"
    priority_strategy: Literal["speed", "reliability", "completeness", "backfill"] = "reliability"
    # ECMWF IFS product: "hres" (deterministic, default) or "ens" (ensemble; adds
    # a member dimension for probabilistic products).
    product: Literal["hres", "ens"] = "hres"
    # ENS member selection (ignored for HRES): "all" (the 50 perturbed members),
    # or a spec like "1-10" or "1,2,3". ECMWF open-data ENS publishes perturbed
    # members 1-50 only (no control); HRES is the deterministic run.
    ensemble_members: str = "all"
    forecast_hour: int = 0
    pressure_levels: list[int] = Field(
        default_factory=lambda: [1000, 850, 700, 500, 300]
    )
    output_timestep: str = "1H"
    output_dir: Path = Path("./forecast/")
    backfill_days: int = 3
    max_workers: int = 2
    # Where the ecmwf_opendata backend downloads from. ECMWF replicates open data
    # to cloud mirrors; a host that cannot reach data.ecmwf.int (e.g. the
    # Kazhydromet server, whose firewall blocks it) can use one that it can.
    # "azure" fetches a short-lived SAS token from planetarycomputer.microsoft.com.
    opendata_source: Literal["ecmwf", "azure", "aws", "google"] = "ecmwf"
    blending: ForecastBlendingConfig = Field(default_factory=ForecastBlendingConfig)

    _VALID_TIMESTEPS = {"1H", "2H", "3H", "6H", "1D"}

    @field_validator("forecast_hour")
    @classmethod
    def validate_forecast_hour(cls, v):
        if v not in (0, 6, 12, 18):
            raise ValueError("forecast_hour must be 0, 6, 12, or 18")
        return v

    @staticmethod
    def parse_members(spec: str) -> list[int]:
        """Resolve an ``ensemble_members`` spec to a sorted member-id list (1..50).

        "all" -> 1..50; "a-b" -> a..b; "a,b,c" -> [a,b,c]. ECMWF open-data ENS
        has perturbed members 1-50 only (no control / member 0).
        """
        spec = (spec or "all").strip().lower()
        if spec == "all":
            members = list(range(1, 51))
        elif "-" in spec:
            a, b = spec.split("-", 1)
            members = list(range(int(a), int(b) + 1))
        else:
            members = [int(x) for x in spec.split(",") if x.strip() != ""]
        if not members or any(m < 1 or m > 50 for m in members):
            raise ValueError(f"ensemble_members '{spec}' must resolve to ids in 1..50")
        return sorted(set(members))

    @field_validator("ensemble_members")
    @classmethod
    def validate_ensemble_members(cls, v):
        ForecastConfig.parse_members(v)  # raises on a bad spec
        return v

    def resolved_members(self) -> list[int]:
        """Member ids for this config (``[]`` for HRES)."""
        return [] if self.product == "hres" else self.parse_members(self.ensemble_members)

    @field_validator("output_timestep")
    @classmethod
    def validate_output_timestep(cls, v):
        allowed = {"1H", "2H", "3H", "6H", "1D"}
        if v not in allowed:
            raise ValueError(f"output_timestep must be one of {allowed}")
        return v


# =============================================================================
# Original Phase 1 Config Sections (updated where needed)
# =============================================================================


class PointCoordinate(BaseModel):
    """A named point coordinate for points spatial mode."""

    name: str
    lon: float
    lat: float
    elevation: Optional[float] = None  # If None, extracted from DEM


class PointsConfig(BaseModel):
    """Configuration for explicit point locations."""

    coordinates: list[PointCoordinate] = Field(default_factory=list)


class CatchmentConfig(BaseModel):
    """Catchment-based spatial unit configuration (e.g. HydroSHEDS)."""

    shapefile: Path
    id_column: str = "HYBAS_ID"
    downstream_column: str = "NEXT_DOWN"
    area_column: str = "SUB_AREA"


class DomainConfig(BaseModel):
    """Spatial domain configuration."""

    dem: Optional[Path] = None
    bbox: Optional[list[float]] = None
    dem_source: str = "glo_90"
    spatial_mode: str = "clusters"
    grid_resolution: Optional[float] = None
    crs: Optional[str | int] = None
    # Horizon angle computation parameters
    horizon_n_directions: int = 8  # Number of azimuth directions (8 = every 45°)
    horizon_max_distance_m: float = 3000.0  # Maximum search distance in meters
    # SVF computation mode: "full" (all pixels), "centroid" (cluster centroids only), "approximate" (cos(slope))
    svf_mode: Literal["full", "centroid", "approximate"] = "centroid"
    # Points mode configuration
    points: Optional[PointsConfig] = None
    # Buffer radius (m) around each point for DEM patch download in sparse points mode.
    # None = auto (horizon_max_distance_m + 1000)
    dem_buffer_m: Optional[float] = None
    # Optional polygon shapefile restricting the modelling domain to its footprint.
    # When set, clustering only covers pixels inside the polygon union (e.g. a
    # basin shapefile); the DEM is still downloaded over bbox/extent. Relative
    # paths resolve against the config file directory.
    mask_shapefile: Optional[Path] = None
    # Optional buffer (m) applied to the mask polygons before rasterizing, to
    # avoid clipping basin-edge pixels. None = no buffer.
    mask_buffer_m: Optional[float] = None

    @model_validator(mode="after")
    def check_dem_or_bbox(self):
        if self.dem is None and self.bbox is None:
            # Allow bbox=None when sparse points mode: coordinates will be used
            # to auto-derive per-point DEM patches
            if (
                self.spatial_mode == "points"
                and self.points is not None
                and self.points.coordinates
            ):
                return self
            raise ValueError("Either 'dem' (path) or 'bbox' (for download) must be provided")
        return self

    @field_validator("spatial_mode")
    @classmethod
    def validate_spatial_mode(cls, v):
        allowed = {"clusters", "polygons", "grid", "points", "catchments"}
        if v not in allowed:
            raise ValueError(f"spatial_mode must be one of {allowed}")
        return v

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if v is not None:
            if len(v) != 4:
                raise ValueError("bbox must be [west, south, east, north]")
            west, south, east, north = v
            if south >= north:
                raise ValueError("south must be less than north")
            if west >= east:
                raise ValueError("west must be less than east")
        return v


class ClusteringConfig(BaseModel):
    """TopoSUB clustering configuration."""

    method: str = "kmeans"
    features: list[str] = Field(
        default=["elevation", "slope", "cos_aspect", "sin_aspect", "svf"]
    )
    n_clusters: int = 100
    feature_standardization: bool = True
    # MiniBatchKMeans batch size (used automatically when samples > 500K)
    # None = auto (256 * n_cores), or specify explicitly for tuning
    minibatch_batch_size: Optional[int] = None
    # Feature weights: multiply each feature by its weight AFTER standardization
    # If None, all features have equal weight (1.0)
    # Length must match features list
    feature_weights: Optional[list[float]] = None

    # Terrain-adaptive clustering: split domain by slope threshold into
    # flat (x,y only) and mountain (full feature set) zones, allocating
    # more clusters to mountains where elevation/aspect differentiation matters.
    terrain_adaptive: bool = False
    # Slope threshold (degrees) separating flat from mountain terrain
    terrain_slope_threshold: float = 5.0
    # Fraction of n_clusters allocated to mountain zone (rest go to flat)
    terrain_mountain_fraction: float = 0.75

    @model_validator(mode="after")
    def check_weights_length(self):
        if self.feature_weights is not None:
            if len(self.feature_weights) != len(self.features):
                raise ValueError(
                    f"feature_weights length ({len(self.feature_weights)}) "
                    f"must match features length ({len(self.features)})"
                )
        return self

    @field_validator("method")
    @classmethod
    def validate_method(cls, v):
        allowed = {"kmeans", "som", "gaussian_mixture"}
        if v not in allowed:
            raise ValueError(f"method must be one of {allowed}")
        return v


class SurfaceTypeEntry(BaseModel):
    """Configuration for a single surface type."""

    n_clusters: Optional[int] = None
    extra_features: list[str] = Field(default_factory=list)
    physics: str = "standard"
    action: Optional[str] = None


class SurfaceTypeConfig(BaseModel):
    """Surface type classification configuration."""

    source: str = "custom"
    stratify: bool = True
    types: dict[str, SurfaceTypeEntry] = Field(default_factory=dict)
    raster_path: Optional[Path] = Field(
        default=None,
        description="Path to custom surface type raster (GeoTIFF). "
        "Values: 0=open, 1=glacier, 2=forest, 3=rock, 4=water.",
    )
    glacier_source: str = Field(
        default="none",
        description="Glacier classification source: 'rgi', 'custom', or 'none'",
    )
    rgi_version: str = "7.0"
    rgi_regions: list[str] = Field(
        default_factory=list,
        description=(
            "NSIDC RGI region ids to load when glacier_source='rgi', e.g. "
            "['13_central_asia', '14_south_asia_west']. Required for that source."
        ),
    )
    rgi_cache: Optional[Path] = Field(
        default=None,
        description="RGI download cache directory (default ~/.cache/tps2/rgi7).",
    )

    @model_validator(mode="after")
    def check_rgi_regions(self) -> "SurfaceTypeConfig":
        if self.glacier_source == "rgi" and not self.rgi_regions:
            raise ValueError(
                "surface_types.glacier_source='rgi' requires rgi_regions, "
                "e.g. ['13_central_asia'] -- otherwise no outlines are loaded "
                "and every pixel silently stays non-glacier."
            )
        return self

    @field_validator("source")
    @classmethod
    def validate_source(cls, v):
        allowed = {"esa_worldcover", "copernicus", "custom"}
        if v not in allowed:
            raise ValueError(f"source must be one of {allowed}")
        return v

    @field_validator("glacier_source")
    @classmethod
    def validate_glacier_source(cls, v):
        allowed = {"rgi", "custom", "none"}
        if v not in allowed:
            raise ValueError(f"glacier_source must be one of {allowed}")
        return v


class InputConfig(BaseModel):
    """Input data source configuration."""

    primary: str = "era5"
    backend: str = "google"
    time_range: list[str] = Field(default_factory=list)
    pressure_levels: list[int] = Field(default=[300, 500, 700, 850, 1000])
    time_resolution: str = "1H"
    output_format: str = "zarr"  # zarr for efficient chunked I/O
    max_workers: int = 4
    cache_dir: Path = Path("./inputs/climate/")
    # S3 Zarr backend config
    s3_zarr_url: Optional[str] = None  # e.g. "s3://bucket/era5.zarr/"
    # Local Zarr cache path (built by `tps2 build-cache`)
    cache_path: Optional[Path] = None  # e.g. "/mnt/server/data/era5/hma.zarr"
    # hybrid backend: precipitation model. "era5" (default) keeps every variable from
    # ERA5; "ifs" takes precipitation from the IFS 9 km analysis via Open-Meteo. Never
    # chosen from the dates or the domain — it is only what this key says.
    hybrid_precip_model: Literal["era5", "ifs"] = Field(
        "era5", description="hybrid backend precipitation model: era5 (default) or ifs (9 km IFS via Open-Meteo)",
    )
    # IFS backend config: openmeteo (2022+), ecmwf_opendata (last ~3 days), auto
    ifs_backend: Literal["openmeteo", "ecmwf_opendata", "auto"] = "auto"
    # Phase 2: Multi-NWP support
    sources: list[NWPSourceConfig] = Field(default_factory=list)
    blending: BlendingConfig = Field(default_factory=BlendingConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        allowed = {"google", "cds", "s3zarr", "openmeteo", "hybrid", "local"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}")
        return v

    @field_validator("time_resolution")
    @classmethod
    def validate_time_resolution(cls, v):
        allowed = {"1H", "2H", "3H", "6H"}
        if v not in allowed:
            raise ValueError(f"time_resolution must be one of {allowed}")
        return v

    @field_validator("time_range")
    @classmethod
    def validate_time_range(cls, v):
        if v and len(v) != 2:
            raise ValueError("time_range must be [start_date, end_date]")
        return v


class DownscalingConfig(BaseModel):
    """Downscaling method selection."""

    # Mode: "simple" uses only surface data (ERA5-Land compatible),
    # "full" uses pressure-level interpolation (requires ERA5 pressure levels)
    mode: Literal["simple", "full"] = "full"
    # Lapse rate for simple mode [K/m], default 6.5 K/km
    lapse_rate: float = 0.0065
    # The next four keys are INFORMATIONAL ONLY: they are recorded and shown by `tps2 info`,
    # but the engine does not read them. Temperature follows `mode` (pressure levels in full
    # mode, `lapse_rate` in simple mode); radiation is always the slope/aspect/horizon/sky-view
    # correction; wind is the log profile, plus Winstral exposure if
    # `wind_config.method: winstral`; humidity is specific humidity, with RH recomputed.
    temperature: str = "lapse_rate"
    radiation: str = "toposcale"
    wind: str = "log_profile"
    # Precipitation downscaling method. Default "none" = pass source precip
    # through unchanged (no elevation gradient). A fixed precip-elevation
    # gradient is poorly constrained in high-relief terrain, so it is opt-in:
    # set "elevation_gradient" and tune `precip_gradient` to enable it.
    precipitation: str = "none"
    # Fractional precip increase per metre of elevation gain. Only applied when
    # precipitation == "elevation_gradient". 0.0003 = +3%/100m (legacy default).
    precip_gradient: float = 0.0003
    # Rain/snow phase partitioning variable. "wet_bulb" (default) uses a
    # pressure-aware psychrometric wet-bulb temperature — a better predictor than
    # air temperature in dry/high terrain, where snow falls well above 0degC air
    # temp. "air_temperature" restores the legacy 2 m air-temperature split.
    phase_method: str = "wet_bulb"
    # Rain/snow thresholds [degC] for the linear mixed-phase ramp: all snow below
    # t_snow_threshold_c, all rain above t_rain_threshold_c. Defaults are
    # recentred for wet-bulb (which runs below air temperature).
    t_snow_threshold_c: float = -0.5
    t_rain_threshold_c: float = 2.5
    # Air-temperature guard rail [degC]: precipitation is forced to all-rain above
    # this air temperature regardless of the wet-bulb temperature. Prevents snow at
    # implausibly warm air temps in very dry air (where wet-bulb can sit below
    # freezing). Set high (e.g. 99) to disable. No-op for "air_temperature".
    t_air_max_c: float = 4.0
    humidity: str = "standard"   # informational only, see above
    # Phase 2: Wind and redistribution configs
    wind_config: WindConfig = Field(default_factory=WindConfig)
    redistribution: RedistributionConfig = Field(default_factory=RedistributionConfig)

    @field_validator("phase_method")
    @classmethod
    def validate_phase_method(cls, v):
        if v not in {"wet_bulb", "air_temperature"}:
            raise ValueError(
                "phase_method must be 'wet_bulb' or 'air_temperature'"
            )
        return v

    @model_validator(mode="after")
    def validate_phase_thresholds(self):
        if self.t_rain_threshold_c <= self.t_snow_threshold_c:
            raise ValueError(
                "t_rain_threshold_c must be greater than t_snow_threshold_c"
            )
        if self.t_air_max_c < self.t_rain_threshold_c:
            raise ValueError(
                "t_air_max_c must be >= t_rain_threshold_c "
                "(the guard rail should not clip below the rain threshold)"
            )
        return self

    @field_validator("lapse_rate")
    @classmethod
    def validate_lapse_rate(cls, v):
        if v < 0 or v > 0.02:
            raise ValueError("lapse_rate must be between 0 and 0.02 K/m (0-20 K/km)")
        return v


VALID_OUTPUT_VARIABLES = frozenset({
    "temperature", "pressure", "precipitation", "rainfall", "snowfall",
    "shortwave_direct", "shortwave_diffuse", "longwave",
    "humidity_specific", "humidity_relative",
    "wind_speed", "wind_direction",
})


class OutputConfig(BaseModel):
    """Output writer configuration."""

    format: str = "netcdf"
    directory: Path = Path("./output/")
    variables: list[str] = Field(
        default_factory=list,
        description="Output variables to compute. Empty list = all computable from available inputs.",
    )
    temporal_resolution: str = "hourly"
    compression: bool = True

    @field_validator("format")
    @classmethod
    def validate_format(cls, v):
        # Phase 2: Extended format support
        allowed = {"netcdf", "fsm", "csv", "smet", "hbv", "cryogrid", "crocus", "zarr"}
        if v not in allowed:
            raise ValueError(f"format must be one of {allowed}")
        return v

    @field_validator("variables")
    @classmethod
    def validate_variables(cls, v):
        if v:
            invalid = [var for var in v if var not in VALID_OUTPUT_VARIABLES]
            if invalid:
                raise ValueError(
                    f"Invalid output variable(s): {invalid}. "
                    f"Valid options: {sorted(VALID_OUTPUT_VARIABLES)}"
                )
        return v


class ExecutionConfig(BaseModel):
    """Execution backend configuration."""

    backend: str = "local"
    n_workers: int = 4
    kernel_backend: str = "rust"
    chunks_time: str = "monthly"
    chunks_space: int = 100
    # Phase 2: SLURM support
    slurm: SLURMConfig = Field(default_factory=SLURMConfig)

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v):
        allowed = {"local", "dask", "slurm"}
        if v not in allowed:
            raise ValueError(f"backend must be one of {allowed}")
        return v

    @field_validator("kernel_backend")
    @classmethod
    def validate_kernel_backend(cls, v):
        allowed = {"python", "rust", "jax"}
        if v not in allowed:
            raise ValueError(f"kernel_backend must be one of {allowed}")
        return v


class ValidationStationsConfig(BaseModel):
    """Station validation configuration."""

    metrics: list[str] = Field(default=["bias", "rmse", "kge", "mae"])
    by_season: bool = True
    by_elevation_band: bool = True


class StationSourceConfig(BaseModel):
    """Configuration for a single station data source."""

    backend: Literal["isd", "ghcnd", "meteoswiss", "imis"] = "isd"
    variables: list[str] = Field(default_factory=list)
    min_years: float = 0.0
    max_stations: Optional[int] = None
    # Local directory of already-downloaded files (meteoswiss / imis only).
    data_dir: Optional[Path] = None
    # Extra backend constructor options, e.g. {"resolution": "daily"} or
    # {"time_label": "interval_start"}; see the backend's docstring.
    options: dict[str, Any] = Field(default_factory=dict)


class StationDataConfig(BaseModel):
    """Configuration for automated station data retrieval."""

    enabled: bool = False
    sources: list[StationSourceConfig] = Field(
        default_factory=lambda: [StationSourceConfig(backend="isd")]
    )
    cache_dir: Path = Path("./station_cache/")
    max_workers: int = 4
    run_qc: bool = True
    max_distance_m: float = 50000.0
    max_elevation_diff_m: float = 500.0


class ValidationConfig(BaseModel):
    """Forcing validation configuration."""

    stations: ValidationStationsConfig = Field(default_factory=ValidationStationsConfig)
    station_data: StationDataConfig = Field(default_factory=StationDataConfig)
    output_dir: Path = Path("./validation/")


class ValidationLabConfig(BaseModel):
    """Validation Lab configuration (see docs/validation_lab.md).

    Names the registry domain, the datasets and protocols enabled here, and
    where scorecards land. One of these lives beside each operational domain's
    run config so `tps2 lab run` is reproducible per domain.
    """

    enabled: bool = False
    #: Registry domain name ("alps", "central_asia", "kazakhstan", "nepal", "europe").
    domain: Optional[str] = None
    #: The operational run's config, read **read-only** for its domain definition
    #: and output locations. Set this and the lab config can live entirely
    #: outside the operational tree: the lab reads that run's outputs and its
    #: cached spatial units, and writes only into its own scorecard/cache dirs.
    #: When unset, the lab config must itself describe the domain.
    run_config: Optional[Path] = None
    #: Dataset bundle YAML files to merge with the core registry. Bench and
    #: site datasets live outside the engine repo (see tps2_benches); this is how
    #: a run declares which of them are in force. Relative paths resolve against
    #: this config file.
    registry_paths: list[Path] = Field(default_factory=list)
    #: Registry dataset ids to score against. Empty = every applicable dataset.
    datasets: list[str] = Field(default_factory=list)
    #: Named holdout protocols (see topopyscale2.validation.protocols).
    protocols: list[str] = Field(default_factory=lambda: ["full"])
    #: Baseline product ids scored alongside TPS2.
    baselines: list[str] = Field(default_factory=lambda: ["era5_raw"])
    scorecard_dir: Path = Path("./scorecards/")
    cache_dir: Path = Path("./lab_cache/")
    #: Identifier of the simulation being scored, recorded in scorecard provenance.
    sim_id: Optional[str] = None
    #: Label for the forcing this run used, recorded in scorecard provenance. The
    #: field that distinguishes two runs of the same domain and suite from each
    #: other — e.g. "europe5" vs "full20" for a pressure-level experiment. Without
    #: it such runs differ only by config_hash, which is not human-readable.
    forcing_version: Optional[str] = None
    #: DA layers active in the scored run (e.g. ["layer1_precip"]), recorded in
    #: scorecard provenance so a corrected run is never mistaken for a raw one.
    da_layers: list[str] = Field(default_factory=list)
    n_bootstrap: int = 200
    seed: int = 0
    #: Snow-model output to score (NetCDF/Zarr with dims time, unit_id).
    #: Defaults to <output.directory>/fsm_output.nc when unset.
    model_output: Optional[Path] = None
    #: Registry dataset id -> path of its normalised observation cache. See
    #: topopyscale2.validation.observations for the two cache formats.
    observations: dict[str, Path] = Field(default_factory=dict)
    #: Build a day-of-year climatology baseline from the model output itself.
    #: Needs a multi-year run; skipped with a warning if the record is shorter.
    climatology_baseline: bool = True
    #: fSCA above which a unit counts as snow-covered (binary + timing metrics).
    fsca_threshold: float = 0.15
    #: SWE [mm] above which albedo is scored as snow-covered.
    snow_mask_swe: float = 10.0
    #: Units of snow depth in the station caches ("m" or "cm").
    station_depth_units: Literal["m", "cm"] = "m"

    @field_validator("observations")
    @classmethod
    def validate_observation_datasets(cls, v):
        try:
            from topopyscale2.validation.registry import REGISTRY
        except ImportError:  # validation not installed: block ignored (see FEATURE_MODULES)
            return v

        unknown = [d for d in v if d not in REGISTRY]
        if unknown:
            raise ValueError(
                f"unknown dataset id(s) in validation_lab.observations: {unknown}; "
                f"known ids: {sorted(REGISTRY)}"
            )
        return v

    @field_validator("protocols")
    @classmethod
    def validate_protocols(cls, v):
        try:
            from topopyscale2.validation.protocols import ProtocolError, get_protocol
        except ImportError:  # validation not installed: block ignored (see FEATURE_MODULES)
            return v

        for name in v:
            try:
                get_protocol(name)
            except ProtocolError as exc:
                raise ValueError(str(exc)) from None
        return v

    @model_validator(mode="after")
    def validate_against_registry(self):
        """Check domain and dataset ids against the *merged* registry.

        This cannot be a per-field validator: datasets may be declared in bundle
        YAML named by ``registry_paths``, which is a sibling field, so the set of
        legal domains and ids is only knowable once the whole model is bound.

        A nested model cannot see the config file's directory, so a *relative*
        bundle path is not resolvable here. Those are deferred to the CLI, which
        resolves against the config and then re-runs this same check — see
        `lab_cli._activate_registry`. Absolute paths are validated here and now.
        Deferring is stated rather than silent: the alternative is rejecting a
        domain that a bundle does declare, which reads as a config error when it
        is not one.
        """
        try:
            from topopyscale2.validation.registry import BundleError, build_registry
        except ImportError:  # validation not installed: block ignored (see FEATURE_MODULES)
            return self

        paths = [Path(p).expanduser() for p in self.registry_paths]
        deferred = [p for p in paths if not p.is_absolute()]

        try:
            merged = build_registry([p for p in paths if p.is_absolute()])
        except BundleError as exc:
            raise ValueError(str(exc)) from None

        if self.domain is not None and not deferred:
            known = merged.known_domains
            if self.domain not in known:
                raise ValueError(
                    f"domain must be one of {sorted(known)}, got '{self.domain}'. "
                    f"A domain declared in a bundle needs that bundle listed in "
                    f"validation_lab.registry_paths."
                )

        unknown = [] if deferred else [d for d in self.datasets if d not in merged.entries]
        if unknown:
            raise ValueError(
                f"unknown dataset id(s) {unknown}; known ids: {sorted(merged.entries)}"
            )

        unknown_obs = (
            [] if deferred else [d for d in self.observations if d not in merged.entries]
        )
        if unknown_obs:
            raise ValueError(
                f"unknown dataset id(s) in validation_lab.observations: {unknown_obs}; "
                f"known ids: {sorted(merged.entries)}"
            )

        return self


class TPS2Config(BaseModel):
    """Root configuration for TopoPyScale 2.0."""

    # Water-year calendar (topopyscale2.water_year): first month of the water year,
    # labelled by the year it ends in. 9 = 1 September (northern hemisphere); set e.g.
    # 3 for a southern-hemisphere domain. Climatology stores record the month they were
    # built with, and are always read on their own calendar.
    water_year_start_month: int = Field(
        9, ge=1, le=12,
        description="First month of the water year (9 = 1 Sep; e.g. 3 for the southern hemisphere)",
    )

    domain: DomainConfig
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    surface_types: SurfaceTypeConfig = Field(default_factory=SurfaceTypeConfig)
    inputs: InputConfig = Field(default_factory=InputConfig)
    downscaling: DownscalingConfig = Field(default_factory=DownscalingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    validation_lab: ValidationLabConfig = Field(default_factory=ValidationLabConfig)
    # Phase 2: New config sections
    polygons: PolygonConfig = Field(default_factory=PolygonConfig)
    catchments: Optional[CatchmentConfig] = None
    qc: QCConfig = Field(default_factory=QCConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    application: ApplicationConfig = Field(default_factory=ApplicationConfig)
    forecast: ForecastConfig = Field(default_factory=ForecastConfig)
    da: DAConfig = Field(default_factory=DAConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    climatology: ClimatologyConfig = Field(default_factory=ClimatologyConfig)
    climate: ClimateConfig = Field(default_factory=ClimateConfig)
    topoclim: TopoclimConfig = Field(default_factory=TopoclimConfig)

    @model_validator(mode="after")
    def _note_absent_features(self):
        """Name, once, each block given for a feature this installation lacks."""
        absent = [b for b in FEATURE_MODULES
                  if b in self.model_fields_set and not feature_installed(FEATURE_MODULES[b])]
        if absent:
            logger.warning(
                "config block(s) %s configure features not included in this release "
                "and are ignored; see the Roadmap in the documentation.",
                ", ".join(f"'{b}:'" for b in absent),
            )
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TPS2Config":
        """Load configuration from a YAML file."""
        path = Path(path).resolve()
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        config = cls(**data)
        config._config_dir = path.parent
        return config
