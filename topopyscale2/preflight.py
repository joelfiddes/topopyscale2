"""Preflight checks for TPS2 simulations.

Validates configuration, estimates runtime and data volumes, and checks
backend availability before committing to a multi-hour pipeline run.
"""

import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from topopyscale2.config.schema import TPS2Config

# Standard ERA5 pressure levels (hPa)
ERA5_STANDARD_LEVELS = frozenset({
    1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 775, 800, 825, 850,
    875, 900, 925, 950, 975, 1000,
})

# ECMWF OpenData default pressure levels (hPa)
ECMWF_OPENDATA_LEVELS = frozenset({1000, 925, 850, 700, 600, 500, 400, 300})

# Open-Meteo IFS available pressure levels (hPa)
OPENMETEO_IFS_LEVELS = frozenset({
    1000, 975, 950, 925, 900, 875, 850, 800, 750, 700,
    650, 600, 550, 500, 450, 400, 350, 300, 250, 200,
    150, 100, 70, 50, 30,
})

# Levels available in ALL forecast backends (safe intersection)
FORECAST_COMMON_LEVELS = ECMWF_OPENDATA_LEVELS & OPENMETEO_IFS_LEVELS

# Standard atmosphere: pressure (hPa) -> approximate height (m)
PRESSURE_TO_HEIGHT = {
    1000: 100, 975: 300, 950: 500, 925: 750, 900: 1000,
    875: 1250, 850: 1500, 825: 1750, 800: 2000, 775: 2250,
    750: 2500, 700: 3000, 650: 3600, 600: 4200, 550: 4900,
    500: 5500, 450: 6200, 400: 7200, 350: 8100, 300: 9200,
    250: 10400, 200: 11800, 150: 13600, 100: 16200,
}

# DEM resolution in degrees per pixel (approximate)
DEM_RESOLUTION_DEG = {
    "glo_90": 90 / 111_000,    # ~0.00081°
    "glo_30": 30 / 111_000,    # ~0.00027°
    "srtm_v3": 30 / 111_000,
    "nasadem": 30 / 111_000,
}

# Hours per day by time resolution string
HOURS_PER_DAY = {"1H": 24, "2H": 12, "3H": 8, "6H": 4}


@dataclass
class CheckResult:
    """Result of a single preflight check."""

    status: Literal["pass", "warn", "fail"]
    message: str
    category: str = ""


@dataclass
class StageEstimate:
    """Runtime and data volume estimate for a pipeline stage."""

    stage: str
    data_volume_gb: float
    time_min_low: float
    time_min_high: float

    @property
    def time_str(self) -> str:
        if self.time_min_low < 1 and self.time_min_high < 1:
            return f"~{self.time_min_high * 60:.0f}s"
        if self.time_min_low == self.time_min_high:
            return f"~{self.time_min_low:.0f} min"
        return f"~{self.time_min_low:.0f}-{self.time_min_high:.0f} min"

    @property
    def volume_str(self) -> str:
        if self.data_volume_gb < 0.01:
            return f"~{self.data_volume_gb * 1000:.0f} MB"
        return f"~{self.data_volume_gb:.1f} GB"


@dataclass
class PreflightReport:
    """Complete preflight report."""

    checks: list[CheckResult] = field(default_factory=list)
    estimates: list[StageEstimate] = field(default_factory=list)

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.status == "pass")

    @property
    def n_warn(self) -> int:
        return sum(1 for c in self.checks if c.status == "warn")

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if c.status == "fail")

    @property
    def ok(self) -> bool:
        return self.n_fail == 0

    @property
    def total_volume_gb(self) -> float:
        return sum(e.data_volume_gb for e in self.estimates)

    @property
    def total_time_low(self) -> float:
        return sum(e.time_min_low for e in self.estimates)

    @property
    def total_time_high(self) -> float:
        return sum(e.time_min_high for e in self.estimates)


def _pressure_to_height(level_hpa: int) -> float:
    """Approximate height (m) for a pressure level using standard atmosphere."""
    if level_hpa in PRESSURE_TO_HEIGHT:
        return PRESSURE_TO_HEIGHT[level_hpa]
    # Linear interpolation between known levels
    sorted_levels = sorted(PRESSURE_TO_HEIGHT.keys(), reverse=True)
    for i in range(len(sorted_levels) - 1):
        p_low, p_high = sorted_levels[i + 1], sorted_levels[i]
        if p_low <= level_hpa <= p_high:
            h_low = PRESSURE_TO_HEIGHT[p_high]
            h_high = PRESSURE_TO_HEIGHT[p_low]
            frac = (p_high - level_hpa) / (p_high - p_low)
            return h_low + frac * (h_high - h_low)
    # Extrapolate for very high/low levels
    if level_hpa > sorted_levels[0]:
        return 0.0
    return PRESSURE_TO_HEIGHT[sorted_levels[-1]]


def _check_bbox(cfg: TPS2Config) -> list[CheckResult]:
    """Check bounding box validity and sanity."""
    results = []
    bbox = cfg.domain.bbox
    if bbox is None:
        if cfg.domain.dem is not None:
            results.append(CheckResult("pass", "Using user-provided DEM (no bbox)", "Domain"))
        elif (
            cfg.domain.spatial_mode == "points"
            and cfg.domain.points is not None
            and cfg.domain.points.coordinates
        ):
            results.append(CheckResult(
                "pass",
                f"Sparse points mode: {len(cfg.domain.points.coordinates)} points (DEM auto-derived)",
                "Domain",
            ))
        else:
            results.append(CheckResult("fail", "No bbox and no DEM path specified", "Domain"))
        return results

    west, south, east, north = bbox

    # Coordinate ranges
    if not (-180 <= west <= 180 and -180 <= east <= 180):
        results.append(CheckResult("fail", f"Longitude out of range: west={west}, east={east}", "Domain"))
    elif not (-90 <= south <= 90 and -90 <= north <= 90):
        results.append(CheckResult("fail", f"Latitude out of range: south={south}, north={north}", "Domain"))
    elif south >= north:
        results.append(CheckResult("fail", f"Inverted bbox: south ({south}) >= north ({north})", "Domain"))
    elif west >= east:
        results.append(CheckResult("fail", f"Inverted bbox: west ({west}) >= east ({east})", "Domain"))
    else:
        span_lon = east - west
        span_lat = north - south
        results.append(CheckResult(
            "pass",
            f"Bbox valid: [{west}, {south}, {east}, {north}] ({span_lon:.1f}\u00b0 \u00d7 {span_lat:.1f}\u00b0)",
            "Domain",
        ))
        if span_lon > 30 or span_lat > 30:
            results.append(CheckResult(
                "warn",
                f"Very large domain ({span_lon:.1f}\u00b0 \u00d7 {span_lat:.1f}\u00b0) \u2014 expect long runtimes",
                "Domain",
            ))
        if span_lon < 0.01 or span_lat < 0.01:
            results.append(CheckResult(
                "warn",
                f"Very small domain ({span_lon:.4f}\u00b0 \u00d7 {span_lat:.4f}\u00b0) \u2014 may have too few DEM pixels",
                "Domain",
            ))

    return results


def _check_time_range(cfg: TPS2Config) -> list[CheckResult]:
    """Check time range validity."""
    results = []
    tr = cfg.inputs.time_range

    if not tr or len(tr) != 2:
        results.append(CheckResult("fail", "time_range must be [start, end]", "Time & Data"))
        return results

    try:
        start = datetime.strptime(tr[0], "%Y-%m-%d")
        end = datetime.strptime(tr[1], "%Y-%m-%d")
    except ValueError as e:
        results.append(CheckResult("fail", f"Cannot parse time_range dates: {e}", "Time & Data"))
        return results

    if start >= end:
        results.append(CheckResult("fail", f"start ({tr[0]}) >= end ({tr[1]})", "Time & Data"))
        return results

    n_days = (end - start).days + 1
    hpd = HOURS_PER_DAY.get(cfg.inputs.time_resolution, 24)
    n_timesteps = n_days * hpd

    results.append(CheckResult(
        "pass",
        f"Time range: {tr[0]} to {tr[1]} ({n_days} days, {n_timesteps:,} timesteps)",
        "Time & Data",
    ))

    now = datetime.now()
    if end > now:
        results.append(CheckResult("warn", f"End date ({tr[1]}) is in the future", "Time & Data"))

    years = n_days / 365.25
    if years > 3:
        results.append(CheckResult(
            "warn",
            f"Long time range ({years:.1f} years) \u2014 expect large data volumes",
            "Time & Data",
        ))

    # Estimate forcing memory footprint and warn if chunking will be needed
    n_lat, n_lon, n_grid = _compute_era5_grid(cfg)
    n_levels = len(cfg.inputs.pressure_levels)
    n_surf_vars = 10  # typical ERA5 surface vars
    n_plev_vars = 6   # typical ERA5 pressure level vars
    forcing_bytes = n_timesteps * n_grid * 4 * (n_surf_vars + n_plev_vars * n_levels)
    forcing_gb = forcing_bytes / 1e9

    try:
        import psutil
        available_ram_gb = psutil.virtual_memory().total / 1e9
    except ImportError:
        available_ram_gb = 32.0  # conservative default

    safe_ram_gb = available_ram_gb * 0.7  # leave 30% for OS + overhead

    if forcing_gb > safe_ram_gb:
        chunk_years = max(1, int(safe_ram_gb / (forcing_gb / years)))
        results.append(CheckResult(
            "warn",
            f"Forcing would use ~{forcing_gb:.0f} GB RAM (available: ~{available_ram_gb:.0f} GB) "
            f"\u2014 will auto-chunk by {chunk_years} year(s)",
            "Time & Data",
        ))
    elif forcing_gb > safe_ram_gb * 0.5:
        results.append(CheckResult(
            "pass",
            f"Forcing RAM estimate: ~{forcing_gb:.1f} GB (fits in {available_ram_gb:.0f} GB)",
            "Time & Data",
        ))

    return results


def _check_pressure_levels(cfg: TPS2Config) -> list[CheckResult]:
    """Check pressure level coverage and validity."""
    results = []
    levels = cfg.inputs.pressure_levels
    mode = cfg.downscaling.mode

    if mode == "simple":
        if levels:
            results.append(CheckResult(
                "pass",
                f"Simple mode \u2014 pressure levels ({levels}) will be ignored",
                "Time & Data",
            ))
        else:
            results.append(CheckResult("pass", "Simple mode \u2014 no pressure levels needed", "Time & Data"))
        return results

    # Full mode
    if not levels:
        results.append(CheckResult("fail", "Full mode requires pressure_levels to be set", "Time & Data"))
        return results

    # Check levels are in ERA5 standard set
    invalid = [l for l in levels if l not in ERA5_STANDARD_LEVELS]
    if invalid:
        results.append(CheckResult(
            "warn",
            f"Non-standard ERA5 pressure levels: {invalid}",
            "Time & Data",
        ))

    sorted_levels = sorted(levels)
    heights = [_pressure_to_height(l) for l in sorted_levels]
    min_h = min(heights)
    max_h = max(heights)

    results.append(CheckResult(
        "pass",
        f"Pressure levels: {sorted_levels} \u2014 brackets ~{min_h:.0f}m to ~{max_h:.0f}m",
        "Time & Data",
    ))

    # Check for large gaps between consecutive levels
    for i in range(len(sorted_levels) - 1):
        h1 = _pressure_to_height(sorted_levels[i])
        h2 = _pressure_to_height(sorted_levels[i + 1])
        gap = abs(h2 - h1)
        if gap > 2000:
            results.append(CheckResult(
                "warn",
                f"Pressure level gap: {sorted_levels[i]}\u2192{sorted_levels[i+1]} hPa "
                f"spans {gap:.0f}m \u2014 consider adding intermediate level",
                "Time & Data",
            ))

    # Check if levels bracket likely elevations for the bbox
    if cfg.domain.bbox is not None:
        # Highest level (lowest pressure) should reach above ~4000m for alpine
        # Lowest level (highest pressure) should reach below ~500m for valleys
        highest_level_height = max_h
        lowest_level_height = min_h
        if lowest_level_height > 500:
            results.append(CheckResult(
                "warn",
                f"Lowest pressure level ({max(sorted_levels)} hPa, ~{lowest_level_height:.0f}m) "
                f"may not cover low-elevation areas \u2014 consider adding 1000 hPa",
                "Time & Data",
            ))

    return results


def _check_backend_pressure_levels(cfg: TPS2Config) -> list[CheckResult]:
    """Check that requested pressure levels exist in the actual backend store.

    For s3zarr backend, probes the zarr store's level dimension to detect
    mismatches early (instead of logging a warning per-day during download).
    """
    logger = logging.getLogger(__name__)
    results = []
    levels = cfg.inputs.pressure_levels

    if not levels or cfg.inputs.backend != "s3zarr":
        return results

    try:
        import xarray as xr

        zarr_url = cfg.inputs.s3_zarr_url
        if not zarr_url:
            return results  # the backend itself reports the missing inputs.s3_zarr_url

        # Build minimal storage options (same logic as S3ZarrBackend.__init__)
        storage_options = {}
        endpoint_url = os.environ.get("AWS_ENDPOINT_URL")
        if endpoint_url:
            storage_options["client_kwargs"] = {"endpoint_url": endpoint_url}

        # Open just enough to read the level coordinate
        ds = xr.open_zarr(zarr_url, storage_options=storage_options, chunks=None)
        if "level" in ds:
            store_levels = sorted(int(v) for v in ds["level"].values)
            ds.close()

            missing = [l for l in levels if l not in store_levels]
            if missing:
                available_in_config = [l for l in levels if l in store_levels]
                results.append(CheckResult(
                    "fail",
                    f"Pressure levels {missing} not in S3 zarr store. "
                    f"Store has: {store_levels}. "
                    f"Usable from your config: {available_in_config}",
                    "Backend",
                ))
            else:
                results.append(CheckResult(
                    "pass",
                    f"All pressure levels {sorted(levels)} available in S3 zarr store",
                    "Backend",
                ))
        else:
            ds.close()

    except Exception as e:
        # Don't fail preflight if we can't probe — just warn
        logger.debug("Could not probe S3 zarr levels: %s", e)
        results.append(CheckResult(
            "warn",
            f"Could not verify pressure levels against S3 zarr store: {e}",
            "Backend",
        ))

    return results


def _check_mode_consistency(cfg: TPS2Config) -> list[CheckResult]:
    """Check mode/data consistency."""
    results = []
    mode = cfg.downscaling.mode
    levels = cfg.inputs.pressure_levels

    if mode == "full" and not levels:
        results.append(CheckResult(
            "fail",
            "Mode 'full' requires pressure_levels but none configured",
            "Time & Data",
        ))
    elif mode == "full" and levels:
        results.append(CheckResult("pass", f"Full mode with {len(levels)} pressure levels", "Time & Data"))
    elif mode == "simple":
        results.append(CheckResult(
            "pass",
            f"Simple mode with lapse_rate={cfg.downscaling.lapse_rate} K/m",
            "Time & Data",
        ))

    return results


def _check_forecast_pressure_levels(cfg: TPS2Config) -> list[CheckResult]:
    """Check forecast pressure level compatibility with reanalysis and backend."""
    results = []
    fc = cfg.forecast

    if not fc.enabled:
        return results

    fc_levels = set(fc.pressure_levels)
    ra_levels = set(cfg.inputs.pressure_levels)
    backend = fc.backend

    # 1. Check forecast levels against backend availability
    if backend == "ecmwf_opendata":
        available = ECMWF_OPENDATA_LEVELS
        backend_name = "ECMWF OpenData"
    elif backend == "openmeteo_ifs":
        available = OPENMETEO_IFS_LEVELS
        backend_name = "Open-Meteo IFS"
    else:
        # auto — check against both backends
        available = None
        backend_name = None

    if available is not None:
        unsupported = fc_levels - available
        if unsupported:
            results.append(CheckResult(
                "fail",
                f"Forecast levels {sorted(unsupported)} not available in {backend_name}. "
                f"Available: {sorted(available)}",
                "Forecast",
            ))
        else:
            results.append(CheckResult(
                "pass",
                f"Forecast levels {sorted(fc_levels)} all supported by {backend_name}",
                "Forecast",
            ))
    else:
        # Auto mode — check against both
        ecmwf_bad = fc_levels - ECMWF_OPENDATA_LEVELS
        openmeteo_bad = fc_levels - OPENMETEO_IFS_LEVELS

        if ecmwf_bad and openmeteo_bad:
            # Some levels unsupported by both backends
            both_bad = ecmwf_bad & openmeteo_bad
            if both_bad:
                results.append(CheckResult(
                    "fail",
                    f"Forecast levels {sorted(both_bad)} not available in any forecast backend",
                    "Forecast",
                ))
            else:
                results.append(CheckResult(
                    "warn",
                    f"Forecast levels {sorted(ecmwf_bad)} not in ECMWF OpenData; "
                    f"{sorted(openmeteo_bad)} not in Open-Meteo IFS. "
                    f"Auto-selection may be constrained",
                    "Forecast",
                ))
        elif ecmwf_bad:
            results.append(CheckResult(
                "warn",
                f"Forecast levels {sorted(ecmwf_bad)} only available via Open-Meteo IFS "
                f"(not ECMWF OpenData)",
                "Forecast",
            ))
        else:
            results.append(CheckResult(
                "pass",
                f"Forecast levels {sorted(fc_levels)} supported by both backends",
                "Forecast",
            ))

    # 2. Check reanalysis/forecast level compatibility for blending
    if fc.blending.enabled:
        if ra_levels == fc_levels:
            results.append(CheckResult(
                "pass",
                f"Reanalysis and forecast pressure levels match ({sorted(ra_levels)})",
                "Forecast",
            ))
        else:
            common = ra_levels & fc_levels
            ra_only = ra_levels - fc_levels
            fc_only = fc_levels - ra_levels

            if not common:
                results.append(CheckResult(
                    "fail",
                    f"No common pressure levels between reanalysis {sorted(ra_levels)} "
                    f"and forecast {sorted(fc_levels)} \u2014 blending will fail",
                    "Forecast",
                ))
            else:
                msg_parts = [
                    f"Reanalysis/forecast level mismatch: "
                    f"{len(common)} common {sorted(common)}"
                ]
                if ra_only:
                    msg_parts.append(f"reanalysis-only {sorted(ra_only)}")
                if fc_only:
                    msg_parts.append(f"forecast-only {sorted(fc_only)}")
                results.append(CheckResult(
                    "warn",
                    "; ".join(msg_parts),
                    "Forecast",
                ))

    # 3. Check reanalysis levels against forecast backend availability
    if fc.blending.enabled and ra_levels != fc_levels:
        # Suggest the intersection as a safe common set
        safe = ra_levels & FORECAST_COMMON_LEVELS
        if safe and safe != ra_levels:
            results.append(CheckResult(
                "warn",
                f"Consider aligning levels to {sorted(safe)} for seamless blending",
                "Forecast",
            ))

    return results


def _estimate_dem_pixels(cfg: TPS2Config) -> int:
    """Estimate number of DEM pixels from bbox and dem_source.

    If grid_resolution is set, the DEM is resampled to that resolution,
    so we estimate pixel count at the target resolution instead of native.
    """
    if cfg.domain.bbox is None:
        return 0
    west, south, east, north = cfg.domain.bbox

    if cfg.domain.grid_resolution is not None and cfg.domain.grid_resolution > 0:
        # grid_resolution is in meters — convert to degrees
        res_deg = cfg.domain.grid_resolution / 111_000
    else:
        res_deg = DEM_RESOLUTION_DEG.get(cfg.domain.dem_source, DEM_RESOLUTION_DEG["glo_90"])

    n_lat = max(1, int(math.ceil((north - south) / res_deg)))
    n_lon = max(1, int(math.ceil((east - west) / res_deg)))
    return n_lat * n_lon


def _check_clusters(cfg: TPS2Config) -> list[CheckResult]:
    """Check cluster count heuristic."""
    results = []

    if cfg.domain.spatial_mode != "clusters":
        results.append(CheckResult("pass", f"Spatial mode: {cfg.domain.spatial_mode} (not clusters)", "Clustering"))
        return results

    n_clusters = cfg.clustering.n_clusters
    n_pixels = _estimate_dem_pixels(cfg)

    if n_pixels > 0:
        ratio = n_pixels / n_clusters
        n_str = f"{n_pixels / 1e6:.0f}M" if n_pixels > 1e6 else f"{n_pixels:,}"

        if cfg.domain.grid_resolution is not None and cfg.domain.grid_resolution > 0:
            res_label = f"{cfg.domain.dem_source} resampled to {cfg.domain.grid_resolution:.0f}m"
        else:
            res_label = cfg.domain.dem_source

        results.append(CheckResult(
            "pass",
            f"DEM pixels: ~{n_str} ({res_label})",
            "Clustering",
        ))
        results.append(CheckResult(
            "pass",
            f"n_clusters={n_clusters:,}, ~{ratio:,.0f} pixels/cluster",
            "Clustering",
        ))

        if n_clusters > n_pixels / 10:
            results.append(CheckResult(
                "warn",
                f"n_clusters ({n_clusters:,}) is > 10% of DEM pixels ({n_pixels:,}) "
                f"\u2014 clusters may be poorly populated",
                "Clustering",
            ))
        if n_clusters < 5:
            results.append(CheckResult(
                "warn",
                f"n_clusters={n_clusters} is very low \u2014 spatial variability may be lost",
                "Clustering",
            ))
    else:
        results.append(CheckResult("pass", f"n_clusters={n_clusters:,}", "Clustering"))

    features = cfg.clustering.features
    results.append(CheckResult("pass", f"Features: {', '.join(features)}", "Clustering"))

    return results


def _check_backends(cfg: TPS2Config) -> list[CheckResult]:
    """Check backend availability."""
    results = []

    # CDS backend: check for .cdsapirc
    if cfg.inputs.backend in ("cds", "hybrid"):
        cdsrc = Path.home() / ".cdsapirc"
        if cdsrc.exists():
            results.append(CheckResult("pass", "CDS API key found (~/.cdsapirc)", "Backend"))
        else:
            status = "warn" if cfg.inputs.backend == "hybrid" else "fail"
            results.append(CheckResult(
                status,
                "CDS API key not found (~/.cdsapirc) \u2014 CDS backend will not work",
                "Backend",
            ))

    # Kernel backend check
    kernel = cfg.execution.kernel_backend
    if kernel == "rust":
        try:
            import topopyscale2.core._rust_kernels  # noqa: F401
            results.append(CheckResult("pass", "Rust kernel available", "Backend"))
        except ImportError:
            results.append(CheckResult(
                "warn",
                "Rust kernel not available \u2014 runs will use the Python kernels "
                "(same results, slower)",
                "Backend",
            ))
    elif kernel == "jax":
        try:
            import jax  # noqa: F401
            results.append(CheckResult("pass", "JAX kernel available", "Backend"))
        except ImportError:
            results.append(CheckResult(
                "fail",
                "JAX not installed \u2014 run `pip install jax jaxlib`",
                "Backend",
            ))
    else:
        results.append(CheckResult("pass", "Python kernel available", "Backend"))

    # Check if other kernels are available (informational)
    if kernel != "rust":
        try:
            import topopyscale2.core._rust_kernels  # noqa: F401
            results.append(CheckResult("pass", "Rust kernel also available (10x speedup)", "Backend"))
        except ImportError:
            results.append(CheckResult(
                "warn",
                "Rust kernel not compiled \u2014 consider `pip install -e .` for 10x speedup",
                "Backend",
            ))

    return results


def _check_disk_space(cfg: TPS2Config, estimates: list[StageEstimate]) -> list[CheckResult]:
    """Check disk space on output and cache partitions."""
    results = []

    def _check_path_space(dir_path: Path, label: str, needed_gb: float):
        check_path = dir_path
        while not check_path.exists():
            check_path = check_path.parent
            if check_path == check_path.parent:
                break
        try:
            usage = shutil.disk_usage(check_path)
            free_gb = usage.free / 1e9
            results.append(CheckResult(
                "pass" if free_gb > needed_gb else "warn",
                f"{label} disk free: {free_gb:.1f} GB on {check_path} "
                f"(need ~{needed_gb:.1f} GB)",
                "Disk",
            ))
        except OSError as e:
            results.append(CheckResult("warn", f"Cannot check {label} disk space: {e}", "Disk"))

    # ERA5 download estimate (find by stage name)
    era5_gb = sum(e.data_volume_gb for e in estimates if "ERA5" in e.stage or "download" in e.stage.lower())
    total_output_gb = sum(e.data_volume_gb for e in estimates) - era5_gb

    # Check cache directory (where ERA5 data is stored)
    cache_dir = Path(cfg.inputs.cache_dir)
    _check_path_space(cache_dir, "Cache", era5_gb * 2)

    # Check output directory
    output_dir = Path(cfg.output.directory)
    _check_path_space(output_dir, "Output", total_output_gb * 2)

    return results


def _check_output_dir(cfg: TPS2Config) -> list[CheckResult]:
    """Check output directory writability."""
    results = []
    output_dir = Path(cfg.output.directory)

    if output_dir.exists():
        if os.access(output_dir, os.W_OK):
            results.append(CheckResult("pass", f"Output directory writable: {output_dir}", "Output"))
        else:
            results.append(CheckResult("fail", f"Output directory not writable: {output_dir}", "Output"))
    else:
        # Check if parent is writable
        parent = output_dir.parent
        while not parent.exists():
            parent = parent.parent
            if parent == parent.parent:
                break
        if os.access(parent, os.W_OK):
            results.append(CheckResult("pass", f"Output directory will be created: {output_dir}", "Output"))
        else:
            results.append(CheckResult("fail", f"Cannot create output directory: {output_dir}", "Output"))

    return results


def _compute_era5_grid(cfg: TPS2Config) -> tuple[int, int, int]:
    """Compute ERA5 grid dimensions and return (n_lat, n_lon, n_grid)."""
    if cfg.domain.bbox is None:
        return 0, 0, 0
    west, south, east, north = cfg.domain.bbox
    n_lat = max(1, math.ceil((north - south) / 0.25))
    n_lon = max(1, math.ceil((east - west) / 0.25))
    return n_lat, n_lon, n_lat * n_lon


def _estimate_stages(cfg: TPS2Config) -> list[StageEstimate]:
    """Compute runtime and data volume estimates for each stage."""
    estimates = []
    n_pixels = _estimate_dem_pixels(cfg)
    n_lat, n_lon, n_grid = _compute_era5_grid(cfg)

    # Time dimensions
    tr = cfg.inputs.time_range
    if tr and len(tr) == 2:
        try:
            start = datetime.strptime(tr[0], "%Y-%m-%d")
            end = datetime.strptime(tr[1], "%Y-%m-%d")
            n_days = (end - start).days + 1
        except ValueError:
            n_days = 1
    else:
        n_days = 1

    hpd = HOURS_PER_DAY.get(cfg.inputs.time_resolution, 24)
    n_timesteps = n_days * hpd
    n_levels = len(cfg.inputs.pressure_levels)

    # --- DEM setup ---
    dem_gb = n_pixels * 4 / 1e9  # 4 bytes per pixel, elevation only
    # Processing: DEM + slope + aspect + SVF
    dem_gb *= 5  # multiple layers
    if n_pixels < 1_000_000:
        dem_time_low, dem_time_high = 0.5, 2.0
    elif n_pixels < 10_000_000:
        dem_time_low, dem_time_high = 2.0, 8.0
    elif n_pixels < 50_000_000:
        dem_time_low, dem_time_high = 5.0, 15.0
    else:
        dem_time_low, dem_time_high = 10.0, 30.0

    # SVF full mode adds significant time
    if cfg.domain.svf_mode == "full" and n_pixels > 1_000_000:
        dem_time_low *= 2
        dem_time_high *= 3

    estimates.append(StageEstimate("DEM setup", dem_gb, dem_time_low, dem_time_high))

    # --- ERA5 download ---
    # Surface: 10 vars * 4 bytes * n_grid * n_timesteps
    # Pressure: 5 vars * n_levels * 4 bytes * n_grid * n_timesteps
    bytes_raw = n_timesteps * n_grid * 4 * (10 + 5 * n_levels)
    era5_gb = bytes_raw / 1e9 / 2.5  # ~2.5x compression

    backend = cfg.inputs.backend
    if backend == "google":
        era5_time_low = n_days * 0.3
        era5_time_high = n_days * 0.7
    elif backend == "s3zarr":
        era5_time_low = n_days * 0.2
        era5_time_high = n_days * 0.5
    elif backend == "cds":
        era5_time_low = n_days * 1.5
        era5_time_high = n_days * 3.0
    elif backend == "hybrid":
        era5_time_low = n_days * 0.3
        era5_time_high = n_days * 1.0
    else:
        era5_time_low = n_days * 0.5
        era5_time_high = n_days * 1.5

    estimates.append(StageEstimate("ERA5 download", era5_gb, era5_time_low, era5_time_high))

    # --- Downscaling ---
    n_clusters = cfg.clustering.n_clusters
    kernel = cfg.execution.kernel_backend
    backend_factor = {"python": 0.05, "rust": 0.005, "jax": 0.02}.get(kernel, 0.05)
    ds_seconds = n_clusters * n_timesteps * backend_factor / 1000
    n_workers = cfg.execution.n_workers or (os.cpu_count() or 4)
    ds_seconds /= max(1, n_workers)
    ds_minutes = ds_seconds / 60

    # Output volume: all output variables * n_clusters * n_timesteps * 4 bytes
    n_out_vars = len(cfg.output.variables) or 11  # default 11 variables
    out_gb = n_clusters * n_timesteps * n_out_vars * 4 / 1e9

    estimates.append(StageEstimate(
        "Downscaling",
        out_gb,
        ds_minutes * 0.8,
        ds_minutes * 1.5,
    ))

    # --- Impact model (if configured) ---
    if cfg.application.model != "none":
        model = cfg.application.model
        if model == "fsm":
            model_seconds = n_clusters * n_timesteps * 0.001
        elif model == "fsm2":
            model_seconds = n_clusters * n_timesteps * 0.0001
        else:
            model_seconds = n_clusters * n_timesteps * 0.0005
        model_minutes = model_seconds / 60
        estimates.append(StageEstimate(
            f"Impact model ({model})",
            out_gb * 0.3,  # model output typically smaller
            model_minutes * 0.8,
            model_minutes * 1.5,
        ))

    return estimates


def run_preflight(
    cfg: TPS2Config,
    skip_disk: bool = False,
) -> PreflightReport:
    """Run all preflight checks on a TPS2 configuration.

    Parameters
    ----------
    cfg : TPS2Config
        Parsed configuration.
    skip_disk : bool
        Skip disk space check (useful for network storage).

    Returns
    -------
    PreflightReport
        Report with all check results and stage estimates.
    """
    report = PreflightReport()

    # Domain checks
    report.checks.extend(_check_bbox(cfg))

    # ERA5 grid info
    n_lat, n_lon, n_grid = _compute_era5_grid(cfg)
    if n_grid > 0:
        report.checks.append(CheckResult(
            "pass",
            f"ERA5 grid: {n_lat} \u00d7 {n_lon} = {n_grid:,} cells",
            "Domain",
        ))

    # Time range checks
    report.checks.extend(_check_time_range(cfg))

    # Pressure level checks
    report.checks.extend(_check_pressure_levels(cfg))

    # Backend-specific pressure level checks (probes actual store)
    report.checks.extend(_check_backend_pressure_levels(cfg))

    # Mode consistency
    report.checks.extend(_check_mode_consistency(cfg))

    # Forecast pressure level compatibility
    report.checks.extend(_check_forecast_pressure_levels(cfg))

    # Cluster checks
    report.checks.extend(_check_clusters(cfg))

    # Backend checks
    report.checks.extend(_check_backends(cfg))

    # Output directory
    report.checks.extend(_check_output_dir(cfg))

    # Estimates
    report.estimates = _estimate_stages(cfg)

    # Disk space (after estimates so we know volumes)
    if not skip_disk:
        report.checks.extend(_check_disk_space(cfg, report.estimates))

    return report


def format_report(report: PreflightReport, config_name: str = "config.yaml") -> str:
    """Format a PreflightReport as a plain-text string.

    Parameters
    ----------
    report : PreflightReport
        The preflight report to format.
    config_name : str
        Config filename for the header.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = [f"Preflight Checks: {config_name}", ""]

    status_icons = {"pass": "\u2713", "warn": "\u26a0", "fail": "\u2717"}

    # Group checks by category
    categories: dict[str, list[CheckResult]] = {}
    for check in report.checks:
        cat = check.category or "General"
        categories.setdefault(cat, []).append(check)

    for cat, checks in categories.items():
        lines.append(f"  {cat}")
        for c in checks:
            icon = status_icons[c.status]
            lines.append(f"  {icon} {c.message}")
        lines.append("")

    # Estimates table
    if report.estimates:
        lines.append("  Estimates")

        # Header
        lines.append(f"  {'Stage':<20} {'Data Volume':<14} {'Est. Time'}")
        lines.append(f"  {'-'*20} {'-'*14} {'-'*14}")

        for est in report.estimates:
            lines.append(f"  {est.stage:<20} {est.volume_str:<14} {est.time_str}")

        # Totals
        total_vol = report.total_volume_gb
        total_low = report.total_time_low
        total_high = report.total_time_high
        if total_low < 1 and total_high < 1:
            total_time_str = f"~{total_high * 60:.0f}s"
        else:
            total_time_str = f"~{total_low:.0f}-{total_high:.0f} min"
        total_vol_str = f"~{total_vol:.1f} GB"
        lines.append(f"  {'-'*20} {'-'*14} {'-'*14}")
        lines.append(f"  {'Total':<20} {total_vol_str:<14} {total_time_str}")
        lines.append("")

    # Summary
    lines.append(f"  {report.n_pass} passed, {report.n_warn} warning(s), {report.n_fail} error(s)")

    return "\n".join(lines)
