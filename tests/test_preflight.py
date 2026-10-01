"""Tests for preflight checks module."""

import pytest

from topopyscale2.config.schema import TPS2Config
from topopyscale2.preflight import (
    CheckResult,
    PreflightReport,
    StageEstimate,
    _check_bbox,
    _check_clusters,
    _check_forecast_pressure_levels,
    _check_mode_consistency,
    _check_pressure_levels,
    _check_time_range,
    _estimate_dem_pixels,
    _estimate_stages,
    _pressure_to_height,
    format_report,
    run_preflight,
)


def _make_config(**overrides) -> TPS2Config:
    """Build a minimal valid TPS2Config with overrides."""
    base = {
        "domain": {"bbox": [7.5, 46.0, 7.7, 46.2], "dem_source": "glo_90"},
        "inputs": {
            "time_range": ["2023-06-01", "2023-06-03"],
            "pressure_levels": [500, 700, 850, 1000],
        },
        "clustering": {"n_clusters": 50},
        "downscaling": {"mode": "full"},
        "execution": {"kernel_backend": "python"},
    }
    # Deep merge overrides
    for key, val in overrides.items():
        if isinstance(val, dict) and key in base and isinstance(base[key], dict):
            base[key].update(val)
        else:
            base[key] = val
    return TPS2Config(**base)


# =============================================================================
# Unit tests for individual check functions
# =============================================================================


class TestPressureToHeight:
    def test_known_levels(self):
        assert _pressure_to_height(1000) == 100
        assert _pressure_to_height(500) == 5500
        assert _pressure_to_height(300) == 9200

    def test_interpolation(self):
        h = _pressure_to_height(600)
        # 600 hPa should be between 500 hPa (5500m) and 700 hPa (3000m)
        assert 3000 < h < 5500


class TestCheckBbox:
    def test_valid_bbox(self):
        cfg = _make_config()
        results = _check_bbox(cfg)
        assert any(r.status == "pass" and "Bbox valid" in r.message for r in results)

    def test_inverted_south_north(self):
        # Pydantic catches this before preflight — verify it raises
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="south must be less than north"):
            _make_config(domain={"bbox": [7.5, 46.2, 7.7, 46.0], "dem_source": "glo_90"})

    def test_inverted_west_east(self):
        # Pydantic catches this before preflight — verify it raises
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="west must be less than east"):
            _make_config(domain={"bbox": [7.7, 46.0, 7.5, 46.2], "dem_source": "glo_90"})

    def test_large_domain_warning(self):
        cfg = _make_config(domain={"bbox": [0.0, 0.0, 40.0, 35.0], "dem_source": "glo_90"})
        results = _check_bbox(cfg)
        assert any(r.status == "warn" and "Very large" in r.message for r in results)

    def test_tiny_domain_warning(self):
        cfg = _make_config(domain={"bbox": [7.5, 46.0, 7.505, 46.005], "dem_source": "glo_90"})
        results = _check_bbox(cfg)
        assert any(r.status == "warn" and "Very small" in r.message for r in results)

    def test_dem_path_no_bbox(self):
        cfg = _make_config(domain={"dem": "/tmp/test.tif", "bbox": None, "dem_source": "glo_90"})
        results = _check_bbox(cfg)
        assert any(r.status == "pass" and "user-provided DEM" in r.message for r in results)

    def test_longitude_out_of_range(self):
        cfg = _make_config(domain={"bbox": [-200.0, 46.0, 7.7, 46.2], "dem_source": "glo_90"})
        results = _check_bbox(cfg)
        assert any(r.status == "fail" and "out of range" in r.message for r in results)


class TestCheckTimeRange:
    def test_valid_range(self):
        cfg = _make_config()
        results = _check_time_range(cfg)
        assert any(r.status == "pass" and "3 days" in r.message for r in results)

    def test_start_after_end(self):
        cfg = _make_config(inputs={
            "time_range": ["2023-06-03", "2023-06-01"],
            "pressure_levels": [500, 700, 850, 1000],
        })
        results = _check_time_range(cfg)
        assert any(r.status == "fail" for r in results)

    def test_long_range_warning(self):
        cfg = _make_config(inputs={
            "time_range": ["2018-01-01", "2023-12-31"],
            "pressure_levels": [500, 700, 850, 1000],
        })
        results = _check_time_range(cfg)
        assert any(r.status == "warn" and "Long time range" in r.message for r in results)

    def test_future_end_warning(self):
        cfg = _make_config(inputs={
            "time_range": ["2023-01-01", "2099-01-01"],
            "pressure_levels": [500, 700, 850, 1000],
        })
        results = _check_time_range(cfg)
        assert any(r.status == "warn" and "future" in r.message for r in results)

    def test_empty_time_range(self):
        cfg = _make_config(inputs={
            "time_range": [],
            "pressure_levels": [500, 700, 850, 1000],
        })
        results = _check_time_range(cfg)
        assert any(r.status == "fail" for r in results)

    def test_bad_date_format(self):
        cfg = _make_config(inputs={
            "time_range": ["not-a-date", "2023-06-03"],
            "pressure_levels": [500, 700, 850, 1000],
        })
        results = _check_time_range(cfg)
        assert any(r.status == "fail" and "parse" in r.message for r in results)


class TestCheckPressureLevels:
    def test_valid_levels_full_mode(self):
        cfg = _make_config()
        results = _check_pressure_levels(cfg)
        assert any(r.status == "pass" and "Pressure levels" in r.message for r in results)

    def test_simple_mode_no_levels(self):
        cfg = _make_config(
            downscaling={"mode": "simple"},
            inputs={"time_range": ["2023-06-01", "2023-06-03"], "pressure_levels": []},
        )
        results = _check_pressure_levels(cfg)
        assert any(r.status == "pass" and "Simple mode" in r.message for r in results)
        assert not any(r.status == "fail" for r in results)

    def test_full_mode_no_levels(self):
        cfg = _make_config(inputs={
            "time_range": ["2023-06-01", "2023-06-03"],
            "pressure_levels": [],
        })
        results = _check_pressure_levels(cfg)
        assert any(r.status == "fail" and "requires pressure_levels" in r.message for r in results)

    def test_large_gap_warning(self):
        cfg = _make_config(inputs={
            "time_range": ["2023-06-01", "2023-06-03"],
            "pressure_levels": [300, 1000],
        })
        results = _check_pressure_levels(cfg)
        assert any(r.status == "warn" and "gap" in r.message for r in results)

    def test_missing_low_elevation_coverage(self):
        # Only high-altitude levels
        cfg = _make_config(inputs={
            "time_range": ["2023-06-01", "2023-06-03"],
            "pressure_levels": [300, 500],
        })
        results = _check_pressure_levels(cfg)
        assert any(r.status == "warn" and "low-elevation" in r.message for r in results)


class TestCheckModeConsistency:
    def test_full_with_levels(self):
        cfg = _make_config()
        results = _check_mode_consistency(cfg)
        assert any(r.status == "pass" and "Full mode" in r.message for r in results)

    def test_full_without_levels(self):
        cfg = _make_config(inputs={
            "time_range": ["2023-06-01", "2023-06-03"],
            "pressure_levels": [],
        })
        results = _check_mode_consistency(cfg)
        assert any(r.status == "fail" for r in results)

    def test_simple_mode(self):
        cfg = _make_config(downscaling={"mode": "simple"})
        results = _check_mode_consistency(cfg)
        assert any(r.status == "pass" and "Simple mode" in r.message for r in results)


class TestCheckClusters:
    def test_reasonable_clusters(self):
        cfg = _make_config()
        results = _check_clusters(cfg)
        passes = [r for r in results if r.status == "pass"]
        assert len(passes) >= 2  # pixels + n_clusters + features

    def test_too_many_clusters(self):
        # Very small bbox with lots of clusters
        cfg = _make_config(
            domain={"bbox": [7.5, 46.0, 7.51, 46.01], "dem_source": "glo_90"},
            clustering={"n_clusters": 50000},
        )
        results = _check_clusters(cfg)
        assert any(r.status == "warn" and "poorly populated" in r.message for r in results)

    def test_very_few_clusters(self):
        cfg = _make_config(clustering={"n_clusters": 3})
        results = _check_clusters(cfg)
        assert any(r.status == "warn" and "very low" in r.message for r in results)


class TestEstimateDemPixels:
    def test_glo90(self):
        cfg = _make_config()
        n = _estimate_dem_pixels(cfg)
        # 0.2° × 0.2° at 90m resolution → roughly 247 × 247 ≈ 61K pixels
        assert 40_000 < n < 100_000

    def test_glo30(self):
        cfg = _make_config(domain={"bbox": [7.5, 46.0, 7.7, 46.2], "dem_source": "glo_30"})
        n_30 = _estimate_dem_pixels(cfg)
        cfg_90 = _make_config()
        n_90 = _estimate_dem_pixels(cfg_90)
        # 30m should have ~9x more pixels than 90m
        assert n_30 > n_90 * 5

    def test_grid_resolution_reduces_pixels(self):
        cfg_native = _make_config()
        cfg_500m = _make_config(domain={"bbox": [7.5, 46.0, 7.7, 46.2], "dem_source": "glo_90", "grid_resolution": 500})
        n_native = _estimate_dem_pixels(cfg_native)
        n_500m = _estimate_dem_pixels(cfg_500m)
        # 500m should have far fewer pixels than 90m native
        assert n_500m < n_native / 10

    def test_no_bbox(self):
        cfg = _make_config(domain={"dem": "/tmp/test.tif", "bbox": None, "dem_source": "glo_90"})
        assert _estimate_dem_pixels(cfg) == 0


class TestEstimateStages:
    def test_returns_stages(self):
        cfg = _make_config()
        estimates = _estimate_stages(cfg)
        stage_names = [e.stage for e in estimates]
        assert "DEM setup" in stage_names
        assert "ERA5 download" in stage_names
        assert "Downscaling" in stage_names

    def test_with_impact_model(self):
        cfg = _make_config(application={"model": "fsm2"})
        estimates = _estimate_stages(cfg)
        stage_names = [e.stage for e in estimates]
        assert any("fsm2" in s for s in stage_names)

    def test_no_impact_model(self):
        cfg = _make_config()
        estimates = _estimate_stages(cfg)
        stage_names = [e.stage for e in estimates]
        assert not any("Impact" in s for s in stage_names)


# =============================================================================
# Integration tests for run_preflight
# =============================================================================


class TestRunPreflight:
    def test_valid_config_all_pass(self):
        cfg = _make_config()
        report = run_preflight(cfg, skip_disk=True)
        assert report.n_fail == 0
        assert report.n_pass > 0
        assert report.ok

    def test_missing_pressure_levels_full_mode(self):
        cfg = _make_config(inputs={
            "time_range": ["2023-06-01", "2023-06-03"],
            "pressure_levels": [],
        })
        report = run_preflight(cfg, skip_disk=True)
        assert report.n_fail > 0
        assert not report.ok

    def test_huge_time_range_warns(self):
        cfg = _make_config(inputs={
            "time_range": ["2015-01-01", "2023-12-31"],
            "pressure_levels": [500, 700, 850, 1000],
        })
        report = run_preflight(cfg, skip_disk=True)
        assert report.n_warn > 0

    def test_absurd_cluster_count_warns(self):
        cfg = _make_config(
            domain={"bbox": [7.5, 46.0, 7.51, 46.01], "dem_source": "glo_90"},
            clustering={"n_clusters": 100000},
        )
        report = run_preflight(cfg, skip_disk=True)
        assert any(
            c.status == "warn" and "poorly populated" in c.message
            for c in report.checks
        )

    def test_estimates_present(self):
        cfg = _make_config()
        report = run_preflight(cfg, skip_disk=True)
        assert len(report.estimates) >= 3
        assert report.total_volume_gb > 0

    def test_skip_disk(self):
        cfg = _make_config()
        report = run_preflight(cfg, skip_disk=True)
        assert not any(c.category == "Disk" for c in report.checks)

    def test_with_disk_check(self):
        cfg = _make_config()
        report = run_preflight(cfg, skip_disk=False)
        assert any(c.category == "Disk" for c in report.checks)

    def test_simple_mode_valid(self):
        cfg = _make_config(
            downscaling={"mode": "simple"},
            inputs={
                "time_range": ["2023-06-01", "2023-06-03"],
                "pressure_levels": [],
            },
        )
        report = run_preflight(cfg, skip_disk=True)
        assert report.n_fail == 0


# =============================================================================
# Report formatting
# =============================================================================


class TestFormatReport:
    def test_format_contains_summary(self):
        cfg = _make_config()
        report = run_preflight(cfg, skip_disk=True)
        text = format_report(report, "test.yaml")
        assert "Preflight Checks: test.yaml" in text
        assert "passed" in text
        assert "warning" in text
        assert "error" in text

    def test_format_contains_estimates(self):
        cfg = _make_config()
        report = run_preflight(cfg, skip_disk=True)
        text = format_report(report)
        assert "Estimates" in text
        assert "DEM setup" in text
        assert "Total" in text


# =============================================================================
# Data classes
# =============================================================================


class TestPreflightReport:
    def test_counts(self):
        report = PreflightReport(checks=[
            CheckResult("pass", "ok"),
            CheckResult("pass", "ok2"),
            CheckResult("warn", "hmm"),
            CheckResult("fail", "bad"),
        ])
        assert report.n_pass == 2
        assert report.n_warn == 1
        assert report.n_fail == 1
        assert not report.ok

    def test_empty_report(self):
        report = PreflightReport()
        assert report.n_pass == 0
        assert report.n_warn == 0
        assert report.n_fail == 0
        assert report.ok


class TestStageEstimate:
    def test_time_str_seconds(self):
        est = StageEstimate("test", 0.1, 0.3, 0.5)
        assert "s" in est.time_str

    def test_time_str_minutes(self):
        est = StageEstimate("test", 1.0, 5.0, 10.0)
        assert "min" in est.time_str

    def test_volume_str_mb(self):
        est = StageEstimate("test", 0.005, 1.0, 2.0)
        assert "MB" in est.volume_str

    def test_volume_str_gb(self):
        est = StageEstimate("test", 2.5, 1.0, 2.0)
        assert "GB" in est.volume_str


# =============================================================================
# Forecast pressure level compatibility checks
# =============================================================================


class TestCheckForecastPressureLevels:
    """Tests for _check_forecast_pressure_levels."""

    def test_forecast_disabled_returns_empty(self):
        cfg = _make_config()
        results = _check_forecast_pressure_levels(cfg)
        assert results == []

    def test_matching_levels_pass(self):
        """Matching reanalysis and forecast levels with blending enabled."""
        cfg = _make_config(
            inputs={"pressure_levels": [1000, 850, 700, 500, 300],
                    "time_range": ["2023-06-01", "2023-06-03"]},
            forecast={"enabled": True,
                      "pressure_levels": [1000, 850, 700, 500, 300],
                      "blending": {"enabled": True}},
        )
        results = _check_forecast_pressure_levels(cfg)
        statuses = [r.status for r in results]
        assert "fail" not in statuses
        # Should have a pass for backend compatibility + a pass for level match
        assert statuses.count("pass") >= 2

    def test_ecmwf_unsupported_level_fails(self):
        """Level not in ECMWF OpenData set should fail."""
        cfg = _make_config(
            forecast={"enabled": True,
                      "backend": "ecmwf_opendata",
                      "pressure_levels": [1000, 850, 750]},
        )
        results = _check_forecast_pressure_levels(cfg)
        fails = [r for r in results if r.status == "fail"]
        assert len(fails) == 1
        assert "750" in fails[0].message

    def test_openmeteo_unsupported_level_fails(self):
        """Level not in Open-Meteo IFS set should fail."""
        # Open-Meteo doesn't support arbitrary levels; use one that's invalid
        cfg = _make_config(
            forecast={"enabled": True,
                      "backend": "openmeteo_ifs",
                      "pressure_levels": [1000, 850, 775]},
        )
        results = _check_forecast_pressure_levels(cfg)
        fails = [r for r in results if r.status == "fail"]
        assert len(fails) == 1
        assert "775" in fails[0].message

    def test_auto_backend_both_supported_passes(self):
        """Levels supported by both backends in auto mode should pass."""
        cfg = _make_config(
            forecast={"enabled": True,
                      "backend": "auto",
                      "pressure_levels": [1000, 850, 700, 500, 300]},
        )
        results = _check_forecast_pressure_levels(cfg)
        statuses = [r.status for r in results]
        assert "fail" not in statuses

    def test_auto_backend_ecmwf_only_warns(self):
        """Level only in Open-Meteo (not ECMWF OpenData) should warn in auto mode."""
        # 750 is in Open-Meteo but NOT in ECMWF OpenData
        cfg = _make_config(
            forecast={"enabled": True,
                      "backend": "auto",
                      "pressure_levels": [1000, 850, 750]},
        )
        results = _check_forecast_pressure_levels(cfg)
        warns = [r for r in results if r.status == "warn"]
        assert len(warns) >= 1
        assert any("750" in w.message for w in warns)

    def test_auto_backend_neither_supported_fails(self):
        """Level not in any forecast backend should fail in auto mode."""
        # 775 is not in ECMWF OpenData or Open-Meteo IFS
        cfg = _make_config(
            forecast={"enabled": True,
                      "backend": "auto",
                      "pressure_levels": [1000, 850, 775]},
        )
        results = _check_forecast_pressure_levels(cfg)
        fails = [r for r in results if r.status == "fail"]
        assert len(fails) == 1
        assert "775" in fails[0].message

    def test_blending_no_common_levels_fails(self):
        """No overlap between reanalysis and forecast levels should fail."""
        cfg = _make_config(
            inputs={"pressure_levels": [500, 300],
                    "time_range": ["2023-06-01", "2023-06-03"]},
            forecast={"enabled": True,
                      "pressure_levels": [1000, 850],
                      "blending": {"enabled": True}},
        )
        results = _check_forecast_pressure_levels(cfg)
        fails = [r for r in results if r.status == "fail"]
        assert any("No common pressure levels" in f.message for f in fails)

    def test_blending_partial_overlap_warns(self):
        """Partial overlap should warn with details."""
        cfg = _make_config(
            inputs={"pressure_levels": [1000, 850, 700, 600],
                    "time_range": ["2023-06-01", "2023-06-03"]},
            forecast={"enabled": True,
                      "pressure_levels": [1000, 850, 500, 300],
                      "blending": {"enabled": True}},
        )
        results = _check_forecast_pressure_levels(cfg)
        warns = [r for r in results if r.status == "warn"]
        assert any("mismatch" in w.message.lower() for w in warns)

    def test_blending_disabled_skips_level_match(self):
        """With blending disabled, level mismatch is not checked."""
        cfg = _make_config(
            inputs={"pressure_levels": [500, 300],
                    "time_range": ["2023-06-01", "2023-06-03"]},
            forecast={"enabled": True,
                      "pressure_levels": [1000, 850, 700, 500, 300],
                      "blending": {"enabled": False}},
        )
        results = _check_forecast_pressure_levels(cfg)
        # No fail about level mismatch
        fails = [r for r in results if r.status == "fail"]
        assert not any("common" in f.message.lower() for f in fails)

    def test_run_preflight_includes_forecast_checks(self):
        """run_preflight should include forecast checks when enabled."""
        cfg = _make_config(
            inputs={"pressure_levels": [1000, 850, 700, 500, 300],
                    "time_range": ["2023-06-01", "2023-06-03"]},
            forecast={"enabled": True,
                      "pressure_levels": [1000, 850, 700, 500, 300],
                      "blending": {"enabled": True}},
        )
        report = run_preflight(cfg, skip_disk=True)
        forecast_checks = [c for c in report.checks if c.category == "Forecast"]
        assert len(forecast_checks) >= 2
