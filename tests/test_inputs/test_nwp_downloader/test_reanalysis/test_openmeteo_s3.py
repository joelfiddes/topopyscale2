"""Tests for OpenMeteoS3Backend."""

from __future__ import annotations

import tempfile
from unittest.mock import patch

import pandas as pd
import pytest

from topopyscale2.inputs.nwp_downloader.bbox import BBox


class TestOpenMeteoS3BackendInit:
    """Test OpenMeteoS3Backend initialization."""

    def test_init_with_valid_params(self, sample_bbox: BBox):
        """Backend should initialize with valid parameters."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                pressure_levels=[850, 500],
                time_resolution="3H",
                cache_dir=tmpdir,
            )
            assert backend.bbox == sample_bbox
            assert backend.time_resolution == "3H"
            assert backend.dataset == "era5"

    def test_init_warns_about_pressure_levels(self, sample_bbox: BBox, caplog):
        """Backend should warn that pressure levels are not available."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            OpenMeteoS3Backend(
                bbox=sample_bbox,
                pressure_levels=[850, 500],
                cache_dir=tmpdir,
            )
            assert "no pressure level data" in caplog.text.lower()

    def test_init_valid_time_resolutions(self, sample_bbox: BBox):
        """Backend should accept valid time resolutions."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for res in ["1H", "2H", "3H", "6H"]:
                backend = OpenMeteoS3Backend(
                    bbox=sample_bbox,
                    time_resolution=res,
                    cache_dir=tmpdir,
                )
                assert backend.time_resolution == res

    def test_init_invalid_time_resolution_raises(self, sample_bbox: BBox):
        """Backend should reject invalid time resolution."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid time_resolution"):
                OpenMeteoS3Backend(
                    bbox=sample_bbox,
                    time_resolution="4H",
                    cache_dir=tmpdir,
                )

    def test_init_dataset_options(self, sample_bbox: BBox):
        """Backend should accept era5 or era5_land dataset."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            for ds in ["era5", "era5_land"]:
                backend = OpenMeteoS3Backend(
                    bbox=sample_bbox,
                    dataset=ds,
                    cache_dir=tmpdir,
                )
                assert backend.dataset == ds


class TestOpenMeteoS3BackendGrid:
    """Test grid indexing for S3 backend."""

    def test_bbox_to_indices_swiss_alps(self, sample_bbox: BBox):
        """Grid indices for Swiss Alps bbox."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            # Check indices are within valid range
            assert 0 <= backend._i_lat_start < 721
            assert 0 < backend._i_lat_end <= 721
            assert 0 <= backend._i_lon_start < 1440
            assert 0 < backend._i_lon_end <= 1440
            # Check lat/lon arrays are extracted
            assert len(backend._lats) > 0
            assert len(backend._lons) > 0

    def test_bbox_to_indices_global(self):
        """Grid indices for global bbox."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        bbox = BBox.from_tuple((-180.0, -90.0, 179.75, 90.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=bbox,
                cache_dir=tmpdir,
            )
            # Should cover full grid
            assert backend._i_lat_start == 0
            assert backend._i_lat_end == 721
            assert backend._i_lon_start == 0
            assert backend._i_lon_end == 1440


class TestOpenMeteoS3BackendChunks:
    """Test chunk ID calculation."""

    def test_chunk_id_reference_point(self, sample_bbox: BBox):
        """Reference date should return reference chunk ID."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            # Reference: chunk 975 starts at 2026-01-11
            chunk_id = backend._date_to_chunk_id(pd.Timestamp("2026-01-11"))
            assert chunk_id == 975

    def test_chunk_id_earlier_date(self, sample_bbox: BBox):
        """Earlier dates should return lower chunk IDs."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            # 21 days before reference
            chunk_id = backend._date_to_chunk_id(pd.Timestamp("2025-12-21"))
            assert chunk_id == 974

    def test_chunk_id_later_date(self, sample_bbox: BBox):
        """Later dates should return higher chunk IDs."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            # 21 days after reference
            chunk_id = backend._date_to_chunk_id(pd.Timestamp("2026-02-01"))
            assert chunk_id == 976

    def test_chunk_time_range(self, sample_bbox: BBox):
        """Chunk time range should span 21 days."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            start, end = backend._chunk_time_range(975)
            duration = end - start
            assert duration == pd.Timedelta(hours=504)  # 21 days


class TestOpenMeteoS3BackendFilePaths:
    """Test file path generation."""

    def test_recent_date_uses_chunk(self, sample_bbox: BBox):
        """Post-2022 dates should use chunk files."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            path, _ = backend._get_file_path("temperature_2m", pd.Timestamp("2025-01-15"))
            assert "chunk_" in path
            assert "year_" not in path

    def test_historical_date_uses_year_file(self, sample_bbox: BBox):
        """Pre-2022 dates should use year files."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            OpenMeteoS3Backend,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = OpenMeteoS3Backend(
                bbox=sample_bbox,
                dataset="era5_land",
                cache_dir=tmpdir,
            )
            path, _ = backend._get_file_path("temperature_2m", pd.Timestamp("2000-07-15"))
            assert "year_2000" in path
            assert "chunk_" not in path


class TestOpenMeteoS3BackendVariables:
    """Test variable mappings."""

    def test_surface_vars_defined(self):
        """Surface variables should be defined with conversions."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            S3_SURFACE_VARS,
        )

        expected = {"temperature_2m", "dew_point_2m", "pressure_msl",
                    "shortwave_radiation", "precipitation"}
        assert set(S3_SURFACE_VARS.keys()) == expected

    def test_temperature_conversion(self):
        """Temperature should convert °C to K."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            S3_SURFACE_VARS,
        )

        _, convert = S3_SURFACE_VARS["temperature_2m"]
        assert abs(convert(0.0) - 273.15) < 0.01
        assert abs(convert(20.0) - 293.15) < 0.01

    def test_radiation_conversion(self):
        """Radiation should convert W/m² to J/m²/h."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            S3_SURFACE_VARS,
        )

        _, convert = S3_SURFACE_VARS["shortwave_radiation"]
        # W/m² * 3600 s = J/m²/h
        assert abs(convert(100.0) - 360000.0) < 1.0

    def test_precipitation_conversion(self):
        """Precipitation should convert mm to m."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import (
            S3_SURFACE_VARS,
        )

        _, convert = S3_SURFACE_VARS["precipitation"]
        assert abs(convert(1.0) - 0.001) < 1e-6

    def test_missing_vars_documented(self):
        """Missing variables should be documented."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3 import MISSING_VARS

        assert "strd" in MISSING_VARS  # longwave
        assert "q" in MISSING_VARS     # specific humidity
        assert "z" in MISSING_VARS     # geopotential


class TestOpenMeteoS3BackendRegistry:
    """Test backend registry integration."""

    def test_backend_in_registry(self):
        """S3 backend should be in the registry."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import BACKEND_REGISTRY

        assert "openmeteo_s3" in BACKEND_REGISTRY

    def test_backend_dependency_defined(self):
        """S3 backend dependency should be defined."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import _BACKEND_DEPS

        assert "openmeteo_s3" in _BACKEND_DEPS
        assert _BACKEND_DEPS["openmeteo_s3"] == "omfiles"

    def test_get_backend_instantiates(self, sample_bbox: BBox):
        """get_backend should instantiate S3 backend."""
        pytest.importorskip("omfiles")
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import get_backend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = get_backend(
                "openmeteo_s3",
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            assert backend.__class__.__name__ == "OpenMeteoS3Backend"


class TestOpenMeteoS3BackendSelectBackend:
    """Test backend selection with S3 option."""

    def test_surface_only_includes_s3_in_candidates(self):
        """Surface-only selection should include S3 as fallback."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import select_backend

        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "openmeteo_s3",
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=None,
            )
            assert name == "openmeteo_s3"
            assert kwargs["dataset"] == "era5"

    def test_prefer_s3_selects_s3_first(self):
        """prefer_s3=True should select S3 over API."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import select_backend

        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            return_value=True,  # All available
        ):
            name, _ = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=None,
                prefer_s3=True,
            )
            assert name == "openmeteo_s3"

    def test_plev_needed_skips_s3(self):
        """When pressure levels needed, S3 should not be selected."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import select_backend

        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x in ["openmeteo_s3", "google"],
        ):
            name, _ = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=[850, 500],
            )
            # S3 has no plev, so should fall back to google
            assert name == "google"
