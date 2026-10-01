"""Tests for ERA5Loader orchestrator."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.reanalysis import ERA5Loader


class TestERA5LoaderInit:
    """Test ERA5Loader initialization."""

    def test_init_with_tuple_bbox(self):
        """ERA5Loader should accept tuple bbox."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert isinstance(loader.bbox, BBox)
            assert loader.bbox.west == 7.7
            assert loader.bbox.south == 45.95

    def test_init_with_bbox_instance(self, sample_bbox: BBox):
        """ERA5Loader should accept BBox instance."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=sample_bbox,
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert loader.bbox == sample_bbox

    def test_init_parses_dates(self):
        """ERA5Loader should parse date strings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-12-31",
                output_dir=tmpdir,
            )
            assert loader.start_date == pd.Timestamp("2020-01-01")
            assert loader.end_date == pd.Timestamp("2020-12-31")

    def test_init_default_pressure_levels(self):
        """ERA5Loader should have default pressure levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert loader.pressure_levels == [300, 500, 700, 850, 1000]

    def test_init_custom_pressure_levels(self):
        """ERA5Loader should accept custom pressure levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                pressure_levels=[850, 500],
                output_dir=tmpdir,
            )
            assert loader.pressure_levels == [850, 500]

    def test_init_default_output_format(self):
        """ERA5Loader should default to netcdf output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert loader.output_format == "netcdf"

    def test_init_zarr_output_format(self):
        """ERA5Loader should accept zarr output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_format="zarr",
                output_dir=tmpdir,
            )
            assert loader.output_format == "zarr"

    def test_init_invalid_output_format_raises(self):
        """ERA5Loader should reject invalid output format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Unknown output_format"):
                ERA5Loader(
                    backend="google",
                    bbox=(7.7, 45.95, 7.85, 46.05),
                    start_date="2020-01-01",
                    end_date="2020-01-03",
                    output_format="parquet",
                    output_dir=tmpdir,
                )

    def test_init_default_time_resolution(self):
        """ERA5Loader should default to 1H time resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert loader.time_resolution == "1H"

    def test_init_custom_time_resolution(self):
        """ERA5Loader should accept custom time resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                time_resolution="3H",
                output_dir=tmpdir,
            )
            assert loader.time_resolution == "3H"

    def test_init_default_max_workers(self):
        """ERA5Loader should default to 4 workers."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert loader.max_workers == 4

    def test_init_default_compute_rh(self):
        """ERA5Loader should default to computing RH."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            assert loader.compute_rh is True


class TestBuildDateList:
    """Test date list building."""

    def test_build_date_list_single_day(self):
        """Single day should return one date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-15",
                end_date="2020-01-15",
                output_dir=tmpdir,
            )
            dates = loader._build_date_list()
            assert len(dates) == 1
            assert dates[0] == pd.Timestamp("2020-01-15")

    def test_build_date_list_multiple_days(self):
        """Multiple days should return all dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-05",
                output_dir=tmpdir,
            )
            dates = loader._build_date_list()
            assert len(dates) == 5
            assert dates[0] == pd.Timestamp("2020-01-01")
            assert dates[-1] == pd.Timestamp("2020-01-05")

    def test_build_date_list_month(self):
        """Full month should return ~30 dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-06-01",
                end_date="2020-06-30",
                output_dir=tmpdir,
            )
            dates = loader._build_date_list()
            assert len(dates) == 30


class TestFilterExisting:
    """Test filtering of already-downloaded dates."""

    def test_filter_existing_empty_dir(self):
        """Empty directory should not filter any dates."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )
            dates = loader._build_date_list()
            filtered = loader._filter_existing(dates)
            assert len(filtered) == len(dates)

    def test_filter_existing_skip_disabled(self):
        """With skip_existing=False, should not filter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=(7.7, 45.95, 7.85, 46.05),
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
                skip_existing=False,
            )
            dates = loader._build_date_list()
            filtered = loader._filter_existing(dates)
            assert len(filtered) == len(dates)


class TestAutoBackendSelection:
    """Test automatic backend selection via 'auto' backend."""

    @patch("topopyscale2.inputs.nwp_downloader.reanalysis.backends.select_backend")
    @patch("topopyscale2.inputs.nwp_downloader.reanalysis.backends.get_backend")
    def test_auto_backend_calls_select_backend(
        self,
        mock_get_backend,
        mock_select_backend,
        sample_bbox: BBox,
    ):
        """Auto backend should call select_backend."""
        mock_select_backend.return_value = ("google", {})
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="auto",
                bbox=sample_bbox,
                start_date="2020-01-01",
                end_date="2020-01-03",
                output_dir=tmpdir,
            )

        mock_select_backend.assert_called_once()
        mock_get_backend.assert_called_once_with(
            "google",
            bbox=sample_bbox,
            pressure_levels=[300, 500, 700, 850, 1000],
            time_resolution="1H",
        )


class TestBackendInstantiation:
    """Test backend instantiation with different backends."""

    def test_google_backend_instantiation(self, sample_bbox: BBox):
        """Should instantiate Google backend correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = ERA5Loader(
                backend="google",
                bbox=sample_bbox,
                start_date="2020-01-01",
                end_date="2020-01-03",
                pressure_levels=[850, 500],
                output_dir=tmpdir,
            )
            assert loader._backend.__class__.__name__ == "GoogleCloudBackend"
            assert loader._backend.pressure_levels == [500, 850]
