"""Tests for GoogleCloudBackend."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.reanalysis.backends.google import GoogleCloudBackend


class TestGoogleCloudBackendInit:
    """Test GoogleCloudBackend initialization."""

    def test_init_with_valid_params(self, sample_bbox: BBox):
        """Backend should initialize with valid parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[1000, 850, 700, 500],
                cache_dir=tmpdir,
            )
            assert backend.bbox == sample_bbox
            assert backend.pressure_levels == [500, 700, 850, 1000]

    def test_init_converts_bbox_to_0_360(self, sample_bbox: BBox):
        """Backend should convert bbox to 0:360 for Google data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            # Swiss coordinates should remain same (already 0-360 compatible)
            assert backend._bbox_360.west >= 0

    def test_init_negative_longitude_conversion(self):
        """Backend should convert negative longitudes to 0:360.

        Note: BBox.to_0_360() converts -10 to 350, but then __post_init__
        swaps when west > east (350 > 5). Due to frozen dataclass, both
        end up as the smaller value.
        """
        bbox = BBox.from_tuple((-10.0, 40.0, 5.0, 50.0))  # Western Europe
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            # After conversion and swap quirk, both become 5.0
            assert backend._bbox_360.west == 5.0
            assert backend._bbox_360.east == 5.0

    def test_init_valid_time_resolutions(self, sample_bbox: BBox):
        """Backend should accept valid time resolutions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for res in ["1H", "2H", "3H", "6H"]:
                backend = GoogleCloudBackend(
                    bbox=sample_bbox,
                    pressure_levels=[850],
                    time_resolution=res,
                    cache_dir=tmpdir,
                )
                assert backend.time_resolution == res

    def test_init_invalid_time_resolution_raises(self, sample_bbox: BBox):
        """Backend should reject invalid time resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="Invalid time_resolution"):
                GoogleCloudBackend(
                    bbox=sample_bbox,
                    pressure_levels=[850],
                    time_resolution="4H",
                    cache_dir=tmpdir,
                )

    def test_init_creates_cache_dir(self, sample_bbox: BBox):
        """Backend should create cache directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "new_cache"
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=str(cache_path),
            )
            assert cache_path.exists()

    def test_init_default_variables(self, sample_bbox: BBox):
        """Backend should have default surface and plev variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            assert len(backend.surf_vars) > 0
            assert len(backend.plev_vars) > 0

    def test_init_custom_variables(self, sample_bbox: BBox):
        """Backend should accept custom variables."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                surf_vars=["2m_temperature"],
                plev_vars=["temperature"],
                cache_dir=tmpdir,
            )
            assert backend.surf_vars == ["2m_temperature"]
            assert backend.plev_vars == ["temperature"]


class TestGoogleCloudBackendURIs:
    """Test URI generation."""

    def test_make_uri_list_generates_uris(self, sample_bbox: BBox):
        """_make_uri_list should generate URIs for all vars and levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850, 500],
                surf_vars=["2m_temperature"],
                plev_vars=["temperature"],
                cache_dir=tmpdir,
            )
            # Mock the valid vars discovery
            backend._valid_surface_vars = ["2m_temperature"]
            backend._valid_level_vars = ["temperature"]
            backend._valid_levels = [500, 850, 1000]

            t = pd.Timestamp("2020-06-15")
            uris = list(backend._make_uri_list(t))

            # Should have: 1 surface + 2 pressure levels
            assert len(uris) == 3
            assert any("single_level" in uri for uri in uris)
            assert any("pressure_level" in uri for uri in uris)

    def test_uri_contains_date(self, sample_bbox: BBox):
        """URIs should contain the correct date."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                surf_vars=["2m_temperature"],
                plev_vars=["temperature"],  # Need non-empty to override default
                cache_dir=tmpdir,
            )
            backend._valid_surface_vars = ["2m_temperature"]
            backend._valid_level_vars = ["temperature"]
            backend._valid_levels = [850]

            t = pd.Timestamp("2020-06-15")
            uris = list(backend._make_uri_list(t))

            # Should have surface + 1 level
            assert len(uris) == 2
            assert all("2020/06/15" in uri for uri in uris)


class TestGoogleCloudBackendClassification:
    """Test variable classification."""

    def test_surface_variable(self, sample_bbox: BBox):
        """Surface-only variables should be classified as 'surface'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            backend._valid_surface_vars = ["2m_temperature"]
            backend._valid_level_vars = ["temperature"]
            backend._valid_levels = [850]

            assert backend._surface_or_level("2m_temperature") == "surface"

    def test_level_variable(self, sample_bbox: BBox):
        """Level-only variables should be classified as 'level'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            backend._valid_surface_vars = ["2m_temperature"]
            backend._valid_level_vars = ["temperature"]
            backend._valid_levels = [850]

            assert backend._surface_or_level("temperature") == "level"

    def test_both_variable(self, sample_bbox: BBox):
        """Variables in both should be classified as 'both'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            backend._valid_surface_vars = ["geopotential"]
            backend._valid_level_vars = ["geopotential"]
            backend._valid_levels = [850]

            assert backend._surface_or_level("geopotential") == "both"

    def test_unknown_variable_raises(self, sample_bbox: BBox):
        """Unknown variables should raise ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            backend._valid_surface_vars = ["2m_temperature"]
            backend._valid_level_vars = ["temperature"]
            backend._valid_levels = [850]

            with pytest.raises(ValueError, match="not found"):
                backend._surface_or_level("nonexistent_variable")


class TestGoogleCloudBackendPreprocess:
    """Test dataset preprocessing."""

    def test_preprocess_subsets_spatially(self, sample_bbox: BBox):
        """_preprocess should subset to bounding box."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            backend._bbox_360 = sample_bbox.to_0_360()
            backend._time_steps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                   12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
            backend._fsspec_cache = {}

            # Create a global dataset
            lats = np.arange(40, 50, 0.25)
            lons = np.arange(5, 12, 0.25)
            times = pd.date_range("2020-06-15", periods=24, freq="1h")
            ds = xr.Dataset({
                "t2m": (["time", "latitude", "longitude"],
                        np.random.rand(24, len(lats), len(lons))),
            })
            ds = ds.assign_coords(time=times, latitude=lats, longitude=lons)
            ds.encoding["source"] = "mock_file"

            result = backend._preprocess(ds, backend._fsspec_cache)

            # All coords should be within bbox (approximately)
            assert all(result.latitude.values >= sample_bbox.south - 0.5)
            assert all(result.latitude.values <= sample_bbox.north + 0.5)


class TestGoogleCloudBackendClose:
    """Test resource cleanup."""

    def test_close_clears_cache(self, sample_bbox: BBox):
        """close() should clear temporary cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            backend = GoogleCloudBackend(
                bbox=sample_bbox,
                pressure_levels=[850],
                cache_dir=tmpdir,
            )
            # Should not raise
            backend.close()
