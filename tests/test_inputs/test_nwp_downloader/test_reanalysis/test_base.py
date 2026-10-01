"""Tests for ERA5Backend abstract base class."""

from __future__ import annotations

import pandas as pd
import pytest
import xarray as xr

from topopyscale2.inputs.nwp_downloader.base import ERA5Backend
from topopyscale2.inputs.nwp_downloader.bbox import BBox


class DummyBackend(ERA5Backend):
    """Concrete implementation for testing the ABC."""

    def fetch_day(self, date: pd.Timestamp) -> tuple[xr.Dataset, xr.Dataset]:
        return xr.Dataset(), xr.Dataset()


class TestERA5BackendABC:
    """Test the ERA5Backend abstract base class."""

    def test_init_stores_bbox(self, sample_bbox: BBox):
        """Backend should store bbox."""
        backend = DummyBackend(
            bbox=sample_bbox,
            pressure_levels=[1000, 850, 700],
        )
        assert backend.bbox == sample_bbox

    def test_init_sorts_pressure_levels(self, sample_bbox: BBox):
        """Backend should sort pressure levels ascending."""
        backend = DummyBackend(
            bbox=sample_bbox,
            pressure_levels=[1000, 500, 850, 700],
        )
        assert backend.pressure_levels == [500, 700, 850, 1000]

    def test_init_default_time_resolution(self, sample_bbox: BBox):
        """Backend should default to 1H time resolution."""
        backend = DummyBackend(
            bbox=sample_bbox,
            pressure_levels=[1000, 850],
        )
        assert backend.time_resolution == "1H"

    def test_init_custom_time_resolution(self, sample_bbox: BBox):
        """Backend should accept custom time resolution."""
        backend = DummyBackend(
            bbox=sample_bbox,
            pressure_levels=[1000, 850],
            time_resolution="3H",
        )
        assert backend.time_resolution == "3H"

    def test_close_is_callable(self, sample_bbox: BBox):
        """Backend close() should be callable (no-op by default)."""
        backend = DummyBackend(
            bbox=sample_bbox,
            pressure_levels=[1000, 850],
        )
        # Should not raise
        backend.close()

    def test_fetch_day_returns_tuple(self, sample_bbox: BBox):
        """fetch_day should return tuple of datasets."""
        backend = DummyBackend(
            bbox=sample_bbox,
            pressure_levels=[1000, 850],
        )
        result = backend.fetch_day(pd.Timestamp("2020-01-01"))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], xr.Dataset)
        assert isinstance(result[1], xr.Dataset)


class TestERA5BackendABCEnforcement:
    """Test that ABC properly enforces interface."""

    def test_cannot_instantiate_abc_directly(self, sample_bbox: BBox):
        """Cannot instantiate ERA5Backend directly."""
        with pytest.raises(TypeError, match="abstract"):
            ERA5Backend(
                bbox=sample_bbox,
                pressure_levels=[1000, 850],
            )

    def test_must_implement_fetch_day(self, sample_bbox: BBox):
        """Subclass must implement fetch_day."""

        class IncompleteBackend(ERA5Backend):
            pass

        with pytest.raises(TypeError, match="fetch_day"):
            IncompleteBackend(
                bbox=sample_bbox,
                pressure_levels=[1000, 850],
            )
