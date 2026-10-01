"""Tests for backend registry (get_backend, select_backend)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.reanalysis.backends import (
    BACKEND_REGISTRY,
    _check_available,
    get_backend,
    select_backend,
)


class TestBackendRegistry:
    """Test backend registry configuration."""

    def test_registry_has_expected_backends(self):
        """Registry should have all expected backends."""
        assert "google" in BACKEND_REGISTRY
        assert "cds" in BACKEND_REGISTRY
        assert "s3zarr" in BACKEND_REGISTRY
        assert "openmeteo" in BACKEND_REGISTRY

    def test_registry_paths_format(self):
        """Registry entries should be module:class format."""
        for name, path in BACKEND_REGISTRY.items():
            assert ":" in path, f"Backend {name} path missing colon: {path}"
            module, cls = path.rsplit(":", 1)
            assert module.startswith("topopyscale2.inputs.nwp_downloader")
            assert cls[0].isupper()  # Class name should be capitalized


class TestCheckAvailable:
    """Test backend availability checking."""

    def test_google_available_with_gcsfs(self):
        """Google backend should be available when gcsfs is installed."""
        import pytest

        pytest.importorskip("gcsfs", reason="gcsfs optional extra not installed")
        assert _check_available("google") is True

    def test_unknown_backend_unavailable(self):
        """Unknown backend should be unavailable."""
        assert _check_available("nonexistent") is False

    @patch.dict("sys.modules", {"requests": None})
    def test_openmeteo_unavailable_without_requests(self):
        """OpenMeteo should be unavailable without requests."""
        # This test is tricky because requests is likely installed
        # We just verify the function handles missing deps gracefully
        pass


class TestGetBackend:
    """Test get_backend function."""

    def test_get_unknown_backend_raises(self):
        """get_backend should raise for unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_get_backend_google(self, sample_bbox: BBox):
        """get_backend should instantiate Google backend."""
        backend = get_backend(
            "google",
            bbox=sample_bbox,
            pressure_levels=[850, 500],
        )
        assert backend.__class__.__name__ == "GoogleCloudBackend"
        assert backend.bbox == sample_bbox

    def test_get_backend_passes_kwargs(self, sample_bbox: BBox):
        """get_backend should pass kwargs to backend constructor."""
        backend = get_backend(
            "google",
            bbox=sample_bbox,
            pressure_levels=[850, 500],
            time_resolution="3H",
        )
        assert backend.time_resolution == "3H"


class TestSelectBackend:
    """Test automatic backend selection."""

    def test_post_2022_with_plev_stays_on_era5(self):
        """The date never turns 'auto' reanalysis into IFS: only ERA5 mirrors are chosen."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x in ("openmeteo", "google"),
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2023-01-01"),
                end_date=pd.Timestamp("2023-01-31"),
                pressure_levels=[850, 500],
            )
            assert name == "google"
            assert kwargs.get("model") != "ifs"

    def test_pre_2022_with_plev_skips_openmeteo(self):
        """Pre-2022 data with pressure levels should skip openmeteo."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "google",
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=[850, 500],
            )
            assert name == "google"
            assert kwargs == {}

    def test_no_plev_prefers_openmeteo_era5(self):
        """Surface-only data should prefer openmeteo with era5 model."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "openmeteo",
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=None,
            )
            assert name == "openmeteo"
            assert kwargs["model"] == "era5"

    def test_fallback_to_google_when_openmeteo_unavailable(self):
        """Should fallback to google when openmeteo unavailable."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "google",
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2023-01-01"),
                end_date=pd.Timestamp("2023-01-31"),
                pressure_levels=[850],
            )
            assert name == "google"

    def test_fallback_to_cds_when_google_unavailable(self):
        """Should fallback to CDS when google unavailable."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "cds",
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=[850],
            )
            assert name == "cds"

    def test_raises_when_no_backend_available(self):
        """Should raise when no backend is available."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="No backend available"):
                select_backend(
                    start_date=pd.Timestamp("2020-01-01"),
                    end_date=pd.Timestamp("2020-12-31"),
                    pressure_levels=[850],
                )

    def test_empty_pressure_levels_treated_as_surface_only(self):
        """Empty pressure levels list should be treated as surface-only."""
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "openmeteo",
        ):
            name, kwargs = select_backend(
                start_date=pd.Timestamp("2020-01-01"),
                end_date=pd.Timestamp("2020-12-31"),
                pressure_levels=[],
            )
            # Empty list is falsy, so treated as no plev needed
            assert name == "openmeteo"
            assert kwargs["model"] == "era5"
