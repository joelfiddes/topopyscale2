"""Tests for HybridBackend."""

from __future__ import annotations

import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.inputs.nwp_downloader.bbox import BBox


class TestHybridBackendInit:
    """Test HybridBackend initialization."""

    def test_init_with_valid_params(self, sample_bbox: BBox):
        """Backend should initialize with valid parameters."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(
                bbox=sample_bbox,
                pressure_levels=[850, 500],
                cache_dir=tmpdir,
            )
            assert backend.bbox == sample_bbox
            # Parent class sorts pressure levels
            assert backend.pressure_levels == [500, 850]

    def test_init_without_pressure_levels(self, sample_bbox: BBox):
        """Backend should initialize without pressure levels."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(
                bbox=sample_bbox,
                pressure_levels=None,
                cache_dir=tmpdir,
            )
            assert backend.pressure_levels == []

    def test_init_include_strd_default(self, sample_bbox: BBox):
        """Backend should include strd by default."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            assert backend.include_strd is True

    def test_init_exclude_strd(self, sample_bbox: BBox):
        """Backend should allow excluding strd."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(
                bbox=sample_bbox,
                include_strd=False,
                cache_dir=tmpdir,
            )
            assert backend.include_strd is False


class TestHybridBackendRegionalDetection:
    """A regional store is used only when one is configured, and only where it covers."""

    # Stand-in extent for a regional store (e.g. a Central Asia subset)
    REGION = (43.0, 24.0, 90.0, 58.0)

    def test_bbox_within(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _bbox_within

        region = BBox.from_tuple(self.REGION)
        assert _bbox_within(BBox.from_tuple((68.0, 39.0, 72.0, 42.0)), region)
        assert not _bbox_within(BBox.from_tuple((6.0, 45.8, 10.5, 47.8)), region)
        assert not _bbox_within(BBox.from_tuple((40.0, 35.0, 50.0, 45.0)), region)  # partial

    def test_no_regional_store_without_url(self):
        """No built-in store: without regional_zarr_url, regional mode is never used."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(bbox=BBox.from_tuple((68.0, 39.0, 72.0, 42.0)),
                                    end_date="2020-06-01", cache_dir=tmpdir)
            assert backend.regional_zarr_url is None
            assert backend._use_regional is False

    def test_outside_region_or_after_store_end_not_regional(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        kw = dict(regional_zarr_url="s3://example/era5.zarr/", regional_bbox=self.REGION,
                  regional_end_date="2023-12-31")
        with tempfile.TemporaryDirectory() as tmpdir:
            europe = HybridBackend(bbox=BBox.from_tuple((6.0, 45.8, 10.5, 47.8)),
                                   end_date="2020-06-01", cache_dir=tmpdir, **kw)
            assert europe._use_regional is False
            late = HybridBackend(bbox=BBox.from_tuple((68.0, 39.0, 72.0, 42.0)),
                                 end_date="2024-06-01", cache_dir=tmpdir, **kw)
            assert late._use_regional is False


class TestS3ZarrNeedsUrl:
    def test_missing_url_is_an_error_naming_the_config_key(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.s3zarr import S3ZarrBackend

        with pytest.raises(ValueError, match="inputs.s3_zarr_url"):
            S3ZarrBackend(bbox=BBox.from_tuple((68.0, 39.0, 72.0, 42.0)), pressure_levels=[500])

class TestHybridBackendRegistry:
    """Test backend registry integration."""

    def test_hybrid_in_registry(self):
        """Hybrid backend should be in the registry."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import BACKEND_REGISTRY

        assert "hybrid" in BACKEND_REGISTRY

    def test_get_backend_instantiates_hybrid(self, sample_bbox: BBox):
        """get_backend should instantiate HybridBackend."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends import get_backend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = get_backend(
                "hybrid",
                bbox=sample_bbox,
                cache_dir=tmpdir,
            )
            assert backend.__class__.__name__ == "HybridBackend"


class TestHybridBackendSubBackendSelection:
    """Test sub-backend selection logic."""

    def test_surface_backend_prefers_s3(self, sample_bbox: BBox):
        """Surface backend should prefer openmeteo_s3."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "openmeteo_s3",
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                backend = HybridBackend(bbox=sample_bbox, cache_dir=tmpdir)
                # Force backend creation by accessing it
                # (would need mocking to fully test)

    def test_plev_backend_requires_google_or_cds(self, sample_bbox: BBox):
        """Plev backend should only use google or cds."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        # Mock only openmeteo_s3 available - should fail to get plev backend
        with patch(
            "topopyscale2.inputs.nwp_downloader.reanalysis.backends._check_available",
            side_effect=lambda x: x == "openmeteo_s3",
        ):
            with tempfile.TemporaryDirectory() as tmpdir:
                backend = HybridBackend(
                    bbox=sample_bbox,
                    pressure_levels=[850],
                    cache_dir=tmpdir,
                )
                # Trying to get plev backend should raise
                with pytest.raises(RuntimeError, match="No plev backend"):
                    backend._get_plev_backend()


class TestHybridBackendPrecipLogic:
    """Test precipitation backend selection."""

    def test_precip_prefers_ifs_for_2022_plus(self, sample_bbox: BBox):
        """Precipitation should prefer IFS for dates >= 2022."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(bbox=sample_bbox, cache_dir=tmpdir)

            # For 2023, should try openmeteo with model=ifs first
            date = pd.Timestamp("2023-06-15")
            # Would need mocking to verify IFS is selected

    def test_precip_uses_s3_for_pre_2022(self, sample_bbox: BBox):
        """Precipitation should use S3 for dates < 2022."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(bbox=sample_bbox, cache_dir=tmpdir)

            # For 2020, should prefer openmeteo_s3 (not IFS)
            date = pd.Timestamp("2020-06-15")
            # Would need mocking to verify S3 is selected


class TestHybridBackendCoordMatching:
    """Test coordinate matching logic."""

    def test_coords_match_identical(self):
        """Identical coordinates should match."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _coords_match

        lats = np.array([46.0, 46.25, 46.5])
        lons = np.array([7.0, 7.25, 7.5])

        ds1 = xr.Dataset(coords={"latitude": lats, "longitude": lons})
        ds2 = xr.Dataset(coords={"latitude": lats, "longitude": lons})

        assert _coords_match(ds1, ds2)

    def test_coords_match_within_tolerance(self):
        """Coordinates within tolerance should match."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _coords_match

        lats1 = np.array([46.0, 46.25, 46.5])
        lats2 = np.array([46.001, 46.251, 46.501])  # Within 0.01 tolerance
        lons = np.array([7.0, 7.25, 7.5])

        ds1 = xr.Dataset(coords={"latitude": lats1, "longitude": lons})
        ds2 = xr.Dataset(coords={"latitude": lats2, "longitude": lons})

        assert _coords_match(ds1, ds2, tol=0.01)

    def test_coords_no_match_different_size(self):
        """Coordinates with different sizes should not match."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _coords_match

        ds1 = xr.Dataset(coords={
            "latitude": np.array([46.0, 46.25, 46.5]),
            "longitude": np.array([7.0, 7.25, 7.5]),
        })
        ds2 = xr.Dataset(coords={
            "latitude": np.array([46.0, 46.25]),  # Different size
            "longitude": np.array([7.0, 7.25]),
        })

        assert not _coords_match(ds1, ds2)

    def test_coords_no_match_outside_tolerance(self):
        """Coordinates outside tolerance should not match."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _coords_match

        lats1 = np.array([46.0, 46.25, 46.5])
        lats2 = np.array([46.1, 46.35, 46.6])  # 0.1 offset
        lons = np.array([7.0, 7.25, 7.5])

        ds1 = xr.Dataset(coords={"latitude": lats1, "longitude": lons})
        ds2 = xr.Dataset(coords={"latitude": lats2, "longitude": lons})

        assert not _coords_match(ds1, ds2, tol=0.01)


class TestHybridBackendVariableGroups:
    """Test variable group definitions."""

    def test_surface_vars_defined(self):
        """Surface variables should be defined."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import SURFACE_VARS

        expected = {"t2m", "d2m", "sp", "ssrd", "u10", "v10", "msl"}
        assert SURFACE_VARS == expected

    def test_precip_vars_defined(self):
        """Precipitation variables should be defined."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import PRECIP_VARS

        assert PRECIP_VARS == {"tp"}

    def test_plev_vars_defined(self):
        """Pressure level variables should be defined."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import PLEV_VARS

        expected = {"t", "z", "u", "v", "q", "r"}
        assert PLEV_VARS == expected

    def test_google_only_vars_defined(self):
        """Google-only variables should be defined."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import (
            GOOGLE_ONLY_PLEV,
            GOOGLE_ONLY_VARS,
        )

        assert "strd" in GOOGLE_ONLY_VARS
        assert "q" in GOOGLE_ONLY_VARS
        assert "u" in GOOGLE_ONLY_PLEV
        assert "v" in GOOGLE_ONLY_PLEV


class TestHybridBackendClose:
    """Test resource cleanup."""

    def test_close_without_backends(self, sample_bbox: BBox):
        """close() should work even if no backends were initialized."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(bbox=sample_bbox, cache_dir=tmpdir)
            # Should not raise
            backend.close()

    def test_close_with_mock_backends(self, sample_bbox: BBox):
        """close() should close all sub-backends."""
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(bbox=sample_bbox, cache_dir=tmpdir)

            # Mock sub-backends
            mock_surface = MagicMock()
            mock_plev = MagicMock()
            backend._surface_backend = mock_surface
            backend._plev_backend = mock_plev

            backend.close()

            mock_surface.close.assert_called_once()
            mock_plev.close.assert_called_once()


class TestOpenMeteoPolicy:
    """Open-Meteo REST API is deprioritised for area requests (rate-limit avoidance)."""

    def test_point_like_bbox(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _is_point_like

        # ~1 ERA5 cell (Zermatt) — point-like
        assert _is_point_like(BBox.from_tuple((7.7, 45.95, 7.85, 46.05)))
        # Degenerate point
        assert _is_point_like(BBox.from_tuple((7.7, 45.95, 7.7, 45.95)))

    def test_area_bbox_not_point_like(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _is_point_like

        # Whole Switzerland — an area
        assert not _is_point_like(BBox.from_tuple((6.0, 45.8, 10.5, 47.8)))

    def test_surface_candidates_area_deprioritises_rest_api(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import (
            _surface_candidates,
        )

        order = _surface_candidates(BBox.from_tuple((6.0, 45.8, 10.5, 47.8)))
        # REST API ("openmeteo") must come after google for areas, and be last resort
        assert order.index("google") < order.index("openmeteo")
        assert order[-1] == "openmeteo"
        # The un-throttled S3 mirror still leads
        assert order[0] == "openmeteo_s3"

    def test_surface_candidates_point_prefers_rest_api(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import (
            _surface_candidates,
        )

        order = _surface_candidates(BBox.from_tuple((7.7, 45.95, 7.85, 46.05)))
        # For point sims the REST API is preferred over google
        assert order.index("openmeteo") < order.index("google")

    def test_precip_candidates_area_pushes_rest_api_last(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import (
            _precip_candidates,
        )

        area = BBox.from_tuple((6.0, 45.8, 10.5, 47.8))
        # ERA5 precip for an area: the rate-limited REST API mirror goes last
        assert _precip_candidates(area, "era5")[-1] == "openmeteo"

    def test_precip_candidates_ifs_only_when_configured(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import (
            _precip_candidates,
        )

        point = BBox.from_tuple((7.7, 45.95, 7.85, 46.05))
        # IFS 9 km precip is available when configured, and only then — no ERA5 fallback
        assert _precip_candidates(point, "ifs") == ["openmeteo"]


class TestHybridFetchFallback:
    """A sub-backend fetch failure (e.g. HTTP 429) falls back to Google/CDS."""

    def test_surface_failure_falls_back(
        self, sample_bbox: BBox, sample_surface_dataset: xr.Dataset
    ):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            # No plev / strd → surface-only path, keeps the test focused.
            backend = HybridBackend(
                bbox=sample_bbox,
                pressure_levels=None,
                include_strd=False,
                cache_dir=tmpdir,
            )

            # Primary surface backend raises like a rate-limited API.
            failing = MagicMock()
            failing.fetch_day.side_effect = RuntimeError("429 Too Many Requests")
            # Same object for precip so no separate precip future is launched.
            backend._surface_backend = failing
            backend._precip_backend = failing

            # Fallback returns real data.
            fallback = MagicMock()
            fallback.fetch_day.return_value = (sample_surface_dataset, xr.Dataset())
            backend._fallback_backend = fallback

            ds_surf, _ = backend.fetch_day(pd.Timestamp("2023-03-01"))

            fallback.fetch_day.assert_called_once()
            assert len(ds_surf.data_vars) > 0

    def test_surface_failure_reraises_without_fallback(self, sample_bbox: BBox):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            backend = HybridBackend(
                bbox=sample_bbox,
                pressure_levels=None,
                include_strd=False,
                cache_dir=tmpdir,
            )

            failing = MagicMock()
            failing.fetch_day.side_effect = RuntimeError("429 Too Many Requests")
            backend._surface_backend = failing
            backend._precip_backend = failing

            # No fallback available → original error surfaces.
            with patch.object(backend, "_get_fallback_backend", return_value=None):
                with pytest.raises(RuntimeError, match="429"):
                    backend.fetch_day(pd.Timestamp("2023-03-01"))


class TestSourceSelectionIsConfiguration:
    """The source is what the configuration names — never inferred from dates or the domain."""

    def test_precip_candidates_never_mix_models(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _precip_candidates

        point = BBox.from_tuple((7.70, 46.00, 7.75, 46.05))
        area = BBox.from_tuple((6.0, 45.0, 10.0, 48.0))
        for bbox in (point, area):
            # ERA5 is served only by ERA5 mirrors, whatever the domain size
            assert set(_precip_candidates(bbox, "era5")) == {"openmeteo_s3", "openmeteo", "google", "cds"}
            # IFS has exactly one server and no fallback to another model
            assert _precip_candidates(bbox, "ifs") == ["openmeteo"]

    def test_default_precip_model_is_era5_for_any_date(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            for end in ("2019-06-01", "2024-06-01"):
                b = HybridBackend(bbox=BBox.from_tuple((7.7, 46.0, 7.75, 46.05)), end_date=end, cache_dir=tmpdir)
                assert b.precip_model == "era5"

    def test_unknown_precip_model_rejected(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import HybridBackend

        with pytest.raises(ValueError, match="precip_model"):
            HybridBackend(bbox=BBox.from_tuple((7.7, 46.0, 7.75, 46.05)), precip_model="icon")

    def test_auto_reanalysis_backend_never_picks_ifs(self, monkeypatch):
        import pandas as pd

        from topopyscale2.inputs.nwp_downloader.reanalysis import backends

        monkeypatch.setattr(backends, "_check_available", lambda name: True)
        for start in ("2015-01-01", "2024-01-01"):
            name, kwargs = backends.select_backend(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=5),
                                                   pressure_levels=[500, 700, 850])
            assert name in ("google", "cds") and kwargs.get("model") != "ifs"
            name, kwargs = backends.select_backend(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=5),
                                                   pressure_levels=[])
            assert kwargs.get("model", "era5") == "era5" and kwargs.get("dataset", "era5") == "era5"

    def test_source_label(self):
        from topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid import _source_label

        class OpenMeteoBackend:
            _model = "ifs"

        class GoogleBackend:
            pass

        assert _source_label(OpenMeteoBackend()) == "openmeteo:ifs"
        assert _source_label(GoogleBackend()) == "google"
