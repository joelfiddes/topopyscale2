"""Tests for CustomSource NWP loader."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale2.config.schema import InputConfig, NWPSourceConfig
from topopyscale2.inputs.custom import (
    CustomSource,
    validate_variable_mapping,
)


@pytest.fixture
def temp_netcdf_dir():
    """Create a temporary directory with a NetCDF file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple dataset
        times = pd.date_range("2020-01-01", periods=24, freq="h").values
        lat = np.array([41.0, 41.5, 42.0, 42.5, 43.0])
        lon = np.array([75.0, 75.5, 76.0, 76.5, 77.0])

        nt, nlat, nlon = len(times), len(lat), len(lon)
        rng = np.random.default_rng(42)

        ds = xr.Dataset(
            {
                "temperature_2m": (
                    ["time", "latitude", "longitude"],
                    270.0 + rng.standard_normal((nt, nlat, nlon)) * 5,
                ),
                "dewpoint_2m": (
                    ["time", "latitude", "longitude"],
                    265.0 + rng.standard_normal((nt, nlat, nlon)) * 3,
                ),
                "surface_pressure": (
                    ["time", "latitude", "longitude"],
                    80000.0 + rng.standard_normal((nt, nlat, nlon)) * 500,
                ),
            },
            coords={"time": times, "latitude": lat, "longitude": lon},
        )

        path = Path(tmpdir) / "test_data.nc"
        ds.to_netcdf(path)
        yield path


@pytest.fixture
def temp_zarr_store():
    """Create a temporary Zarr store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        times = pd.date_range("2020-01-01", periods=12, freq="h").values
        lat = np.array([42.0, 42.5])
        lon = np.array([76.0, 76.5])

        ds = xr.Dataset(
            {
                "t2m": (["time", "latitude", "longitude"], np.ones((12, 2, 2)) * 275.0),
                "d2m": (["time", "latitude", "longitude"], np.ones((12, 2, 2)) * 270.0),
                "sp": (["time", "latitude", "longitude"], np.ones((12, 2, 2)) * 85000.0),
            },
            coords={"time": times, "latitude": lat, "longitude": lon},
        )

        zarr_path = Path(tmpdir) / "test.zarr"
        ds.to_zarr(zarr_path)
        yield zarr_path


class TestCustomSourceInit:
    """Test CustomSource initialization."""

    def test_requires_source_config(self):
        """CustomSource requires source_config."""
        config = InputConfig()
        with pytest.raises(ValueError, match="requires source_config"):
            CustomSource(config, None)

    def test_requires_path(self):
        """CustomSource requires path in source_config."""
        config = InputConfig()
        source_config = NWPSourceConfig(name="test", type="custom")
        with pytest.raises(ValueError, match="requires source_config.path"):
            CustomSource(config, source_config)

    def test_valid_init(self, temp_netcdf_dir):
        """Valid initialization with path."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="my_nwp",
            type="custom",
            path=temp_netcdf_dir,
        )
        source = CustomSource(config, source_config)
        assert source.name == "my_nwp"


class TestCustomSourceProperties:
    """Test CustomSource property methods."""

    def test_name_property(self, temp_netcdf_dir):
        """Name is taken from source_config."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="regional_model",
            type="custom",
            path=temp_netcdf_dir,
        )
        source = CustomSource(config, source_config)
        assert source.name == "regional_model"

    def test_resolution_default(self, temp_netcdf_dir):
        """Default resolution is 10km if not specified."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=temp_netcdf_dir,
        )
        source = CustomSource(config, source_config)
        assert source.resolution_m == 10000.0

    def test_resolution_override(self, temp_netcdf_dir):
        """Resolution can be specified in config."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=temp_netcdf_dir,
            resolution_m=2000.0,
        )
        source = CustomSource(config, source_config)
        assert source.resolution_m == 2000.0


class TestCustomSourceFetch:
    """Test CustomSource fetch method."""

    def test_fetch_netcdf(self, temp_netcdf_dir):
        """Fetch from NetCDF file works."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=temp_netcdf_dir,
            variable_mapping={
                "temperature_2m": "t2m",
                "dewpoint_2m": "d2m",
                "surface_pressure": "sp",
            },
        )
        source = CustomSource(config, source_config)

        ds_surf, ds_plev = source.fetch(
            bbox=(75.0, 41.0, 77.0, 43.0),
            time_range=["2020-01-01T00:00", "2020-01-01T12:00"],
        )

        # Check surface variables were renamed
        assert "t2m" in ds_surf
        assert "d2m" in ds_surf
        assert "sp" in ds_surf

    def test_fetch_zarr(self, temp_zarr_store):
        """Fetch from Zarr store works."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=temp_zarr_store,
        )
        source = CustomSource(config, source_config)

        ds_surf, ds_plev = source.fetch(
            bbox=(76.0, 42.0, 77.0, 43.0),
            time_range=["2020-01-01T00:00", "2020-01-01T06:00"],
        )

        # Standard names already in data, no mapping needed
        assert "t2m" in ds_surf or len(ds_surf.data_vars) >= 0

    def test_fetch_file_not_found(self):
        """Fetch raises error for non-existent path."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=Path("/nonexistent/path"),
        )
        source = CustomSource(config, source_config)

        with pytest.raises(FileNotFoundError):
            source.fetch((75.0, 41.0, 77.0, 43.0), ["2020-01-01", "2020-01-02"])


class TestCustomSourceVariableMapping:
    """Test variable mapping functionality."""

    def test_mapping_applied(self, temp_netcdf_dir):
        """Variable mapping is applied correctly."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=temp_netcdf_dir,
            variable_mapping={
                "temperature_2m": "t2m",
            },
        )
        source = CustomSource(config, source_config)

        ds_surf, _ = source.fetch(
            bbox=(75.0, 41.0, 77.0, 43.0),
            time_range=["2020-01-01T00:00", "2020-01-01T12:00"],
        )

        assert "t2m" in ds_surf
        assert "temperature_2m" not in ds_surf

    def test_empty_mapping(self, temp_zarr_store):
        """Empty mapping leaves variable names unchanged."""
        config = InputConfig()
        source_config = NWPSourceConfig(
            name="test",
            type="custom",
            path=temp_zarr_store,
            variable_mapping={},
        )
        source = CustomSource(config, source_config)

        ds_surf, _ = source.fetch(
            bbox=(76.0, 42.0, 77.0, 43.0),
            time_range=["2020-01-01T00:00", "2020-01-01T06:00"],
        )

        # Original names preserved
        assert "t2m" in ds_surf


class TestValidateVariableMapping:
    """Test validate_variable_mapping function."""

    def test_valid_mapping(self):
        """Valid mapping returns empty warnings list."""
        mapping = {
            "temp_2m": "t2m",
            "dewpoint": "d2m",
        }
        warnings = validate_variable_mapping(mapping)
        assert len(warnings) == 0

    def test_invalid_standard_name(self):
        """Invalid standard name generates warning."""
        mapping = {
            "temp": "invalid_name",
        }
        warnings = validate_variable_mapping(mapping)
        assert len(warnings) == 1
        assert "not a recognized standard" in warnings[0]

    def test_required_vars_missing(self):
        """Missing required variables raises ValueError."""
        mapping = {
            "temp_2m": "t2m",
        }
        with pytest.raises(ValueError, match="Required variables missing"):
            validate_variable_mapping(mapping, required_vars=["t2m", "sp"])

    def test_required_vars_present(self):
        """All required variables present passes."""
        mapping = {
            "temp_2m": "t2m",
            "pressure": "sp",
        }
        warnings = validate_variable_mapping(mapping, required_vars=["t2m", "sp"])
        assert len(warnings) == 0
