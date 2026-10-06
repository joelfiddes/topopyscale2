"""Tests for CryoGrid output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.cryogrid import CryoGridWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=24, n_units=1):
    """Create test forcing dataset."""
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values

    if n_units == 1:
        return xr.Dataset(
            {
                "temperature": ("time", np.full(nt, 263.15)),  # -10C
                "precipitation": ("time", np.full(nt, 0.001)),
                "rainfall": ("time", np.full(nt, 0.0)),
                "snowfall": ("time", np.full(nt, 0.001)),
                "shortwave_direct": ("time", np.full(nt, 150.0)),
                "shortwave_diffuse": ("time", np.full(nt, 50.0)),
                "longwave": ("time", np.full(nt, 230.0)),
                "humidity_specific": ("time", np.full(nt, 0.002)),
                "humidity_relative": ("time", np.full(nt, 0.80)),
                "wind_speed": ("time", np.full(nt, 4.0)),
                "pressure": ("time", np.full(nt, 98000.0)),
            },
            coords={"time": times},
        )
    else:
        unit_ids = [f"unit_{i:04d}" for i in range(n_units)]
        return xr.Dataset(
            {
                "temperature": (["time", "unit_id"], np.full((nt, n_units), 263.15)),
                "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.001)),
                "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0)),
                "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.001)),
                "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 150.0)),
                "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 50.0)),
                "longwave": (["time", "unit_id"], np.full((nt, n_units), 230.0)),
                "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.002)),
                "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.80)),
                "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 4.0)),
                "pressure": (["time", "unit_id"], np.full((nt, n_units), 98000.0)),
            },
            coords={"time": times, "unit_id": unit_ids},
        )


def make_units(n=1):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(120.0 + i * 0.1, 68.0, 50.0 + i * 10),  # Arctic location
            surface_type="tundra",
            area_m2=10000.0,
        )
        for i in range(n)
    ]


class TestCryoGridWriter:
    def test_write_creates_file(self, tmp_path):
        """Test that write creates a NetCDF file."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".nc"

    def test_file_naming(self, tmp_path):
        """Test CryoGrid file naming convention."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("site_001", (120.0, 68.0, 50.0))
        writer = CryoGridWriter()

        paths = writer.write(ds, [unit], tmp_path)

        # Should contain unit id and date range
        assert "cryogrid_site_001" in paths[0].name
        assert "20200101" in paths[0].name

    def test_cryogrid_variables(self, tmp_path):
        """Test CryoGrid-specific variable names."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        # Check CryoGrid variable names
        assert "Tair" in result.data_vars
        assert "q" in result.data_vars
        assert "wind" in result.data_vars
        assert "Sin" in result.data_vars
        assert "Lin" in result.data_vars
        assert "p" in result.data_vars
        assert "snowfall" in result.data_vars
        assert "rainfall" in result.data_vars

        result.close()

    def test_flux_placeholders(self, tmp_path):
        """Test flux placeholder variables are included."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CryoGridWriter(include_flux_placeholders=True)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "Qh" in result.data_vars
        assert "Qe" in result.data_vars
        assert "Qg" in result.data_vars

        # Check they are NaN placeholders
        assert np.all(np.isnan(result["Qh"].values))
        assert np.all(np.isnan(result["Qg"].values))

        result.close()

    def test_no_flux_placeholders(self, tmp_path):
        """Test flux placeholders can be omitted."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CryoGridWriter(include_flux_placeholders=False)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "Qh" not in result.data_vars
        assert "Qe" not in result.data_vars
        assert "Qg" not in result.data_vars

        result.close()

    def test_temperature_values(self, tmp_path):
        """Test temperature values are preserved."""
        ds = make_forcing_dataset()
        ds["temperature"] = ds["temperature"] * 0 + 253.15  # -20C
        units = make_units(1)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        np.testing.assert_allclose(result["Tair"].values, 253.15)

        result.close()

    def test_shortwave_sum(self, tmp_path):
        """Test Sin is sum of direct and diffuse shortwave."""
        ds = make_forcing_dataset()
        ds["shortwave_direct"] = ds["shortwave_direct"] * 0 + 100.0
        ds["shortwave_diffuse"] = ds["shortwave_diffuse"] * 0 + 40.0
        units = make_units(1)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        np.testing.assert_allclose(result["Sin"].values, 140.0)

        result.close()

    def test_precipitation_rate_conversion(self, tmp_path):
        """Test precipitation converted from amount to rate (m/s)."""
        ds = make_forcing_dataset()
        # 1 kg/m2 per hour = 1 mm per hour
        # = 0.001 m / 3600 s = 2.778e-7 m/s
        ds["snowfall"] = ds["snowfall"] * 0 + 1.0  # 1 kg/m2 per timestep
        units = make_units(1)
        writer = CryoGridWriter(timestep_seconds=3600.0)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        # Rate should be 1/(1000*3600) m/s
        expected_rate = 1.0 / (1000.0 * 3600.0)
        np.testing.assert_allclose(result["snowfall"].values, expected_rate, rtol=1e-6)

        result.close()

    def test_global_attributes(self, tmp_path):
        """Test global attributes are present."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert result.attrs["Conventions"] == "CF-1.8"
        assert "CryoGrid" in result.attrs["title"]
        assert "TopoPyScale" in result.attrs["institution"]
        assert "station_id" in result.attrs
        assert "latitude" in result.attrs
        assert "longitude" in result.attrs
        assert "elevation" in result.attrs

        result.close()

    def test_unit_metadata_in_attrs(self, tmp_path):
        """Test unit metadata is in global attributes."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("arctic_site", (120.5, 68.5, 100.0), surface_type="tundra")
        writer = CryoGridWriter()

        paths = writer.write(ds, [unit], tmp_path)
        result = xr.open_dataset(paths[0])

        assert result.attrs["station_id"] == "arctic_site"
        assert result.attrs["latitude"] == 68.5
        assert result.attrs["longitude"] == 120.5
        assert result.attrs["elevation"] == 100.0
        assert result.attrs["surface_type"] == "tundra"

        result.close()

    def test_multiple_units(self, tmp_path):
        """Test writing multiple units creates multiple files."""
        ds = make_forcing_dataset(n_units=3)
        units = make_units(3)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_compression(self, tmp_path):
        """Test compression option works (file is created successfully)."""
        ds = make_forcing_dataset(nt=100)
        units = make_units(1)

        writer_comp = CryoGridWriter(compression=True, complevel=4)
        writer_uncomp = CryoGridWriter(compression=False)

        paths_comp = writer_comp.write(ds, units, tmp_path / "comp")
        paths_uncomp = writer_uncomp.write(ds, units, tmp_path / "uncomp")

        # Both files should be created and readable
        assert paths_comp[0].exists()
        assert paths_uncomp[0].exists()

        # Verify both are valid NetCDF files
        result_comp = xr.open_dataset(paths_comp[0])
        result_uncomp = xr.open_dataset(paths_uncomp[0])
        assert result_comp.sizes["time"] == 100
        assert result_uncomp.sizes["time"] == 100
        result_comp.close()
        result_uncomp.close()

    def test_write_combined(self, tmp_path):
        """Test write_combined creates single multi-point file."""
        ds = make_forcing_dataset(n_units=3)
        units = make_units(3)
        writer = CryoGridWriter()

        paths = writer.write_combined(ds, units, tmp_path)

        assert len(paths) == 1
        result = xr.open_dataset(paths[0])

        assert "point" in result.dims
        assert result.sizes["point"] == 3

        result.close()

    def test_time_dimension(self, tmp_path):
        """Test time dimension is correct."""
        nt = 48
        ds = make_forcing_dataset(nt=nt)
        units = make_units(1)
        writer = CryoGridWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert result.sizes["time"] == nt

        result.close()
