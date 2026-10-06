"""Tests for Crocus/SAFRAN output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.crocus import CrocusWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=24, n_units=1):
    """Create test forcing dataset."""
    times = pd.date_range("2020-01-15", periods=nt, freq="h").values

    if n_units == 1:
        return xr.Dataset(
            {
                "temperature": ("time", np.full(nt, 268.15)),  # -5C
                "precipitation": ("time", np.full(nt, 0.002)),
                "rainfall": ("time", np.full(nt, 0.0)),
                "snowfall": ("time", np.full(nt, 0.002)),
                "shortwave_direct": ("time", np.full(nt, 180.0)),
                "shortwave_diffuse": ("time", np.full(nt, 70.0)),
                "longwave": ("time", np.full(nt, 250.0)),
                "humidity_specific": ("time", np.full(nt, 0.003)),
                "humidity_relative": ("time", np.full(nt, 0.85)),
                "wind_speed": ("time", np.full(nt, 5.0)),
                "pressure": ("time", np.full(nt, 75000.0)),
            },
            coords={"time": times},
        )
    else:
        unit_ids = [f"unit_{i:04d}" for i in range(n_units)]
        return xr.Dataset(
            {
                "temperature": (["time", "unit_id"], np.full((nt, n_units), 268.15)),
                "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.002)),
                "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0)),
                "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.002)),
                "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 180.0)),
                "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 70.0)),
                "longwave": (["time", "unit_id"], np.full((nt, n_units), 250.0)),
                "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.003)),
                "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.85)),
                "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 5.0)),
                "pressure": (["time", "unit_id"], np.full((nt, n_units), 75000.0)),
            },
            coords={"time": times, "unit_id": unit_ids},
        )


def make_units(n=1):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(6.5 + i * 0.1, 45.0, 2000.0 + i * 200),  # Alps location
            surface_type="glacier",
            area_m2=10000.0,
        )
        for i in range(n)
    ]


class TestCrocusWriter:
    def test_write_creates_file(self, tmp_path):
        """Test that write creates a NetCDF file."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".nc"

    def test_file_naming_forcing(self, tmp_path):
        """Test Crocus FORCING file naming convention."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("point_001", (6.5, 45.0, 2000.0))
        writer = CrocusWriter()

        paths = writer.write(ds, [unit], tmp_path)

        # Should start with FORCING_
        assert paths[0].name.startswith("FORCING_")
        assert "point_001" in paths[0].name

    def test_crocus_variables(self, tmp_path):
        """Test Crocus/SAFRAN-specific variable names."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        # Check SURFEX/Crocus variable names
        assert "Tair" in result.data_vars
        assert "Qair" in result.data_vars
        assert "Wind" in result.data_vars
        assert "PSurf" in result.data_vars
        assert "DIR_SWdown" in result.data_vars
        assert "SCA_SWdown" in result.data_vars
        assert "LWdown" in result.data_vars
        assert "Rainf" in result.data_vars
        assert "Snowf" in result.data_vars

        result.close()

    def test_separate_shortwave_components(self, tmp_path):
        """Test direct and diffuse shortwave are separate."""
        ds = make_forcing_dataset()
        ds["shortwave_direct"] = ds["shortwave_direct"] * 0 + 200.0
        ds["shortwave_diffuse"] = ds["shortwave_diffuse"] * 0 + 80.0
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        np.testing.assert_allclose(result["DIR_SWdown"].values.flatten(), 200.0)
        np.testing.assert_allclose(result["SCA_SWdown"].values.flatten(), 80.0)

        result.close()

    def test_precipitation_rate_conversion(self, tmp_path):
        """Test precipitation converted to rate (kg/m2/s)."""
        ds = make_forcing_dataset()
        # 3.6 kg/m2 per hour = 1e-3 kg/m2/s
        ds["snowfall"] = ds["snowfall"] * 0 + 3.6
        units = make_units(1)
        writer = CrocusWriter(timestep_seconds=3600.0)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        expected_rate = 3.6 / 3600.0  # kg/m2/s
        np.testing.assert_allclose(
            result["Snowf"].values.flatten(), expected_rate, rtol=1e-6
        )

        result.close()

    def test_number_of_points_dimension(self, tmp_path):
        """Test Number_of_points dimension is present (SURFEX convention)."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "Number_of_points" in result.dims
        assert result.sizes["Number_of_points"] == 1

        result.close()

    def test_coordinate_variables(self, tmp_path):
        """Test LAT, LON, ZS coordinates are present."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("test", (6.75, 45.5, 2500.0))
        writer = CrocusWriter()

        paths = writer.write(ds, [unit], tmp_path)
        result = xr.open_dataset(paths[0])

        assert "LAT" in result.coords
        assert "LON" in result.coords
        assert "ZS" in result.coords

        np.testing.assert_allclose(result["LAT"].values, 45.5)
        np.testing.assert_allclose(result["LON"].values, 6.75)
        np.testing.assert_allclose(result["ZS"].values, 2500.0)

        result.close()

    def test_global_attributes(self, tmp_path):
        """Test global attributes are present."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "Conventions" in result.attrs
        assert "Crocus" in result.attrs["title"] or "SAFRAN" in result.attrs["title"]
        assert "TopoPyScale" in result.attrs["institution"]

        result.close()

    def test_wind_direction_optional(self, tmp_path):
        """Test optional wind direction variable."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter(include_wind_dir=True)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "Wind_DIR" in result.data_vars
        # Should be NaN (placeholder)
        assert np.all(np.isnan(result["Wind_DIR"].values))

        result.close()

    def test_no_wind_direction(self, tmp_path):
        """Test wind direction can be excluded."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter(include_wind_dir=False)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "Wind_DIR" not in result.data_vars

        result.close()

    def test_co2_optional(self, tmp_path):
        """Test optional CO2 concentration variable."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter(include_co2=True, co2_ppm=420.0)

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert "CO2air" in result.data_vars
        # Check it's approximately correct (420 ppm -> kg/kg)
        co2_values = result["CO2air"].values
        assert np.all(co2_values > 0)

        result.close()

    def test_multiple_units(self, tmp_path):
        """Test writing multiple units creates multiple files."""
        ds = make_forcing_dataset(n_units=4)
        units = make_units(4)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 4
        for path in paths:
            assert path.exists()

    def test_compression(self, tmp_path):
        """Test compression option works (file is created successfully)."""
        ds = make_forcing_dataset(nt=100)
        units = make_units(1)

        writer_comp = CrocusWriter(compression=True, complevel=4)
        writer_uncomp = CrocusWriter(compression=False)

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
        ds = make_forcing_dataset(n_units=5)
        units = make_units(5)
        writer = CrocusWriter()

        paths = writer.write_combined(ds, units, tmp_path)

        assert len(paths) == 1
        result = xr.open_dataset(paths[0])

        assert "Number_of_points" in result.dims
        assert result.sizes["Number_of_points"] == 5

        result.close()

    def test_time_dimension(self, tmp_path):
        """Test time dimension is correct."""
        nt = 72
        ds = make_forcing_dataset(nt=nt)
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert result.sizes["time"] == nt

        result.close()

    def test_variable_units(self, tmp_path):
        """Test variable units attributes."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CrocusWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert result["Tair"].attrs["units"] == "K"
        assert result["PSurf"].attrs["units"] == "Pa"
        assert result["DIR_SWdown"].attrs["units"] == "W m-2"
        assert result["Rainf"].attrs["units"] == "kg m-2 s-1"

        result.close()
