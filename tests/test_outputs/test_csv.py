"""Tests for CSV output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.csv_writer import CSVWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=24, n_units=1):
    """Create test forcing dataset."""
    times = pd.date_range("2020-03-15 00:00", periods=nt, freq="h").values

    if n_units == 1:
        return xr.Dataset(
            {
                "temperature": ("time", np.full(nt, 278.15)),  # 5C
                "precipitation": ("time", np.full(nt, 0.0002)),
                "rainfall": ("time", np.full(nt, 0.0002)),
                "snowfall": ("time", np.full(nt, 0.0)),
                "shortwave_direct": ("time", np.full(nt, 250.0)),
                "shortwave_diffuse": ("time", np.full(nt, 120.0)),
                "longwave": ("time", np.full(nt, 290.0)),
                "humidity_specific": ("time", np.full(nt, 0.006)),
                "humidity_relative": ("time", np.full(nt, 0.70)),
                "wind_speed": ("time", np.full(nt, 2.5)),
                "pressure": ("time", np.full(nt, 95000.0)),
            },
            coords={"time": times},
        )
    else:
        unit_ids = [f"unit_{i:04d}" for i in range(n_units)]
        return xr.Dataset(
            {
                "temperature": (["time", "unit_id"], np.full((nt, n_units), 278.15)),
                "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.0002)),
                "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0002)),
                "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.0)),
                "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 250.0)),
                "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 120.0)),
                "longwave": (["time", "unit_id"], np.full((nt, n_units), 290.0)),
                "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.006)),
                "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.70)),
                "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 2.5)),
                "pressure": (["time", "unit_id"], np.full((nt, n_units), 95000.0)),
            },
            coords={"time": times, "unit_id": unit_ids},
        )


def make_units(n=1):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(8.0 + i * 0.1, 47.5, 1000.0 + i * 100),
            surface_type="open",
            area_m2=2500.0,
        )
        for i in range(n)
    ]


class TestCSVWriter:
    def test_write_creates_file(self, tmp_path):
        """Test that write creates a CSV file."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".csv"

    def test_file_naming(self, tmp_path):
        """Test CSV file naming convention."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("station_A", (8.0, 47.5, 1000.0))
        writer = CSVWriter()

        paths = writer.write(ds, [unit], tmp_path)

        assert paths[0].name == "forcing_station_A.csv"

    def test_metadata_header(self, tmp_path):
        """Test metadata header comments."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=True)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            first_line = f.readline()

        assert first_line.startswith("#")
        assert "TopoPyScale" in first_line

    def test_no_metadata_header(self, tmp_path):
        """Test output without metadata header."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=False)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            first_line = f.readline()

        # First line should be column headers, not comment
        assert not first_line.startswith("#")
        assert "time" in first_line

    def test_column_headers(self, tmp_path):
        """Test column headers are present."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=False)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            header_line = f.readline()

        assert "time" in header_line
        assert "temperature" in header_line

    def test_data_row_count(self, tmp_path):
        """Test correct number of data rows."""
        nt = 48
        ds = make_forcing_dataset(nt=nt)
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=False)

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        assert len(df) == nt

    def test_temperature_values(self, tmp_path):
        """Test temperature values are correct."""
        ds = make_forcing_dataset()
        ds["temperature"] = ds["temperature"] * 0 + 285.0
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=False)

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        np.testing.assert_allclose(df["temperature"].values, 285.0)

    def test_date_format(self, tmp_path):
        """Test custom date format."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(
            date_format="%Y-%m-%d %H:%M",
            include_metadata_header=False,
        )

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        assert "2020-03-15 00:00" == df["time"].iloc[0]

    def test_multiple_units(self, tmp_path):
        """Test writing multiple units creates multiple files."""
        ds = make_forcing_dataset(n_units=3)
        units = make_units(3)
        writer = CSVWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_single_file_long_format(self, tmp_path):
        """Test combined single file in long format."""
        ds = make_forcing_dataset(nt=12, n_units=3)
        units = make_units(3)
        writer = CSVWriter(single_file=True, wide_format=False)

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "forcing_combined.csv"

        # Long format should have unit_id column
        with open(paths[0]) as f:
            content = f.read()
        assert "unit_id" in content

    def test_single_file_wide_format(self, tmp_path):
        """Test combined single file in wide format."""
        ds = make_forcing_dataset(nt=12, n_units=2)
        units = make_units(2)
        writer = CSVWriter(single_file=True, wide_format=True)

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "forcing_wide.csv"

        # Wide format should have unit-specific columns
        with open(paths[0]) as f:
            content = f.read()
        assert "temperature_unit_0000" in content
        assert "temperature_unit_0001" in content

    def test_variable_subset(self, tmp_path):
        """Test outputting subset of variables."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(
            variables=["temperature", "precipitation"],
            include_metadata_header=False,
        )

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        assert "temperature" in df.columns
        assert "precipitation" in df.columns
        assert "wind_speed" not in df.columns

    def test_na_representation(self, tmp_path):
        """Test NaN values are represented correctly."""
        ds = make_forcing_dataset()
        ds["temperature"].values[0] = np.nan
        units = make_units(1)
        writer = CSVWriter(na_rep="NA", include_metadata_header=False)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        # First data row should have NA for temperature
        assert "NA" in content

    def test_float_format(self, tmp_path):
        """Test float format precision."""
        ds = make_forcing_dataset()
        ds["temperature"] = ds["temperature"] * 0 + 273.123456789
        units = make_units(1)
        writer = CSVWriter(float_format="%.3f", include_metadata_header=False)

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        # Should be rounded to 3 decimal places
        assert df["temperature"].iloc[0] == 273.123

    def test_read_helper(self, tmp_path):
        """Test static read helper method."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=True)

        paths = writer.write(ds, units, tmp_path)

        df = CSVWriter.read(paths[0])

        assert "temperature" in df.columns
        assert len(df) == 24

    def test_long_format_row_count(self, tmp_path):
        """Test long format has correct total rows."""
        nt = 12
        n_units = 3
        ds = make_forcing_dataset(nt=nt, n_units=n_units)
        units = make_units(n_units)
        writer = CSVWriter(
            single_file=True,
            wide_format=False,
            include_metadata_header=False,
        )

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        # Long format: nt * n_units rows
        assert len(df) == nt * n_units

    def test_wide_format_row_count(self, tmp_path):
        """Test wide format has correct row count."""
        nt = 12
        n_units = 3
        ds = make_forcing_dataset(nt=nt, n_units=n_units)
        units = make_units(n_units)
        writer = CSVWriter(
            single_file=True,
            wide_format=True,
            include_metadata_header=False,
        )

        paths = writer.write(ds, units, tmp_path)

        df = pd.read_csv(paths[0])
        # Wide format: nt rows
        assert len(df) == nt

    def test_metadata_includes_variables(self, tmp_path):
        """Test metadata header includes variable info."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = CSVWriter(include_metadata_header=True)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        # Should have variable descriptions
        assert "temperature:" in content.lower() or "air temperature" in content.lower()
