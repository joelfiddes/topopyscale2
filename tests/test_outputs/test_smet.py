"""Tests for SMET output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.smet import SMETWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=24, n_units=1):
    """Create test forcing dataset."""
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values

    if n_units == 1:
        return xr.Dataset(
            {
                "temperature": ("time", np.full(nt, 273.15)),
                "precipitation": ("time", np.full(nt, 0.001)),  # kg/m2
                "rainfall": ("time", np.full(nt, 0.0003)),
                "snowfall": ("time", np.full(nt, 0.0007)),
                "shortwave_direct": ("time", np.full(nt, 200.0)),
                "shortwave_diffuse": ("time", np.full(nt, 100.0)),
                "longwave": ("time", np.full(nt, 280.0)),
                "humidity_specific": ("time", np.full(nt, 0.005)),
                "humidity_relative": ("time", np.full(nt, 0.75)),
                "wind_speed": ("time", np.full(nt, 3.5)),
                "pressure": ("time", np.full(nt, 85000.0)),
            },
            coords={"time": times},
        )
    else:
        unit_ids = [f"unit_{i:04d}" for i in range(n_units)]
        return xr.Dataset(
            {
                "temperature": (["time", "unit_id"], np.full((nt, n_units), 273.15)),
                "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.001)),
                "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0003)),
                "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.0007)),
                "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 200.0)),
                "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 100.0)),
                "longwave": (["time", "unit_id"], np.full((nt, n_units), 280.0)),
                "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.005)),
                "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.75)),
                "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 3.5)),
                "pressure": (["time", "unit_id"], np.full((nt, n_units), 85000.0)),
            },
            coords={"time": times, "unit_id": unit_ids},
        )


def make_units(n=1):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(7.5 + i * 0.1, 46.5, 2500.0 + i * 100),
            surface_type="open",
            area_m2=900.0,
        )
        for i in range(n)
    ]


class TestSMETWriter:
    def test_write_creates_file(self, tmp_path):
        """Test that write creates a SMET file."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".smet"

    def test_file_naming(self, tmp_path):
        """Test SMET file naming convention."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("test_station", (7.5, 46.5, 2500.0))
        writer = SMETWriter()

        paths = writer.write(ds, [unit], tmp_path)

        assert paths[0].name == "test_station.smet"

    def test_smet_header_format(self, tmp_path):
        """Test SMET header structure."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        # Check header markers
        assert "SMET 1.1 ASCII" in content
        assert "[HEADER]" in content
        assert "[DATA]" in content
        assert "station_id" in content
        assert "latitude" in content
        assert "longitude" in content
        assert "altitude" in content
        assert "nodata" in content
        assert "fields" in content

    def test_smet_header_values(self, tmp_path):
        """Test SMET header contains correct values."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("test_001", (7.5, 46.5, 2500.0))
        writer = SMETWriter()

        paths = writer.write(ds, [unit], tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        assert "station_id       = test_001" in content
        assert "latitude         = 46.500000" in content
        assert "longitude        = 7.500000" in content
        assert "altitude         = 2500.0" in content

    def test_data_row_count(self, tmp_path):
        """Test correct number of data rows."""
        nt = 48
        ds = make_forcing_dataset(nt=nt)
        units = make_units(1)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        # Count lines after [DATA]
        data_start = next(i for i, line in enumerate(lines) if "[DATA]" in line) + 1
        data_lines = [l for l in lines[data_start:] if l.strip()]

        assert len(data_lines) == nt

    def test_data_column_count(self, tmp_path):
        """Test correct number of data columns."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = SMETWriter(include_precip_phase=True)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        # Get first data line
        data_start = content.index("[DATA]") + len("[DATA]\n")
        first_data_line = content[data_start:].split("\n")[0]
        columns = first_data_line.split()

        # timestamp + TA + RH + VW + ISWR + ILWR + PSUM + PSUM_PH = 8
        assert len(columns) == 8

    def test_temperature_values(self, tmp_path):
        """Test temperature values are written correctly."""
        ds = make_forcing_dataset()
        ds["temperature"] = ds["temperature"] * 0 + 280.5  # Set known value
        units = make_units(1)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        data_start = content.index("[DATA]") + len("[DATA]\n")
        first_data_line = content[data_start:].split("\n")[0]
        columns = first_data_line.split()

        # TA is second column (after timestamp)
        ta = float(columns[1])
        assert abs(ta - 280.5) < 0.01

    def test_shortwave_sum(self, tmp_path):
        """Test that shortwave is sum of direct and diffuse."""
        ds = make_forcing_dataset()
        ds["shortwave_direct"] = ds["shortwave_direct"] * 0 + 150.0
        ds["shortwave_diffuse"] = ds["shortwave_diffuse"] * 0 + 80.0
        units = make_units(1)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        data_start = content.index("[DATA]") + len("[DATA]\n")
        first_data_line = content[data_start:].split("\n")[0]
        columns = first_data_line.split()

        # ISWR is 5th column (index 4)
        iswr = float(columns[4])
        assert abs(iswr - 230.0) < 0.01

    def test_precip_phase_calculation(self, tmp_path):
        """Test precipitation phase is correctly calculated."""
        ds = make_forcing_dataset()
        ds["rainfall"] = ds["rainfall"] * 0 + 0.8  # 80% rain
        ds["snowfall"] = ds["snowfall"] * 0 + 0.2  # 20% snow
        units = make_units(1)
        writer = SMETWriter(include_precip_phase=True)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        data_start = content.index("[DATA]") + len("[DATA]\n")
        first_data_line = content[data_start:].split("\n")[0]
        columns = first_data_line.split()

        # PSUM_PH is 8th column (index 7)
        phase = float(columns[7])
        assert abs(phase - 0.8) < 0.01  # 0.8 = 80% rain

    def test_multiple_units(self, tmp_path):
        """Test writing multiple units creates multiple files."""
        ds = make_forcing_dataset(n_units=3)
        units = make_units(3)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_no_precip_phase(self, tmp_path):
        """Test output without precipitation phase column."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = SMETWriter(include_precip_phase=False)

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        # Check fields line does not have PSUM_PH
        assert "PSUM_PH" not in content

        data_start = content.index("[DATA]") + len("[DATA]\n")
        first_data_line = content[data_start:].split("\n")[0]
        columns = first_data_line.split()

        # 7 columns without phase
        assert len(columns) == 7

    def test_timestamp_format(self, tmp_path):
        """Test timestamp format in SMET file."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = SMETWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            content = f.read()

        data_start = content.index("[DATA]") + len("[DATA]\n")
        first_data_line = content[data_start:].split("\n")[0]
        timestamp = first_data_line.split()[0]

        # Check ISO format
        assert "2020-01-01T00:00:00" == timestamp
