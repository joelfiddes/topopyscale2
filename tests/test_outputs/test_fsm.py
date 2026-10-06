"""Tests for FSM output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.fsm import FSMWriter
from topopyscale2.spatial.units import SpatialUnit


def make_single_unit_forcing(nt=24):
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    return xr.Dataset(
        {
            "temperature": ("time", np.full(nt, 270.0)),
            "precipitation": ("time", np.full(nt, 0.001)),
            "rainfall": ("time", np.full(nt, 0.0003)),
            "snowfall": ("time", np.full(nt, 0.0007)),
            "shortwave_direct": ("time", np.full(nt, 200.0)),
            "shortwave_diffuse": ("time", np.full(nt, 100.0)),
            "longwave": ("time", np.full(nt, 250.0)),
            "humidity_specific": ("time", np.full(nt, 0.003)),
            "humidity_relative": ("time", np.full(nt, 0.85)),
            "wind_speed": ("time", np.full(nt, 2.5)),
            "pressure": ("time", np.full(nt, 75000.0)),
        },
        coords={"time": times},
    )


class TestFSMWriter:
    def test_file_count(self, tmp_path):
        units = [
            SpatialUnit(f"u{i}", (76.0, 42.0, 3000.0 + i * 100))
            for i in range(5)
        ]
        ds = make_single_unit_forcing()
        writer = FSMWriter()

        paths = writer.write(ds, units, tmp_path)
        assert len(paths) == 5

    def test_file_naming(self, tmp_path):
        unit = SpatialUnit("test_001", (76.0, 42.0, 3000.0))
        ds = make_single_unit_forcing()
        writer = FSMWriter()

        paths = writer.write(ds, [unit], tmp_path)
        assert paths[0].name == "fsm_unit_test_001.txt"

    def test_column_count(self, tmp_path):
        """Each line should have 12 values."""
        unit = SpatialUnit("u0", (76.0, 42.0, 3000.0))
        ds = make_single_unit_forcing()
        writer = FSMWriter()

        paths = writer.write(ds, [unit], tmp_path)
        with open(paths[0]) as f:
            lines = f.readlines()

        assert len(lines) == 24
        # Check first line has expected number of fields
        fields = lines[0].split()
        assert len(fields) == 12

    def test_values_correct(self, tmp_path):
        """Check that written values match input."""
        unit = SpatialUnit("u0", (76.0, 42.0, 3000.0))
        ds = make_single_unit_forcing()
        writer = FSMWriter()

        paths = writer.write(ds, [unit], tmp_path)
        with open(paths[0]) as f:
            first_line = f.readline().split()

        # year, month, day, hour
        assert int(first_line[0]) == 2020
        assert int(first_line[1]) == 1
        assert int(first_line[2]) == 1
        assert int(first_line[3]) == 0

        # SW = direct + diffuse = 300
        assert float(first_line[4]) == 300.0

        # Ta = 270.0
        assert float(first_line[8]) == 270.0
