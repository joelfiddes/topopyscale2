"""Tests for HBV output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.hbv import HBVWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=48, n_units=1):
    """Create test forcing dataset (hourly)."""
    times = pd.date_range("2020-06-01", periods=nt, freq="h").values

    if n_units == 1:
        return xr.Dataset(
            {
                "temperature": ("time", np.full(nt, 288.15)),  # 15C
                "precipitation": ("time", np.full(nt, 0.0005)),  # kg/m2 per hour
                "rainfall": ("time", np.full(nt, 0.0005)),
                "snowfall": ("time", np.full(nt, 0.0)),
                "shortwave_direct": ("time", np.full(nt, 300.0)),
                "shortwave_diffuse": ("time", np.full(nt, 150.0)),
                "longwave": ("time", np.full(nt, 320.0)),
                "humidity_specific": ("time", np.full(nt, 0.008)),
                "humidity_relative": ("time", np.full(nt, 0.65)),
                "wind_speed": ("time", np.full(nt, 2.0)),
                "pressure": ("time", np.full(nt, 90000.0)),
            },
            coords={"time": times},
        )
    else:
        unit_ids = [f"unit_{i:04d}" for i in range(n_units)]
        return xr.Dataset(
            {
                "temperature": (["time", "unit_id"], np.full((nt, n_units), 288.15)),
                "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.0005)),
                "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0005)),
                "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.0)),
                "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 300.0)),
                "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 150.0)),
                "longwave": (["time", "unit_id"], np.full((nt, n_units), 320.0)),
                "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.008)),
                "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.65)),
                "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 2.0)),
                "pressure": (["time", "unit_id"], np.full((nt, n_units), 90000.0)),
            },
            coords={"time": times, "unit_id": unit_ids},
        )


def make_units(n=1, lat=46.5):
    """Create test spatial units."""
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(7.5 + i * 0.1, lat, 1500.0 + i * 100),
            surface_type="open",
            area_m2=1e6,  # 1 km2
        )
        for i in range(n)
    ]


class TestHBVWriter:
    def test_write_creates_file(self, tmp_path):
        """Test that write creates an HBV file."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = HBVWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".txt"

    def test_file_naming(self, tmp_path):
        """Test HBV file naming convention."""
        ds = make_forcing_dataset()
        unit = SpatialUnit("catchment_A", (7.5, 46.5, 1500.0))
        writer = HBVWriter()

        paths = writer.write(ds, [unit], tmp_path)

        assert paths[0].name == "hbv_catchment_A.txt"

    def test_header_present(self, tmp_path):
        """Test header comments are present."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = HBVWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            first_line = f.readline()

        assert first_line.startswith("#")
        assert "TopoPyScale" in first_line

    def test_daily_aggregation(self, tmp_path):
        """Test that hourly data is aggregated to daily."""
        # 72 hours = 3 days
        ds = make_forcing_dataset(nt=72)
        units = make_units(1)
        writer = HBVWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        # Count non-comment lines
        data_lines = [l for l in lines if not l.startswith("#")]
        assert len(data_lines) == 3

    def test_temperature_conversion(self, tmp_path):
        """Test temperature is converted from Kelvin to Celsius."""
        ds = make_forcing_dataset()
        ds["temperature"] = ds["temperature"] * 0 + 293.15  # 20C
        units = make_units(1)
        writer = HBVWriter(temp_unit="celsius")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        temp = float(data_line.split()[1])

        assert abs(temp - 20.0) < 0.1

    def test_kelvin_output(self, tmp_path):
        """Test temperature can be output in Kelvin."""
        ds = make_forcing_dataset()
        ds["temperature"] = ds["temperature"] * 0 + 293.15
        units = make_units(1)
        writer = HBVWriter(temp_unit="kelvin")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        temp = float(data_line.split()[1])

        assert abs(temp - 293.15) < 0.1

    def test_precipitation_sum(self, tmp_path):
        """Hourly amounts are summed to a daily total in mm (1 kg m-2 is 1 mm of water).

        Regression: the writer multiplied by 1000, as if TPS2 precipitation were in metres,
        exporting 1000x too much (2026-10-06).
        """
        ds = make_forcing_dataset(nt=24)  # 24 hours = 1 day
        # 1 kg m-2 (= 1 mm) per hour * 24 hours = 24 mm
        ds["precipitation"] = ds["precipitation"] * 0 + 1.0
        units = make_units(1)
        writer = HBVWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        precip = float(data_line.split()[2])

        assert abs(precip - 24.0) < 0.1

    def test_column_count(self, tmp_path):
        """Test correct number of columns: Date T P PET."""
        ds = make_forcing_dataset()
        units = make_units(1)
        writer = HBVWriter()

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        columns = data_line.split()

        assert len(columns) == 4  # Date, T, P, PET

    def test_pet_computed(self, tmp_path):
        """Test PET is computed and positive in summer."""
        ds = make_forcing_dataset(nt=24)
        ds["temperature"] = ds["temperature"] * 0 + 298.15  # 25C (warm summer day)
        units = make_units(1, lat=46.5)  # Mid-latitude
        writer = HBVWriter(pet_method="hamon")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        pet = float(data_line.split()[3])

        # PET should be positive for warm day at mid-latitude
        assert pet > 0

    def test_pet_zero_for_cold(self, tmp_path):
        """Test PET is zero or near-zero for cold temperatures."""
        ds = make_forcing_dataset(nt=24)
        ds["temperature"] = ds["temperature"] * 0 + 263.15  # -10C
        units = make_units(1)
        writer = HBVWriter(pet_method="hamon")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        pet = float(data_line.split()[3])

        assert pet == 0.0  # Hamon PET is 0 for T <= 0

    def test_pet_none_method(self, tmp_path):
        """Test PET is zero when method is 'none'."""
        ds = make_forcing_dataset(nt=24)
        ds["temperature"] = ds["temperature"] * 0 + 298.15
        units = make_units(1)
        writer = HBVWriter(pet_method="none")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        pet = float(data_line.split()[3])

        assert pet == 0.0

    def test_multiple_units(self, tmp_path):
        """Test writing multiple units creates multiple files."""
        ds = make_forcing_dataset(n_units=3)
        units = make_units(3)
        writer = HBVWriter()

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 3
        for path in paths:
            assert path.exists()

    def test_single_file_mode(self, tmp_path):
        """Test combined single file output."""
        ds = make_forcing_dataset(n_units=3)
        units = make_units(3)
        writer = HBVWriter(single_file=True)

        paths = writer.write(ds, units, tmp_path)

        assert len(paths) == 1
        assert paths[0].name == "hbv_forcing.txt"

    def test_date_format_custom(self, tmp_path):
        """Test custom date format."""
        ds = make_forcing_dataset(nt=24)
        units = make_units(1)
        writer = HBVWriter(date_format="%Y-%m-%d")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        date_str = data_line.split()[0]

        assert date_str == "2020-06-01"

    def test_date_format_compact(self, tmp_path):
        """Test compact date format."""
        ds = make_forcing_dataset(nt=24)
        units = make_units(1)
        writer = HBVWriter(date_format="%Y%m%d")

        paths = writer.write(ds, units, tmp_path)

        with open(paths[0]) as f:
            lines = f.readlines()

        data_line = [l for l in lines if not l.startswith("#")][0]
        date_str = data_line.split()[0]

        assert date_str == "20200601"
