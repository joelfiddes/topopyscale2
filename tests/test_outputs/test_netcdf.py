"""Tests for CF-NetCDF output writer."""

import numpy as np
import pandas as pd
import xarray as xr

from topopyscale2.outputs.netcdf import CFNetCDFWriter
from topopyscale2.spatial.units import SpatialUnit


def make_forcing_dataset(nt=24, n_units=3):
    times = pd.date_range("2020-01-01", periods=nt, freq="h").values
    unit_ids = [f"unit_{i:04d}" for i in range(n_units)]

    ds = xr.Dataset(
        {
            "temperature": (["time", "unit_id"], np.full((nt, n_units), 275.0)),
            "precipitation": (["time", "unit_id"], np.full((nt, n_units), 0.001)),
            "rainfall": (["time", "unit_id"], np.full((nt, n_units), 0.0005)),
            "snowfall": (["time", "unit_id"], np.full((nt, n_units), 0.0005)),
            "shortwave_direct": (["time", "unit_id"], np.full((nt, n_units), 200.0)),
            "shortwave_diffuse": (["time", "unit_id"], np.full((nt, n_units), 100.0)),
            "longwave": (["time", "unit_id"], np.full((nt, n_units), 280.0)),
            "humidity_specific": (["time", "unit_id"], np.full((nt, n_units), 0.005)),
            "humidity_relative": (["time", "unit_id"], np.full((nt, n_units), 0.7)),
            "wind_speed": (["time", "unit_id"], np.full((nt, n_units), 3.0)),
            "pressure": (["time", "unit_id"], np.full((nt, n_units), 80000.0)),
        },
        coords={"time": times, "unit_id": unit_ids},
    )
    return ds


def make_units(n=3):
    return [
        SpatialUnit(
            id=f"unit_{i:04d}",
            centroid=(76.0 + i * 0.1, 42.0, 3000.0 + i * 100),
            surface_type="open",
            area_m2=900.0,
        )
        for i in range(n)
    ]


class TestCFNetCDFWriter:
    def test_write_creates_file(self, tmp_path):
        ds = make_forcing_dataset()
        units = make_units()
        writer = CFNetCDFWriter()

        paths = writer.write(ds, units, tmp_path)
        assert len(paths) == 1
        assert paths[0].exists()
        assert paths[0].suffix == ".nc"

    def test_cf_compliance(self, tmp_path):
        ds = make_forcing_dataset()
        units = make_units()
        writer = CFNetCDFWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        assert result.attrs["Conventions"] == "CF-1.8"
        assert "air_temperature" == result["temperature"].attrs["standard_name"]
        assert result["temperature"].attrs["units"] == "K"
        result.close()

    def test_round_trip(self, tmp_path):
        ds = make_forcing_dataset()
        units = make_units()
        writer = CFNetCDFWriter()

        paths = writer.write(ds, units, tmp_path)
        result = xr.open_dataset(paths[0])

        np.testing.assert_allclose(result["temperature"].values, 275.0)
        assert result.sizes["time"] == 24
        result.close()

    def test_compression(self, tmp_path):
        ds = make_forcing_dataset(nt=100, n_units=10)
        units = make_units(10)

        compressed = CFNetCDFWriter(compression=True)
        uncompressed = CFNetCDFWriter(compression=False)

        p_comp = compressed.write(ds, units, tmp_path / "comp")
        p_uncomp = uncompressed.write(ds, units, tmp_path / "uncomp")

        # Compressed file should be smaller (constant data compresses well)
        assert p_comp[0].stat().st_size <= p_uncomp[0].stat().st_size
