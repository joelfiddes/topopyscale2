"""Export a finished run's forcing to the input formats of impact models (`tps2 export`).

Reads ``output/forcing.nc`` (or ``.zarr``) from a simulation directory and writes one of:

======== ==================================================== ==========================
format   for                                                  files
======== ==================================================== ==========================
smet     SNOWPACK, Alpine3D, MeteoIO (SMET 1.1)                one ``.smet`` per unit
fsm      FSM1 / FSM2 single-point met file                     one ``.txt`` per unit
fsm2     FSM2 multi-point met file (all units in one file)     one ``met_input.txt``
csv      spreadsheets, R, pandas                               one ``.csv`` per unit
crocus   Crocus / SURFEX (FORCING.nc, SAFRAN-style)            one ``.nc`` per unit
cryogrid CryoGrid (permafrost)                                 one ``.nc`` per unit
hbv      HBV-light style daily T, P, PET (Hamon)               one ``.txt`` per unit
======== ==================================================== ==========================

The terrain units come from the forcing file itself (its ``unit``, ``latitude``,
``longitude`` and ``elevation`` coordinates), so an export needs no other file from the run.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr

from topopyscale2.spatial.units import SpatialUnit

FORMATS = {
    "smet": ("smet", "SMETWriter", "SNOWPACK / Alpine3D / MeteoIO SMET files"),
    "fsm": ("fsm", "FSMWriter", "FSM single-point met files"),
    "fsm2": ("fsm2", "FSM2Writer", "FSM2 multi-point met file"),
    "csv": ("csv_writer", "CSVWriter", "CSV per unit"),
    "crocus": ("crocus", "CrocusWriter", "Crocus / SURFEX FORCING NetCDF"),
    "cryogrid": ("cryogrid", "CryoGridWriter", "CryoGrid forcing NetCDF"),
    "hbv": ("hbv", "HBVWriter", "HBV daily T, P, PET text files"),
}


def find_forcing(sim_dir: Path) -> Path:
    out = Path(sim_dir) / "output"
    for name in ("forcing.nc", "forcing.zarr"):
        if (out / name).exists():
            return out / name
    raise FileNotFoundError(f"no forcing in {out}: expected forcing.nc or forcing.zarr (run `tps2 run` first)")


def units_from_forcing(ds: xr.Dataset) -> list[SpatialUnit]:
    """One SpatialUnit per forcing unit, from the coordinates `tps2 run` writes."""
    dim = "unit" if "unit" in ds.dims else "unit_id"
    missing = [c for c in ("latitude", "longitude", "elevation") if c not in ds.coords]
    if missing:
        raise ValueError(f"forcing has no {missing} coordinates; re-run `tps2 run` to write them")
    return [
        SpatialUnit(id=str(u), centroid=(float(lon), float(lat), float(z)))
        for u, lat, lon, z in zip(ds[dim].values, ds["latitude"].values,
                                  ds["longitude"].values, ds["elevation"].values)
    ]


def export(sim_dir: Path, fmt: str, output_dir: Path | None = None) -> list[Path]:
    """Write the run in ``sim_dir`` as ``fmt``; return the files written."""
    import importlib

    if fmt not in FORMATS:
        raise ValueError(f"unknown export format '{fmt}'; choose from {', '.join(FORMATS)}")
    module, cls, _ = FORMATS[fmt]
    writer = getattr(importlib.import_module(f"topopyscale2.outputs.{module}"), cls)()

    sim_dir = Path(sim_dir)
    path = find_forcing(sim_dir)
    opener = xr.open_zarr if path.suffix == ".zarr" else xr.open_dataset
    with opener(path) as ds:
        ds = ds.load()
    units = units_from_forcing(ds)
    # The writers index units along `unit_id` (the FSM output convention); forcing uses `unit`.
    if "unit" in ds.dims:
        ds = ds.rename({"unit": "unit_id"})

    output_dir = Path(output_dir) if output_dir else sim_dir / "output" / fmt
    output_dir.mkdir(parents=True, exist_ok=True)
    written = writer.write(ds, units, output_dir)
    return list(written) if isinstance(written, (list, tuple)) else [written]
