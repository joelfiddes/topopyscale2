"""Output writers for downscaled forcing data.

Names are imported lazily (PEP 562): importing this package loads no writer, so an
installation without the snow-model or dashboard modules can still use the forcing
writers. Submodule imports (``from topopyscale2.outputs import netcdf``) are unaffected.
"""

import importlib

# public name -> submodule that defines it
_EXPORTS = {
    "CFNetCDFWriter": "netcdf",
    "ZarrWriter": "zarr_writer",
    "CSVWriter": "csv_writer",
    "SMETWriter": "smet",
    "RasterMapper": "raster_mapper",
    "hillshade": "plots",
    "plot_raster_map": "plots",
    "plot_model_maps": "plots",
    "FSMWriter": "fsm",
    "FSM2Writer": "fsm2",
    "HBVWriter": "hbv",
    "CryoGridWriter": "cryogrid",
    "CrocusWriter": "crocus",
    "generate_dashboard": "dashboard",
    "update_live_dashboard": "dashboard",
}

# format name -> writer class name
_FORMATS = {
    "netcdf": "CFNetCDFWriter",
    "fsm": "FSMWriter",
    "fsm2": "FSM2Writer",
    "smet": "SMETWriter",
    "snowpack": "SMETWriter",  # alias
    "hbv": "HBVWriter",
    "cryogrid": "CryoGridWriter",
    "crocus": "CrocusWriter",
    "safran": "CrocusWriter",  # alias
    "zarr": "ZarrWriter",
    "csv": "CSVWriter",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value  # cache: later lookups skip __getattr__
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


def get_writer(format: str, **kwargs):
    """Factory function to get an output writer by format name.

    Parameters
    ----------
    format : str
        Output format name. Available formats:
        - "netcdf": CF-compliant NetCDF (CFNetCDFWriter)
        - "fsm": FSM text format (FSMWriter)
        - "fsm2": FSM2 multi-point text format (FSM2Writer)
        - "smet": SNOWPACK SMET format (SMETWriter)
        - "hbv": HBV daily format (HBVWriter)
        - "cryogrid": CryoGrid NetCDF format (CryoGridWriter)
        - "crocus": Crocus/SAFRAN NetCDF format (CrocusWriter)
        - "zarr": Zarr chunked format (ZarrWriter)
        - "csv": CSV tabular format (CSVWriter)
    **kwargs
        Additional arguments passed to writer constructor.

    Returns
    -------
    OutputWriter
        Initialized output writer.

    Raises
    ------
    ValueError
        If format is not recognized.
    """
    if format not in _FORMATS:
        raise ValueError(f"Unknown output format '{format}'. Available: {list(_FORMATS)}")
    try:
        writer = __getattr__(_FORMATS[format])
    except ModuleNotFoundError as e:
        raise ValueError(
            f"Output format '{format}' is not available in this installation ({e})."
        ) from e
    return writer(**kwargs)


__all__ = [*_EXPORTS, "get_writer"]
