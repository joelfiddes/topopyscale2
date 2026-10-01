"""ecmwf-downloader: Download ECMWF climate data from multiple sources."""

from topopyscale2.inputs.nwp_downloader._version import __version__
from topopyscale2.inputs.nwp_downloader.bbox import BBox
from topopyscale2.inputs.nwp_downloader.derived import (
    compute_relative_humidity,
    compute_surface_geopotential,
)
from topopyscale2.inputs.nwp_downloader.reanalysis import ERA5Loader

# Forecast/IFS loaders resolve on first use (PEP 562): a reanalysis-only release
# does not ship them. name -> submodule.
_LAZY = {
    "ForecastLoader": "forecast",
    "IFSLoader": "ifs",
    "IFSForecastLoader": "ifs_forecast",  # Legacy, use forecast.ForecastLoader
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "__version__",
    "BBox",
    "ERA5Loader",
    "ForecastLoader",
    "IFSForecastLoader",  # Deprecated: use ForecastLoader
    "IFSLoader",
    "compute_relative_humidity",
    "compute_surface_geopotential",
]
