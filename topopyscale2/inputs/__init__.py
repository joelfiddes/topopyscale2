"""Multi-NWP input sources for TopoPyScale 2.0.

This module provides a unified interface for fetching meteorological forcing
data from multiple NWP sources (ERA5, HRES, ICON, COSMO, GFS, custom).

Example usage
-------------
>>> from topopyscale2.inputs import get_source, create_blender
>>> from topopyscale2._optional import not_in_release
from topopyscale2.config.schema import InputConfig, NWPSourceConfig, feature_installed
>>>
>>> # Single source
>>> config = InputConfig()
>>> era5 = get_source("era5", config)
>>> ds_surf, ds_plev = era5.fetch(bbox, time_range)
>>>
>>> # Multiple sources with blending
>>> config.blending.enabled = True
>>> config.sources = [
...     NWPSourceConfig(name="hres", type="hres", priority=1),
...     NWPSourceConfig(name="era5", type="era5", priority=2),
... ]
>>> blender = create_blender(config)
>>> ds_surf, ds_plev, source_mask = blender.fetch_blended(bbox, time_range)
"""

import importlib
from typing import TYPE_CHECKING, Optional, Union

from topopyscale2._optional import not_in_release
from topopyscale2.config.schema import InputConfig, NWPSourceConfig, feature_installed
from topopyscale2.inputs.base import DEFAULT_VARIABLE_MAPPINGS, STANDARD_VARIABLES, MeteoSource
from topopyscale2.inputs.custom import CustomSource, validate_variable_mapping
from topopyscale2.inputs.era5 import ERA5Source, merge_era5_expver
from topopyscale2.inputs.resolution import (
    adaptive_lapse_rate_weight,
    adaptive_precipitation_gradient_weight,
    adaptive_radiation_partitioning_weight,
    adaptive_wind_correction_weight,
    compute_correction_weights,
    estimate_effective_resolution,
    resolution_scaling_factor,
)
from topopyscale2.inputs.units import (
    convert_era5,
    convert_era5_pressure,
    convert_era5_surface,
    detect_time_step,
)

if TYPE_CHECKING:
    from topopyscale2.inputs.blending import SourceBlender

# Source type registry: name -> (module, class). Resolved on use, so a release that
# ships only ERA5 + custom sources imports none of the others.
_SOURCE_TYPES = {
    "era5": ("topopyscale2.inputs.era5", "ERA5Source"),
    "hres": ("topopyscale2.inputs.hres", "HRESSource"),
    "icon": ("topopyscale2.inputs.hres", "ICONSource"),
    "cosmo": ("topopyscale2.inputs.hres", "COSMOSource"),
    "gfs": ("topopyscale2.inputs.hres", "GFSSource"),
    "custom": ("topopyscale2.inputs.custom", "CustomSource"),
}

# Optional names re-exported lazily (PEP 562): name -> module.
_LAZY = {
    "SourceBlender": "topopyscale2.inputs.blending",
    "create_blender_from_config": "topopyscale2.inputs.blending",
    "HRESSource": "topopyscale2.inputs.hres",
    "ICONSource": "topopyscale2.inputs.hres",
    "COSMOSource": "topopyscale2.inputs.hres",
    "GFSSource": "topopyscale2.inputs.hres",
}


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(_LAZY[name]), name)
    globals()[name] = value
    return value


def _source_class(source_type: str):
    module, cls = _SOURCE_TYPES[source_type]
    try:
        return getattr(importlib.import_module(module), cls)
    except ModuleNotFoundError as e:
        raise not_in_release(f"Source type '{source_type}'", e, module) from e


def get_source(
    source_type: str,
    config: InputConfig,
    source_config: Optional[NWPSourceConfig] = None,
) -> MeteoSource:
    """Factory function to create a data source instance.

    Parameters
    ----------
    source_type : str
        Type of source: "era5", "hres", "icon", "cosmo", "gfs", or "custom".
    config : InputConfig
        General input configuration.
    source_config : NWPSourceConfig, optional
        Source-specific configuration (path, variable mapping, etc.).
        Required for "custom" source type.

    Returns
    -------
    MeteoSource
        Instantiated data source.

    Raises
    ------
    ValueError
        If source_type is unknown.

    Examples
    --------
    >>> config = InputConfig()
    >>> era5 = get_source("era5", config)
    >>> ds_surf, ds_plev = era5.fetch((75.0, 41.0, 77.0, 43.0), ["2020-01-01", "2020-01-02"])
    """
    if source_type not in _SOURCE_TYPES:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Available types: {list(_SOURCE_TYPES.keys())}"
        )

    source_class = _source_class(source_type)

    # ERA5Source has a simpler constructor
    if source_type == "era5":
        return source_class(config)

    # Other sources take optional source_config
    return source_class(config, source_config)


def create_blender(
    config: InputConfig,
    sources: Optional[dict[str, MeteoSource]] = None,
) -> Optional["SourceBlender"]:
    """Create a SourceBlender from configuration.

    Parameters
    ----------
    config : InputConfig
        Input configuration with sources and blending settings.
    sources : dict[str, MeteoSource], optional
        Pre-instantiated source objects. If not provided, sources
        will be instantiated using get_source.

    Returns
    -------
    SourceBlender or None
        Configured blender, or None if blending is disabled.

    Examples
    --------
    >>> config = InputConfig()
    >>> config.blending.enabled = True
    >>> config.sources = [
    ...     NWPSourceConfig(name="hres", type="hres", priority=1),
    ...     NWPSourceConfig(name="era5", type="era5", priority=2),
    ... ]
    >>> blender = create_blender(config)
    """
    try:
        from topopyscale2.inputs.blending import create_blender_from_config
    except ModuleNotFoundError as e:
        raise not_in_release("Multi-source blending", e, "topopyscale2.inputs.blending") from e
    return create_blender_from_config(config, sources)


def list_available_sources() -> list[str]:
    """List available source types.

    Returns
    -------
    list[str]
        Names of available source types.
    """
    return [name for name, (module, _) in _SOURCE_TYPES.items() if feature_installed(module)]


def get_source_resolution(source_type: str) -> float:
    """Get the default resolution for a source type.

    Parameters
    ----------
    source_type : str
        Type of source.

    Returns
    -------
    float
        Default resolution in meters.
    """
    default_resolutions = {
        "era5": 31000.0,
        "hres": 9000.0,
        "icon": 6500.0,
        "cosmo": 1000.0,
        "gfs": 28000.0,
        "custom": 10000.0,
    }
    return default_resolutions.get(source_type, 10000.0)


__all__ = [
    # Factory functions
    "get_source",
    "create_blender",
    "list_available_sources",
    "get_source_resolution",
    # Source classes
    "ERA5Source",
    "merge_era5_expver",
    "HRESSource",
    "ICONSource",
    "COSMOSource",
    "GFSSource",
    "CustomSource",
    # Blending
    "SourceBlender",
    # Protocol and constants
    "MeteoSource",
    "STANDARD_VARIABLES",
    "DEFAULT_VARIABLE_MAPPINGS",
    # Resolution utilities
    "resolution_scaling_factor",
    "compute_correction_weights",
    "estimate_effective_resolution",
    "adaptive_lapse_rate_weight",
    "adaptive_radiation_partitioning_weight",
    "adaptive_wind_correction_weight",
    "adaptive_precipitation_gradient_weight",
    # Validation
    "validate_variable_mapping",
    # Unit conversion
    "convert_era5",
    "convert_era5_surface",
    "convert_era5_pressure",
    "detect_time_step",
]
