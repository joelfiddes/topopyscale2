"""Optimal chunk computation for Zarr storage.

Provides utilities for computing chunk sizes optimized for different
data access patterns and NWP data sources.
"""

from typing import Optional

import numpy as np


def compute_optimal_chunks(
    shape: tuple[int, ...],
    dtype: np.dtype,
    target_mb: float = 100,
    dim_names: Optional[tuple[str, ...]] = None,
) -> dict[str, int]:
    """Compute optimal chunk sizes for given array shape and dtype.

    This function computes chunk sizes that result in approximately
    the target chunk size in memory, balancing between dimensions
    based on typical access patterns.

    Parameters
    ----------
    shape : tuple of int
        Array shape.
    dtype : np.dtype
        Array data type.
    target_mb : float
        Target chunk size in megabytes (default: 100 MB).
    dim_names : tuple of str, optional
        Dimension names. If not provided, uses dim_0, dim_1, etc.

    Returns
    -------
    dict
        Chunk sizes keyed by dimension name.

    Examples
    --------
    >>> compute_optimal_chunks((8760, 1000), np.float64, target_mb=100)
    {'dim_0': 1752, 'dim_1': 1000}

    >>> compute_optimal_chunks((8760, 100, 200), np.float32, target_mb=50,
    ...                        dim_names=("time", "lat", "lon"))
    {'time': 876, 'lat': 100, 'lon': 200}
    """
    ndim = len(shape)
    if dim_names is None:
        dim_names = tuple(f"dim_{i}" for i in range(ndim))

    if len(dim_names) != ndim:
        raise ValueError(f"dim_names length ({len(dim_names)}) must match shape ({ndim})")

    # Convert dtype to numpy dtype if string
    dtype = np.dtype(dtype)
    bytes_per_element = dtype.itemsize

    # Target bytes
    target_bytes = target_mb * 1024 * 1024

    # Total array size
    total_elements = np.prod(shape)
    total_bytes = total_elements * bytes_per_element

    # If array fits in target, use full shape
    if total_bytes <= target_bytes:
        return dict(zip(dim_names, shape))

    # Compute chunk sizes
    # Strategy: For time-series data, prefer larger time chunks
    # For spatial data, keep spatial dims together

    # Start with full dimensions
    chunks = list(shape)

    # Calculate initial chunk bytes
    def chunk_bytes():
        return np.prod(chunks) * bytes_per_element

    # Reduce dimensions iteratively, starting with the largest
    while chunk_bytes() > target_bytes:
        # Find the largest dimension that can be reduced
        max_idx = None
        max_val = 0
        for i, c in enumerate(chunks):
            if c > 1 and c > max_val:
                max_idx = i
                max_val = c

        if max_idx is None:
            break

        # Reduce by factor of 2
        chunks[max_idx] = max(1, chunks[max_idx] // 2)

    # Ensure minimum chunk of 1 on all dimensions
    chunks = [max(1, c) for c in chunks]

    return dict(zip(dim_names, chunks))


# Preset chunk configurations for different NWP sources
_NWP_CHUNK_PRESETS = {
    # ERA5 - hourly, global 0.25 degree
    "era5": {
        "time": 24,  # 1 day of hourly data
        "latitude": 100,
        "longitude": 200,
        "level": -1,  # Full pressure levels
    },
    # ERA5-Land - hourly, 0.1 degree
    "era5_land": {
        "time": 24,
        "latitude": 200,
        "longitude": 400,
    },
    # HRES (high resolution forecast)
    "hres": {
        "time": 6,  # 6-hourly steps
        "latitude": 150,
        "longitude": 300,
        "level": -1,
    },
    # GFS (NCEP)
    "gfs": {
        "time": 8,  # 3-hourly
        "latitude": 100,
        "longitude": 200,
        "level": -1,
    },
    # MERRA-2
    "merra2": {
        "time": 24,
        "lat": 90,
        "lon": 144,
        "lev": -1,
    },
    # Downscaled output (point/cluster data)
    "downscaled": {
        "time": 168,  # 1 week of hourly
        "unit_id": -1,  # All spatial units
    },
    # Default
    "default": {
        "time": 24,
    },
}


def chunks_for_nwp(source_type: str) -> dict[str, int]:
    """Get preset chunk sizes for a given NWP source type.

    Parameters
    ----------
    source_type : str
        NWP source type: "era5", "era5_land", "hres", "gfs",
        "merra2", "downscaled", or "default".

    Returns
    -------
    dict
        Chunk sizes keyed by dimension name.
        -1 indicates full dimension size.

    Examples
    --------
    >>> chunks_for_nwp("era5")
    {'time': 24, 'latitude': 100, 'longitude': 200, 'level': -1}

    >>> chunks_for_nwp("downscaled")
    {'time': 168, 'unit_id': -1}
    """
    source_type = source_type.lower().replace("-", "_")

    if source_type not in _NWP_CHUNK_PRESETS:
        # Try common variations
        if source_type.startswith("era5"):
            if "land" in source_type:
                source_type = "era5_land"
            else:
                source_type = "era5"
        else:
            source_type = "default"

    return _NWP_CHUNK_PRESETS[source_type].copy()


def resolve_chunks(
    chunks: dict[str, int],
    shape_dict: dict[str, int],
) -> dict[str, int]:
    """Resolve chunk specification against actual dimension sizes.

    Handles -1 (full dimension) and ensures chunks don't exceed dim size.

    Parameters
    ----------
    chunks : dict
        Chunk specification (may contain -1 for full dimension).
    shape_dict : dict
        Actual dimension sizes.

    Returns
    -------
    dict
        Resolved chunk sizes.

    Examples
    --------
    >>> resolve_chunks({"time": 24, "unit_id": -1}, {"time": 100, "unit_id": 50})
    {'time': 24, 'unit_id': 50}
    """
    resolved = {}
    for dim, size in shape_dict.items():
        if dim in chunks:
            chunk = chunks[dim]
            if chunk == -1:
                resolved[dim] = size
            else:
                resolved[dim] = min(chunk, size)
        else:
            # Use full dimension if not specified
            resolved[dim] = size
    return resolved


def compressor_encoding(compressor) -> dict:
    """xarray ``to_zarr`` encoding entry for one compressor, for the installed zarr.

    Zarr 3 stores take ``compressors`` (a sequence of codecs); recent xarray rejects
    the zarr-2 ``compressor`` key for them outright. Zarr 2 stores take ``compressor``.
    """
    try:
        import zarr
        major = int(zarr.__version__.split(".")[0])
    except ImportError:
        major = 0
    return {"compressors": (compressor,)} if major >= 3 else {"compressor": compressor}
