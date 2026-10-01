"""Format conversion utilities for NetCDF and Zarr."""

from pathlib import Path
from typing import Optional, Union

import xarray as xr

try:
    import zarr
    ZARR_VERSION = int(zarr.__version__.split(".")[0])
except ImportError:
    ZARR_VERSION = 0

from topopyscale2.storage.chunking import (
    compressor_encoding,
    compute_optimal_chunks,
    resolve_chunks,
)


def netcdf_to_zarr(
    nc_path: Union[str, Path],
    zarr_path: Union[str, Path],
    chunks: Optional[dict[str, int]] = None,
    compression: Optional[str] = "zstd",
    compression_level: int = 3,
    consolidated: bool = True,
) -> Path:
    """Convert NetCDF file to Zarr store with optional rechunking.

    Parameters
    ----------
    nc_path : str or Path
        Path to input NetCDF file.
    zarr_path : str or Path
        Path for output Zarr store.
    chunks : dict, optional
        Target chunk sizes. If None, computes optimal chunks.
        Use -1 for full dimension size.
    compression : str, optional
        Compression algorithm: "zstd", "lz4", "gzip", "zlib", or None.
        Default: "zstd".
    compression_level : int
        Compression level. Default: 3.
    consolidated : bool
        Whether to consolidate metadata. Default: True.

    Returns
    -------
    Path
        Path to created Zarr store.

    Examples
    --------
    >>> netcdf_to_zarr("era5.nc", "era5.zarr", chunks={"time": 24})
    PosixPath('era5.zarr')

    >>> netcdf_to_zarr("data.nc", "data.zarr")  # Auto-compute chunks
    PosixPath('data.zarr')
    """
    nc_path = Path(nc_path)
    zarr_path = Path(zarr_path)

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    # Open NetCDF
    ds = xr.open_dataset(nc_path)

    try:
        # Determine chunks
        if chunks is None:
            # Auto-compute based on first data variable
            for var in ds.data_vars:
                da = ds[var]
                shape = da.shape
                dtype = da.dtype
                dim_names = da.dims
                chunks = compute_optimal_chunks(
                    shape, dtype, target_mb=100, dim_names=dim_names
                )
                break
            else:
                chunks = {}

        # Resolve chunk sizes
        shape_dict = dict(ds.sizes)
        resolved_chunks = resolve_chunks(chunks, shape_dict)

        # Chunk dataset
        ds = ds.chunk(resolved_chunks)

        # Get compressor
        compressor = _get_compressor(compression, compression_level)

        # Build encoding
        encoding = {}
        if compressor is not None:
            for var in ds.data_vars:
                encoding[var] = compressor_encoding(compressor)

        # Create parent directory
        zarr_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Zarr
        ds.to_zarr(
            zarr_path,
            mode="w",
            encoding=encoding if encoding else None,
            consolidated=consolidated,
        )

    finally:
        ds.close()

    return zarr_path


def zarr_to_netcdf(
    zarr_path: Union[str, Path],
    nc_path: Union[str, Path],
    engine: str = "netcdf4",
    unlimited_dims: Optional[list[str]] = None,
) -> Path:
    """Convert Zarr store to NetCDF file.

    Parameters
    ----------
    zarr_path : str or Path
        Path to input Zarr store.
    nc_path : str or Path
        Path for output NetCDF file.
    engine : str
        NetCDF engine: "netcdf4" or "scipy". Default: "netcdf4".
    unlimited_dims : list, optional
        Dimensions to make unlimited (usually ["time"]).

    Returns
    -------
    Path
        Path to created NetCDF file.

    Examples
    --------
    >>> zarr_to_netcdf("data.zarr", "data.nc")
    PosixPath('data.nc')

    >>> zarr_to_netcdf("data.zarr", "data.nc", unlimited_dims=["time"])
    PosixPath('data.nc')
    """
    zarr_path = Path(zarr_path)
    nc_path = Path(nc_path)

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    # Try to open with consolidated metadata first
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(zarr_path, consolidated=False)

    try:
        # Load into memory for NetCDF write (Zarr may be lazily loaded)
        ds = ds.compute()

        # Create parent directory
        nc_path.parent.mkdir(parents=True, exist_ok=True)

        # Build encoding
        encoding = {}
        if unlimited_dims:
            for var in list(ds.data_vars) + list(ds.coords):
                if var in ds:
                    var_dims = ds[var].dims
                    for dim in unlimited_dims:
                        if dim in var_dims:
                            encoding[var] = encoding.get(var, {})
                            # Unlimited dim will be set via unlimited_dims param
                            break

        # Write to NetCDF
        ds.to_netcdf(
            nc_path,
            engine=engine,
            unlimited_dims=unlimited_dims,
            encoding=encoding if encoding else None,
        )

    finally:
        ds.close()

    return nc_path


def _get_compressor(compression: Optional[str], level: int = 3):
    """Get compressor based on zarr version.

    Parameters
    ----------
    compression : str or None
        Compression algorithm.
    level : int
        Compression level.

    Returns
    -------
    Compressor or None
    """
    if compression is None:
        return None

    # Zarr 3.x uses zarr.codecs
    if ZARR_VERSION >= 3:
        try:
            from zarr.codecs import GzipCodec, ZstdCodec
        except ImportError:
            import warnings
            warnings.warn(
                "zarr.codecs not available. Compression disabled."
            )
            return None

        compressors = {
            "zstd": lambda: ZstdCodec(level=level),
            "gzip": lambda: GzipCodec(level=level),
            "lz4": lambda: None,  # May not be available
            "zlib": lambda: GzipCodec(level=level),
        }

        if compression not in compressors:
            raise ValueError(
                f"Unknown compression: {compression}. "
                f"Supported: {list(compressors.keys())}"
            )

        return compressors[compression]()

    # Zarr 2.x uses numcodecs
    try:
        import numcodecs
    except ImportError:
        import warnings
        warnings.warn(
            "numcodecs is not installed. Compression disabled. "
            "Install with: pip install numcodecs"
        )
        return None

    compressors = {
        "zstd": lambda: numcodecs.Zstd(level=level),
        "lz4": lambda: numcodecs.LZ4(acceleration=level),
        "gzip": lambda: numcodecs.GZip(level=level),
        "zlib": lambda: numcodecs.Zlib(level=level),
    }

    if compression not in compressors:
        raise ValueError(
            f"Unknown compression: {compression}. "
            f"Supported: {list(compressors.keys())}"
        )

    return compressors[compression]()


def get_zarr_info(zarr_path: Union[str, Path]) -> dict:
    """Get information about a Zarr store.

    Parameters
    ----------
    zarr_path : str or Path
        Path to Zarr store.

    Returns
    -------
    dict
        Store information including dimensions, variables, size.
    """
    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr store not found: {zarr_path}")

    # Try to open
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
    except Exception:
        ds = xr.open_zarr(zarr_path, consolidated=False)

    # Compute size
    total_bytes = 0
    for item in zarr_path.rglob("*"):
        if item.is_file():
            total_bytes += item.stat().st_size

    # Get chunk info
    chunk_info = {}
    for var in ds.data_vars:
        da = ds[var]
        if da.chunks is not None:
            # Get first chunk size for each dimension
            chunk_info[var] = {
                dim: chunks[0] if chunks else da.sizes[dim]
                for dim, chunks in zip(da.dims, da.chunks)
            }

    info = {
        "path": str(zarr_path),
        "size_bytes": total_bytes,
        "size_mb": total_bytes / (1024 * 1024),
        "dimensions": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coordinates": list(ds.coords),
        "chunks": chunk_info,
        "attrs": dict(ds.attrs),
        "consolidated": (zarr_path / ".zmetadata").exists(),
    }

    ds.close()
    return info


def get_netcdf_info(nc_path: Union[str, Path]) -> dict:
    """Get information about a NetCDF file.

    Parameters
    ----------
    nc_path : str or Path
        Path to NetCDF file.

    Returns
    -------
    dict
        File information including dimensions, variables, size.
    """
    nc_path = Path(nc_path)

    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")

    ds = xr.open_dataset(nc_path)

    info = {
        "path": str(nc_path),
        "size_bytes": nc_path.stat().st_size,
        "size_mb": nc_path.stat().st_size / (1024 * 1024),
        "dimensions": dict(ds.sizes),
        "variables": list(ds.data_vars),
        "coordinates": list(ds.coords),
        "attrs": dict(ds.attrs),
    }

    ds.close()
    return info
