"""Zarr-based implementation of ForcingStore protocol."""

import datetime
from pathlib import Path
from typing import Optional, Union

import xarray as xr

try:
    import zarr
    ZARR_VERSION = int(zarr.__version__.split(".")[0])
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
    ZARR_VERSION = 0

from topopyscale2.storage.chunking import compressor_encoding, resolve_chunks


class ZarrStore:
    """Zarr-based storage backend implementing ForcingStore protocol.

    Provides efficient storage for meteorological forcing data with:
    - Configurable chunking for optimal access patterns
    - Compression (zstd default)
    - Consolidated metadata for faster opens
    - Append mode for time-series data

    Parameters
    ----------
    chunks : dict, optional
        Chunk sizes per dimension. Use -1 for full dimension.
        Defaults to {"time": 24}.
    compression : str, optional
        Compression algorithm: "zstd", "lz4", "gzip", "zlib", or None.
        Default: "zstd".
    compression_level : int
        Compression level (meaning varies by algorithm). Default: 3.
    consolidated : bool
        Whether to consolidate metadata. Default: True.

    Examples
    --------
    >>> store = ZarrStore(chunks={"time": 168}, compression="zstd")
    >>> store.write(ds, "/data/forcing.zarr")
    >>> ds = store.read("/data/forcing.zarr")
    """

    def __init__(
        self,
        chunks: Optional[dict[str, int]] = None,
        compression: Optional[str] = "zstd",
        compression_level: int = 3,
        consolidated: bool = True,
    ):
        if not HAS_ZARR:
            raise ImportError(
                "zarr is required for ZarrStore. "
                "Install with: pip install zarr"
            )

        self.chunks = chunks or {"time": 24}
        self.compression = compression
        self.compression_level = compression_level
        self.consolidated = consolidated

    def _get_compressor(self):
        """Get compressor based on settings and zarr version.

        For zarr 3.x, uses zarr.codecs.
        For zarr 2.x, uses numcodecs.
        """
        if self.compression is None:
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
                "zstd": lambda: ZstdCodec(level=self.compression_level),
                "gzip": lambda: GzipCodec(level=self.compression_level),
                # lz4 and zlib may not be available in zarr 3 codecs
                "lz4": lambda: None,
                "zlib": lambda: GzipCodec(level=self.compression_level),
            }

            if self.compression not in compressors:
                raise ValueError(
                    f"Unknown compression: {self.compression}. "
                    f"Supported: {list(compressors.keys())}"
                )

            return compressors[self.compression]()

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
            "zstd": lambda: numcodecs.Zstd(level=self.compression_level),
            "lz4": lambda: numcodecs.LZ4(acceleration=self.compression_level),
            "gzip": lambda: numcodecs.GZip(level=self.compression_level),
            "zlib": lambda: numcodecs.Zlib(level=self.compression_level),
        }

        if self.compression not in compressors:
            raise ValueError(
                f"Unknown compression: {self.compression}. "
                f"Supported: {list(compressors.keys())}"
            )

        return compressors[self.compression]()

    def _resolve_chunks(self, ds: xr.Dataset) -> dict[str, int]:
        """Resolve chunk sizes for dataset dimensions."""
        shape_dict = dict(ds.sizes)
        return resolve_chunks(self.chunks, shape_dict)

    def _add_global_attrs(self, ds: xr.Dataset) -> xr.Dataset:
        """Add global metadata attributes."""
        ds = ds.copy()
        ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": "TopoPyScale 2.0 meteorological forcing",
            "institution": "TopoPyScale",
            "history": (
                f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()} "
                "by TopoPyScale 2.0"
            ),
            "format": "Zarr",
            "compression": str(self.compression),
        })
        return ds

    def write(self, data: xr.Dataset, path: Union[str, Path]) -> Path:
        """Write dataset to Zarr store.

        Parameters
        ----------
        data : xr.Dataset
            Dataset to write.
        path : str or Path
            Output path (directory for Zarr store).

        Returns
        -------
        Path
            Path to written store.
        """
        path = Path(path)

        # Add metadata
        ds = self._add_global_attrs(data)

        # Resolve and apply chunks
        chunks = self._resolve_chunks(ds)
        ds = ds.chunk(chunks)

        # Build encoding with compression
        compressor = self._get_compressor()
        encoding = {}
        if compressor is not None:
            for var in ds.data_vars:
                encoding[var] = compressor_encoding(compressor)

        # Create parent directories
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to Zarr
        ds.to_zarr(
            path,
            mode="w",
            encoding=encoding if encoding else None,
            consolidated=self.consolidated,
        )

        return path

    def read(self, path: Union[str, Path], **kwargs) -> xr.Dataset:
        """Read dataset from Zarr store.

        Parameters
        ----------
        path : str or Path
            Path to Zarr store.
        **kwargs
            Additional arguments passed to xr.open_zarr.

        Returns
        -------
        xr.Dataset
            Loaded dataset.
        """
        path = Path(path)

        # Set defaults for kwargs
        if "consolidated" not in kwargs:
            # Try consolidated first, fall back if not available
            try:
                return xr.open_zarr(path, consolidated=True, **kwargs)
            except Exception:
                return xr.open_zarr(path, consolidated=False, **kwargs)

        return xr.open_zarr(path, **kwargs)

    def append(
        self,
        data: xr.Dataset,
        path: Union[str, Path],
        dim: str = "time",
    ) -> Path:
        """Append data along a dimension to existing Zarr store.

        Parameters
        ----------
        data : xr.Dataset
            Data to append.
        path : str or Path
            Path to existing Zarr store.
        dim : str
            Dimension to append along (default: "time").

        Returns
        -------
        Path
            Path to updated store.

        Raises
        ------
        FileNotFoundError
            If store does not exist at path.
        """
        path = Path(path)

        if not self.exists(path):
            raise FileNotFoundError(f"Zarr store not found at {path}")

        # Resolve chunks
        chunks = self._resolve_chunks(data)
        ds = data.chunk(chunks)

        # Append along dimension
        ds.to_zarr(
            path,
            mode="a",
            append_dim=dim,
            consolidated=self.consolidated,
        )

        return path

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if Zarr store exists at path.

        Parameters
        ----------
        path : str or Path
            Path to check.

        Returns
        -------
        bool
            True if store exists and is valid Zarr.
        """
        path = Path(path)

        if not path.exists():
            return False

        # Zarr 2.x markers
        if (path / ".zgroup").exists() or (path / ".zarray").exists():
            return True

        # Check for .zattrs (consolidated metadata marker)
        if (path / ".zattrs").exists():
            return True

        # Zarr 3.x markers
        if (path / "zarr.json").exists():
            return True

        # Try opening to verify
        try:
            zarr.open(str(path), mode="r")
            return True
        except Exception:
            return False

    def get_info(self, path: Union[str, Path]) -> dict:
        """Get information about a Zarr store.

        Parameters
        ----------
        path : str or Path
            Path to Zarr store.

        Returns
        -------
        dict
            Store information including size, variables, dimensions.
        """
        path = Path(path)

        if not self.exists(path):
            raise FileNotFoundError(f"Zarr store not found at {path}")

        ds = self.read(path)

        # Calculate store size
        total_bytes = 0
        for item in path.rglob("*"):
            if item.is_file():
                total_bytes += item.stat().st_size

        info = {
            "path": str(path),
            "size_bytes": total_bytes,
            "size_mb": total_bytes / (1024 * 1024),
            "variables": list(ds.data_vars),
            "dimensions": dict(ds.sizes),
            "coords": list(ds.coords),
            "compression": ds.attrs.get("compression", "unknown"),
            "consolidated": (path / ".zmetadata").exists(),
        }

        ds.close()
        return info

    def rechunk(
        self,
        path: Union[str, Path],
        output_path: Union[str, Path],
        target_chunks: dict[str, int],
    ) -> Path:
        """Rechunk a Zarr store to new chunk sizes.

        Parameters
        ----------
        path : str or Path
            Input Zarr store path.
        output_path : str or Path
            Output Zarr store path.
        target_chunks : dict
            New chunk sizes.

        Returns
        -------
        Path
            Path to rechunked store.
        """
        path = Path(path)
        output_path = Path(output_path)

        ds = self.read(path)

        # Load into memory to avoid chunk alignment issues
        # This is necessary when source and target chunks don't align
        ds = ds.compute()

        # Clear encoding from original store to avoid conflicts
        for var in ds.data_vars:
            ds[var].encoding.clear()
        for coord in ds.coords:
            if hasattr(ds[coord], "encoding"):
                ds[coord].encoding.clear()

        # Update store chunks
        old_chunks = self.chunks
        self.chunks = target_chunks

        try:
            result = self.write(ds, output_path)
        finally:
            self.chunks = old_chunks

        return result
