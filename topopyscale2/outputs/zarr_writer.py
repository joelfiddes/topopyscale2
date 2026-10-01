"""Zarr chunked format output writer with compression."""

import datetime
from pathlib import Path
from typing import Optional, Union

import xarray as xr

from topopyscale2.outputs.base import CF_ATTRIBUTES
from topopyscale2.spatial.units import SpatialUnit


class ZarrWriter:
    """Write forcing as Zarr store with chunked compression.

    Zarr is a format for storage of chunked, compressed, N-dimensional arrays.
    It is optimized for parallel access and cloud storage, making it ideal
    for large-scale forcing datasets.

    Features:
    - Chunked storage for efficient partial reads
    - zstd compression (or alternatives)
    - Cloud-friendly (works with s3, gcs paths)
    - Parallel write support
    - Append mode for time-series data
    """

    model_name = "generic"
    file_format = "zarr"

    def __init__(
        self,
        compression: str = "zstd",
        compression_level: int = 3,
        chunks: Optional[dict[str, int]] = None,
        consolidated: bool = True,
        append_dim: Optional[str] = None,
    ):
        """Initialize Zarr writer.

        Parameters
        ----------
        compression : str
            Compression algorithm: "zstd", "lz4", "gzip", "zlib", or None.
        compression_level : int
            Compression level (meaning varies by algorithm).
        chunks : dict, optional
            Chunk sizes per dimension. Defaults to {"time": 24, "unit_id": -1}.
            Use -1 for full dimension size.
        consolidated : bool
            Whether to consolidate metadata (faster opens).
        append_dim : str, optional
            Dimension to append along for incremental writes.
        """
        self.compression = compression
        self.compression_level = compression_level
        self.chunks = chunks or {"time": 24}
        self.consolidated = consolidated
        self.append_dim = append_dim

    def _get_encoding(self, ds: xr.Dataset) -> dict:
        """Get encoding dict for zarr write, handling zarr 2.x vs 3.x differences.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to encode.

        Returns
        -------
        dict
            Encoding dictionary for xarray.to_zarr().
        """
        if self.compression is None:
            return {}

        # Check zarr version to determine encoding strategy
        import zarr

        zarr_major = int(zarr.__version__.split(".")[0])

        encoding = {}
        for var in ds.data_vars:
            if zarr_major >= 3:
                # Zarr 3.x: use codec configuration dict with "name" and "configuration" keys
                if self.compression == "zstd":
                    encoding[var] = {
                        "compressors": [
                            {"name": "zstd", "configuration": {"level": self.compression_level}}
                        ]
                    }
                elif self.compression == "gzip":
                    encoding[var] = {
                        "compressors": [
                            {"name": "gzip", "configuration": {"level": self.compression_level}}
                        ]
                    }
                elif self.compression == "zlib":
                    encoding[var] = {
                        "compressors": [
                            {"name": "zlib", "configuration": {"level": self.compression_level}}
                        ]
                    }
                elif self.compression == "lz4":
                    encoding[var] = {
                        "compressors": [{"name": "lz4", "configuration": {}}]
                    }
                else:
                    raise ValueError(f"Unknown compression: {self.compression}")
            else:
                # Zarr 2.x: use numcodecs compressor objects
                try:
                    import numcodecs
                except ImportError:
                    import warnings
                    warnings.warn(
                        "numcodecs is not installed. Zarr compression is disabled. "
                        "Install with: pip install numcodecs"
                    )
                    return {}

                if self.compression == "zstd":
                    compressor = numcodecs.Zstd(level=self.compression_level)
                elif self.compression == "lz4":
                    compressor = numcodecs.LZ4(acceleration=self.compression_level)
                elif self.compression == "gzip":
                    compressor = numcodecs.GZip(level=self.compression_level)
                elif self.compression == "zlib":
                    compressor = numcodecs.Zlib(level=self.compression_level)
                else:
                    raise ValueError(f"Unknown compression: {self.compression}")

                encoding[var] = {"compressor": compressor}

        return encoding

    def _prepare_dataset(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
    ) -> xr.Dataset:
        """Prepare dataset with CF attributes and unit metadata.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data.
        units : list[SpatialUnit]
            Spatial units.

        Returns
        -------
        xr.Dataset
            Prepared dataset.
        """
        ds = forcing.copy()

        # Add CF attributes to variables
        for var in ds.data_vars:
            if var in CF_ATTRIBUTES:
                ds[var].attrs.update(CF_ATTRIBUTES[var])

        # Add unit metadata as coordinates
        if units and "unit_id" in ds.dims:
            ds = ds.assign_coords(
                unit_lat=("unit_id", [u.y for u in units]),
                unit_lon=("unit_id", [u.x for u in units]),
                unit_elevation=("unit_id", [u.elevation for u in units]),
                unit_surface_type=("unit_id", [u.surface_type for u in units]),
                unit_area=("unit_id", [u.area_m2 for u in units]),
            )

        # Global attributes
        ds.attrs.update({
            "Conventions": "CF-1.8",
            "title": "TopoPyScale 2.0 downscaled meteorological forcing",
            "institution": "TopoPyScale",
            "source": "ERA5 reanalysis, topographically downscaled",
            "history": f"Created {datetime.datetime.now(datetime.timezone.utc).isoformat()} by TopoPyScale 2.0",
            "format": "Zarr",
            "compression": str(self.compression),
        })

        return ds

    def _resolve_chunks(self, ds: xr.Dataset) -> dict[str, int]:
        """Resolve chunk sizes for dataset dimensions.

        Parameters
        ----------
        ds : xr.Dataset
            Dataset to chunk.

        Returns
        -------
        dict
            Resolved chunk sizes.
        """
        resolved = {}
        for dim in ds.dims:
            if dim in self.chunks:
                chunk_size = self.chunks[dim]
                if chunk_size == -1:
                    resolved[dim] = ds.sizes[dim]
                else:
                    resolved[dim] = min(chunk_size, ds.sizes[dim])
            else:
                # Default chunk for unknown dimensions
                resolved[dim] = ds.sizes[dim]
        return resolved

    def write(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        output_dir: Path,
    ) -> list[Path]:
        """Write forcing dataset to Zarr store.

        Parameters
        ----------
        forcing : xr.Dataset
            Forcing data with dims (time, unit_id) or (time,).
        units : list[SpatialUnit]
            Spatial units with metadata.
        output_dir : Path
            Output directory. The Zarr store will be created here.

        Returns
        -------
        list[Path]
            Written store paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare dataset
        ds = self._prepare_dataset(forcing, units)

        # Resolve chunks
        chunks = self._resolve_chunks(ds)
        ds = ds.chunk(chunks)

        # Determine store name from time range
        if "time" in ds.dims and ds.sizes["time"] > 0:
            t_start = str(ds.time.values[0])[:10].replace("-", "")
            t_end = str(ds.time.values[-1])[:10].replace("-", "")
            store_name = f"forcing_{t_start}_{t_end}.zarr"
        else:
            store_name = "forcing.zarr"

        store_path = output_dir / store_name

        # Build encoding with compression
        encoding = self._get_encoding(ds)

        # Write to Zarr
        if self.append_dim and store_path.exists():
            ds.to_zarr(
                store_path,
                mode="a",
                append_dim=self.append_dim,
                consolidated=self.consolidated,
            )
        else:
            ds.to_zarr(
                store_path,
                mode="w",
                encoding=encoding,
                consolidated=self.consolidated,
            )

        return [store_path]

    def append(
        self,
        forcing: xr.Dataset,
        units: list[SpatialUnit],
        store_path: Path,
        dim: str = "time",
    ) -> Path:
        """Append data along a dimension to existing Zarr store.

        Parameters
        ----------
        forcing : xr.Dataset
            New forcing data to append.
        units : list[SpatialUnit]
            Spatial units.
        store_path : Path
            Existing Zarr store path.
        dim : str
            Dimension to append along.

        Returns
        -------
        Path
            Store path.
        """
        ds = self._prepare_dataset(forcing, units)
        chunks = self._resolve_chunks(ds)
        ds = ds.chunk(chunks)

        ds.to_zarr(
            store_path,
            mode="a",
            append_dim=dim,
            consolidated=self.consolidated,
        )

        return store_path

    @staticmethod
    def open(store_path: Union[str, Path], **kwargs) -> xr.Dataset:
        """Open a Zarr store as xarray Dataset.

        Parameters
        ----------
        store_path : str or Path
            Path to Zarr store.
        **kwargs
            Additional arguments to xr.open_zarr.

        Returns
        -------
        xr.Dataset
            Loaded dataset.
        """
        return xr.open_zarr(store_path, **kwargs)
