"""ForcingStore protocol - abstract interface for storage backends."""

from pathlib import Path
from typing import Protocol, Union

import xarray as xr


class ForcingStore(Protocol):
    """Protocol for forcing data storage backends.

    This protocol defines the interface that all storage backends must implement.
    It supports write, read, append, and existence checking operations.

    Example usage:
        >>> store = ZarrStore(compression="zstd")
        >>> store.write(ds, Path("/data/forcing.zarr"))
        >>> ds = store.read(Path("/data/forcing.zarr"))
    """

    def write(self, data: xr.Dataset, path: Union[str, Path]) -> Path:
        """Write dataset to storage.

        Parameters
        ----------
        data : xr.Dataset
            Dataset to write.
        path : str or Path
            Output path (file or directory depending on backend).

        Returns
        -------
        Path
            Path to written store.
        """
        ...

    def read(self, path: Union[str, Path], **kwargs) -> xr.Dataset:
        """Read dataset from storage.

        Parameters
        ----------
        path : str or Path
            Path to read from.
        **kwargs
            Backend-specific read options.

        Returns
        -------
        xr.Dataset
            Loaded dataset.
        """
        ...

    def append(
        self,
        data: xr.Dataset,
        path: Union[str, Path],
        dim: str = "time"
    ) -> Path:
        """Append data along a dimension to existing store.

        Parameters
        ----------
        data : xr.Dataset
            Data to append.
        path : str or Path
            Path to existing store.
        dim : str
            Dimension to append along (default: "time").

        Returns
        -------
        Path
            Path to updated store.
        """
        ...

    def exists(self, path: Union[str, Path]) -> bool:
        """Check if store exists at path.

        Parameters
        ----------
        path : str or Path
            Path to check.

        Returns
        -------
        bool
            True if store exists.
        """
        ...
