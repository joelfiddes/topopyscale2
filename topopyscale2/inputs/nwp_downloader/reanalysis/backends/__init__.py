"""Backend registry for ERA5 data sources."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from topopyscale2.inputs.nwp_downloader.base import ERA5Backend

logger = logging.getLogger(__name__)

BACKEND_REGISTRY: dict[str, str] = {
    "cds": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.cds:CDSBackend",
    "google": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.google:GoogleCloudBackend",
    "s3zarr": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.s3zarr:S3ZarrBackend",
    "openmeteo": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo:OpenMeteoBackend",
    "openmeteo_s3": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.openmeteo_s3:OpenMeteoS3Backend",
    "hybrid": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.hybrid:HybridBackend",
    "local": "topopyscale2.inputs.nwp_downloader.reanalysis.backends.local:LocalCacheBackend",
}

# Key dependency for each backend used by the availability check.
# Note: hybrid has no direct dependency, it uses other backends.
_BACKEND_DEPS: dict[str, str] = {
    "openmeteo": "requests",
    "openmeteo_s3": "omfiles",  # Also requires fsspec, s3fs
    "google": "gcsfs",
    "cds": "cdsapi",
    "s3zarr": "zarr",
    "local": "zarr",
    # "hybrid" not listed — it delegates to other backends
}


def _check_available(backend_name: str) -> bool:
    """Return True if the key dependency for *backend_name* can be imported."""
    dep = _BACKEND_DEPS.get(backend_name)
    if dep is None:
        return False
    try:
        importlib.import_module(dep)
        return True
    except ImportError:
        return False


def select_backend(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    pressure_levels: list[int] | None,
    prefer_s3: bool = False,
) -> tuple[str, dict]:
    """Choose the fastest available backend for the given parameters.

    Only reanalysis (ERA5) sources are ever chosen: ``auto`` picks the fastest
    *mirror*, never a different model. The date does not change the source.

    Priority logic:
        plev needed        → google → cds
        no plev needed     → openmeteo(era5) → openmeteo_s3 → google → cds
        no plev + prefer_s3 → openmeteo_s3 → openmeteo(era5) → google → cds

    Note:
        Use prefer_s3=True for large regions (>1000 grid points) where API rate
        limits apply, or when exact 0.25° grid alignment is required.

    Returns:
        (backend_name, backend_kwargs) ready to pass to ``get_backend``.

    Raises:
        RuntimeError: If no backend dependency is installed.
    """
    plev_needed = bool(pressure_levels)

    if plev_needed:
        candidates = [
            ("google", {}),
            ("cds", {}),
        ]
    elif prefer_s3:
        # Large regions or grid alignment required: prefer S3
        candidates = [
            ("openmeteo_s3", {"dataset": "era5"}),
            ("openmeteo", {"model": "era5", "start_date": str(start_date.date()), "end_date": str(end_date.date())}),
            ("google", {}),
            ("cds", {}),
        ]
    else:
        # Default: API is faster for small regions
        candidates = [
            ("openmeteo", {"model": "era5", "start_date": str(start_date.date()), "end_date": str(end_date.date())}),
            ("openmeteo_s3", {"dataset": "era5"}),
            ("google", {}),
            ("cds", {}),
        ]

    for name, kwargs in candidates:
        if _check_available(name):
            logger.info("Auto-selected ERA5 backend: %s", name)
            return name, kwargs

    tried = [name for name, _ in candidates]
    raise RuntimeError(
        f"No backend available. Tried: {tried}. "
        "Install one of: requests (openmeteo), omfiles (openmeteo_s3), gcsfs (google), cdsapi (cds)."
    )


def get_backend(name: str, **kwargs) -> "ERA5Backend":
    """Instantiate a backend by name.

    Args:
        name: Backend name. Available backends:
            - 'hybrid': Multi-backend (default) — routes variables to optimal sources:
              surface from S3, precip from IFS (9km), plev+strd from Google
            - 'google': Google ARCO-ERA5 (Zarr, fast, 1940+)
            - 'cds': Copernicus Climate Data Store (NetCDF, reliable, 1940+)
            - 's3zarr': Regional S3 Zarr (Central Asia, fastest when bbox fits)
            - 'openmeteo': Open-Meteo REST API (fastest, surface 1940+, plev 2022+)
            - 'openmeteo_s3': Open-Meteo S3 direct (.om files, no rate limits,
              surface only, exact 0.25° grid, use for large regions)
            - 'local': Local Zarr cache (built by ``tps2 build-cache``,
              fastest for repeated regional simulations)
        **kwargs: Passed to the backend constructor.

    Returns:
        An ERA5Backend instance.
    """
    if name not in BACKEND_REGISTRY:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {list(BACKEND_REGISTRY.keys())}"
        )

    module_path, class_name = BACKEND_REGISTRY[name].rsplit(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)
