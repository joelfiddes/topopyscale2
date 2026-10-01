"""Surface type classification."""

from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import xarray as xr

#: Canonical surface-type codes. One definition, because the clusterer names
#: units from these and a mismatch silently mislabels whole surface types.
SURFACE_TYPE_CODES: dict[str, int] = {
    "open": 0,
    "glacier": 1,
    "forest": 2,
    "rock": 3,
    "water": 4,
}
SURFACE_TYPE_NAMES: dict[int, str] = {v: k for k, v in SURFACE_TYPE_CODES.items()}


class SurfaceTypeClassifier:
    """Classify pixels by surface type from raster data.

    Sources: a custom raster, or RGI 7.0 glacier outlines. API-based land-cover
    sources (ESA WorldCover, Copernicus) are still deferred.
    """

    def classify(
        self,
        dem: xr.Dataset,
        raster_path: Optional[Path] = None,
        glacier_source: str = "none",
        rgi_regions: Optional[list[str]] = None,
        rgi_cache: Optional[Path] = None,
    ) -> Optional[xr.DataArray]:
        """Classify surface types.

        Parameters
        ----------
        dem : xr.Dataset
            DEM dataset (for coordinate reference and grid).
        raster_path : Path, optional
            Path to custom surface type raster (GeoTIFF).
        glacier_source : str
            ``"rgi"`` to mark glacier pixels from RGI 7.0 outlines,
            ``"custom"`` / ``"none"`` to take them from the raster (or not at
            all).
        rgi_regions : list of str, optional
            NSIDC RGI region ids, required when ``glacier_source="rgi"``.
        rgi_cache : Path, optional
            RGI download cache directory.

        Returns
        -------
        xr.DataArray or None
            Surface type codes (see ``SURFACE_TYPE_CODES``), or None when no
            source is configured -- in which case clustering stays unstratified.

        Notes
        -----
        With both a raster and RGI, the raster provides the base classification
        and RGI overrides the glacier class. RGI is the better glacier source
        (it is a glacier inventory, not a land-cover product that happens to
        have a snow/ice class), but it says nothing about forest or water, so
        the two are complementary rather than alternatives.
        """
        base = self._load_custom(raster_path, dem) if raster_path is not None else None

        if glacier_source != "rgi":
            return base

        if not rgi_regions:
            raise ValueError(
                "surface_types.glacier_source='rgi' requires "
                "surface_types.rgi_regions, e.g. ['13_central_asia']. See "
                "topopyscale2/inputs/rgi.py for the region ids."
            )

        from topopyscale2.inputs import rgi as rgi_mod

        kwargs = {} if rgi_cache is None else {"cache": Path(rgi_cache)}
        mask = rgi_mod.glacier_mask(dem, list(rgi_regions), **kwargs)

        if base is None:
            codes = np.full(mask.shape, SURFACE_TYPE_CODES["open"], dtype=np.int32)
        else:
            codes = base.values.astype(np.int32).copy()
            if codes.shape != mask.shape:
                raise ValueError(
                    f"surface type raster {codes.shape} does not match the DEM "
                    f"grid {mask.shape}"
                )
        codes[mask] = SURFACE_TYPE_CODES["glacier"]

        return xr.DataArray(
            codes,
            dims=["y", "x"],
            coords={"y": dem.y, "x": dem.x},
            attrs={
                "source": "rgi" if base is None else "custom+rgi",
                "rgi_regions": ",".join(rgi_regions),
                "type_codes": dict(SURFACE_TYPE_CODES),
            },
        )

    def _load_custom(self, path: Path, dem: xr.Dataset) -> xr.DataArray:
        """Load a custom surface type raster."""
        with rasterio.open(path) as src:
            data = src.read(1)

        # Assume same grid as DEM
        return xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": dem.y, "x": dem.x},
            attrs={"source": "custom", "path": str(path)},
        )

    def reclassify(
        self,
        raw: xr.DataArray,
        mapping: dict[int, str],
    ) -> xr.DataArray:
        """Reclassify raw surface type codes to TPS2 types.

        Parameters
        ----------
        raw : xr.DataArray
            Raw classification raster.
        mapping : dict[int, str]
            Maps source class codes to TPS2 types
            (glacier, forest, open, rock, water).

        Returns
        -------
        xr.DataArray
            Reclassified raster with string labels as integer codes.
        """
        # Use the canonical codes. This used to carry its own map with
        # open=3/rock=4/water=5, which disagreed with the clusterer's
        # 0=open/3=rock/4=water -- so a reclassified raster came out with
        # "open" pixels named rock, rock named water, and water named
        # "type_5". Nothing caught it because no shipped config reclassifies.
        result = np.full_like(raw.values, SURFACE_TYPE_CODES["open"], dtype=np.int32)

        for src_code, tps2_type in mapping.items():
            if tps2_type in SURFACE_TYPE_CODES:
                result[raw.values == src_code] = SURFACE_TYPE_CODES[tps2_type]

        return xr.DataArray(
            result,
            dims=raw.dims,
            coords=raw.coords,
            attrs={"type_codes": dict(SURFACE_TYPE_CODES)},
        )
