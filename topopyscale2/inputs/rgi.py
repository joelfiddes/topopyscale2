"""RGI 7.0 glacier outlines: fetch, cache and rasterise onto a model grid.

The Randolph Glacier Inventory is the standard global glacier outline dataset.
TPS2 uses it for one thing: deciding which pixels are ice, so that clustering
can keep glacier and non-glacier terrain in separate units and the glacier ones
can be handed to GFSM.

Regional files come from NSIDC and need an Earthdata login (a ``~/.netrc``
entry for ``urs.earthdata.nasa.gov``). Downloads are cached per region, so the
cost is paid once per machine.

Region ids are the NSIDC directory names, e.g. ``13_central_asia``,
``14_south_asia_west``, ``15_south_asia_east``, ``11_central_europe``,
``10_north_asia``, ``01_alaska``. Listing:
https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/regional_files/RGI2000-v7.0-G/
"""

from __future__ import annotations

import glob
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

BASE = (
    "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/"
    "regional_files/RGI2000-v7.0-G"
)

DEFAULT_CACHE = Path.home() / ".cache" / "tps2" / "rgi7"


def fetch_region(region: str, cache: Path = DEFAULT_CACHE) -> Path:
    """Download and unzip one RGI 7.0 regional file, or return the cached copy.

    Parameters
    ----------
    region : str
        NSIDC region id, e.g. ``"13_central_asia"``.
    cache : Path
        Directory to hold the zip and the unpacked shapefile.

    Returns
    -------
    Path
        The regional shapefile.
    """
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    name = f"RGI2000-v7.0-G-{region}"
    d = cache / name
    hits = glob.glob(str(d / "*.shp"))
    if hits:
        return Path(hits[0])

    z = cache / f"{name}.zip"
    if not z.exists():
        print(f"  downloading {name}.zip (Earthdata login via ~/.netrc) ...")
        subprocess.run(
            ["curl", "-sS", "-L", "-n", "-b", str(cache / "cookies"),
             "-c", str(cache / "cookies"), "-o", str(z), f"{BASE}/{name}.zip"],
            check=True,
        )
    import zipfile  # not the unzip command: it does not exist on Windows

    with zipfile.ZipFile(z) as zf:
        zf.extractall(d)
    hits = glob.glob(str(d / "*.shp"))
    if not hits:
        raise FileNotFoundError(
            f"no shapefile in {d} after unzipping {z}. If the download was an "
            f"HTML error page rather than a zip, the Earthdata credentials in "
            f"~/.netrc are missing or wrong."
        )
    return Path(hits[0])


def grid_transform(dem: xr.Dataset | xr.DataArray):
    """Affine transform for a DEM whose coords are pixel centres.

    Works for either y orientation: the sign of the y step carries it, and
    rasterio inverts the affine rather than assuming north-up.
    """
    from rasterio.transform import Affine

    x = np.asarray(dem.x.values, dtype=float)
    y = np.asarray(dem.y.values, dtype=float)
    if x.size < 2 or y.size < 2:
        raise ValueError("need at least 2 pixels in each dimension")
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    return Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(dx, dy)


def glacier_mask(
    dem: xr.Dataset | xr.DataArray,
    regions: list[str],
    cache: Path = DEFAULT_CACHE,
    min_area_km2: float = 0.05,
    crs: Optional[str] = None,
) -> np.ndarray:
    """Rasterise RGI glacier outlines onto the DEM grid.

    Parameters
    ----------
    dem : xr.Dataset or xr.DataArray
        Must have ``x``/``y`` coords and a ``crs`` attribute (or pass ``crs``).
    regions : list of str
        NSIDC region ids to read.
    cache : Path
        RGI download cache.
    min_area_km2 : float
        Drop glaciers smaller than this. The default 0.05 km2 is well below one
        pixel at any resolution TPS2 runs at, so it removes only entries that
        cannot be resolved anyway.
    crs : str, optional
        Override the DEM CRS.

    Returns
    -------
    np.ndarray
        Boolean array shaped like the DEM grid; True where a glacier polygon
        covers the pixel centre.
    """
    import geopandas as gpd
    import pandas as pd
    import rasterio.warp
    from rasterio.features import rasterize

    from topopyscale2.spatial.dem import resolve_dem_crs

    dem_crs = resolve_dem_crs(dem, crs)
    if dem_crs is None:
        raise ValueError(
            "DEM records no CRS on the dataset or on elevation/slope/aspect, "
            "and no crs= override was given; cannot place RGI outlines on the grid"
        )

    transform = grid_transform(dem)
    ny = dem.y.size
    nx = dem.x.size

    # RGI ships in EPSG:4326, so clip on read using the grid footprint in
    # lon/lat -- reading a whole region and filtering afterwards is far slower.
    left, top = transform * (0, 0)
    right, bottom = transform * (nx, ny)
    w, s, e, n = rasterio.warp.transform_bounds(
        dem_crs, "EPSG:4326",
        min(left, right), min(top, bottom), max(left, right), max(top, bottom),
    )

    parts = []
    for region in regions:
        shp = fetch_region(region, cache)
        g = gpd.read_file(shp, bbox=(w, s, e, n))
        if "area_km2" in g.columns:
            g = g[g["area_km2"] >= min_area_km2]
        print(f"  RGI {region}: {len(g)} glaciers in domain")
        if len(g):
            parts.append(g.to_crs(dem_crs))

    if not parts:
        print("  RGI: no glaciers intersect this domain")
        return np.zeros((ny, nx), dtype=bool)

    glaciers = pd.concat(parts)
    mask = rasterize(
        ((geom, 1) for geom in glaciers.geometry),
        out_shape=(ny, nx),
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)
    print(f"  RGI: {int(mask.sum()):,} of {mask.size:,} pixels are glacier "
          f"({100 * mask.mean():.2f}%)")
    return mask
