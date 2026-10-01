#!/usr/bin/env python3
"""
Download DEM for a bounding box and save as GeoTIFF in a specified CRS.

Supports multiple DEM sources via dem-stitcher:
- glo_30: Copernicus GLO-30 (30m global, recommended)
- glo_90: Copernicus GLO-90 (90m global)
- srtm_v3: SRTM v3 (30m, 60°N to 56°S)
- nasadem: NASA DEM (30m, reprocessed SRTM)
- 3dep: 3DEP (10m, US only)

CRS can be specified as:
- EPSG code: --epsg 32643
- Proj4 string: --crs "+proj=aea +lat_1=45 +lat_2=51 ..."
- WKT string: --crs "PROJCS[...]"
- Built-in preset: --crs kaz_albers

Usage:
    python download_dem.py --bbox 76.5 42.0 77.5 43.0 --epsg 32643 --output dem.tif
    python download_dem.py --bbox 68 50 78 55 --crs kaz_albers --output dem_kaz.tif
    python download_dem.py --bbox 68 50 78 55 --crs "+proj=aea +lat_1=45 +lat_2=51 +lat_0=48 +lon_0=68 +ellps=WGS84 +units=m" -o dem.tif
"""

import argparse
import math
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    from dem_stitcher import stitch_dem
    from dem_stitcher.datasets import DATASETS
except ImportError:
    print("dem-stitcher not installed. Install with: pip install dem-stitcher")
    raise

try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import Resampling, calculate_default_transform, reproject
except ImportError:
    print("rasterio not installed. Install with: pip install rasterio")
    raise


# Built-in CRS presets for Central Asia work
CRS_PRESETS = {
    # Albers Equal Area Conic optimized for Kazakhstan (45-55°N, 46-87°E)
    "kaz_albers": "+proj=aea +lat_1=45 +lat_2=51 +lat_0=48 +lon_0=68 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
    # Albers for Northern Kazakhstan specifically (50-55°N, 60-78°E)
    "kaz_north_albers": "+proj=aea +lat_1=50 +lat_2=54 +lat_0=52 +lon_0=68 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
    # Lambert Conformal Conic for Kazakhstan (preserves shape/angles)
    "kaz_lcc": "+proj=lcc +lat_1=45 +lat_2=51 +lat_0=48 +lon_0=68 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
    # Central Asia Albers (broader coverage: Tajikistan, Kyrgyzstan, etc.)
    "central_asia_albers": "+proj=aea +lat_1=35 +lat_2=45 +lat_0=40 +lon_0=70 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs",
}

# Native resolution in arc-seconds for tile/size estimation
_SOURCE_ARCSEC = {
    "glo_30": 1,
    "glo_90": 3,
    "srtm_v3": 1,
    "nasadem": 1,
    "3dep": 1/3,
}


def _estimate_download(bbox, source):
    """Estimate tile count and raw array size for a download."""
    west, south, east, north = bbox
    arcsec = _SOURCE_ARCSEC.get(source, 1)
    res_deg = arcsec / 3600

    width_px = int(math.ceil((east - west) / res_deg))
    height_px = int(math.ceil((north - south) / res_deg))

    # Tiles are typically 1°x1° for glo/srtm
    n_tiles = int(math.ceil(east - west)) * int(math.ceil(north - south))

    bytes_raw = width_px * height_px * 4  # float32
    gb_raw = bytes_raw / (1024 ** 3)

    return width_px, height_px, n_tiles, gb_raw


def parse_crs(crs_input: str | int | None, epsg: int | None = None) -> CRS:
    """
    Parse CRS from various input formats.

    Parameters
    ----------
    crs_input : str or int or None
        CRS as EPSG code, proj4 string, WKT, or preset name
    epsg : int or None
        EPSG code (alternative to crs_input)

    Returns
    -------
    CRS
        Rasterio CRS object
    """
    # EPSG takes precedence if both provided
    if epsg is not None:
        return CRS.from_epsg(epsg)

    if crs_input is None:
        return CRS.from_epsg(4326)

    # Check if it's a preset name
    if isinstance(crs_input, str) and crs_input.lower() in CRS_PRESETS:
        return CRS.from_proj4(CRS_PRESETS[crs_input.lower()])

    # Try to parse as EPSG integer
    if isinstance(crs_input, int):
        return CRS.from_epsg(crs_input)

    # Try to parse string as EPSG code
    if isinstance(crs_input, str):
        # Handle "EPSG:32643" format
        if crs_input.upper().startswith("EPSG:"):
            return CRS.from_epsg(int(crs_input.split(":")[1]))

        # Try as integer string
        try:
            return CRS.from_epsg(int(crs_input))
        except ValueError:
            pass

        # Try as proj4
        if crs_input.strip().startswith("+"):
            return CRS.from_proj4(crs_input)

        # Try as WKT
        if crs_input.strip().startswith(("PROJCS", "GEOGCS", "PROJCRS", "GEOGCRS")):
            return CRS.from_wkt(crs_input)

        # Let rasterio try to figure it out
        return CRS.from_user_input(crs_input)

    raise ValueError(f"Could not parse CRS from: {crs_input}")


def _trim_nodata(array, transform, nodata):
    """Trim rows/columns that are entirely nodata from the array edges."""
    if nodata is not None and not np.isnan(nodata):
        valid = array != nodata
    else:
        valid = ~np.isnan(array)

    # Find bounding box of valid data
    valid_rows = np.where(valid.any(axis=1))[0]
    valid_cols = np.where(valid.any(axis=0))[0]

    if len(valid_rows) == 0 or len(valid_cols) == 0:
        return array, transform, array.shape[1], array.shape[0]

    r0, r1 = valid_rows[0], valid_rows[-1] + 1
    c0, c1 = valid_cols[0], valid_cols[-1] + 1

    trimmed = array[r0:r1, c0:c1]
    # Shift the transform origin to the top-left of the trimmed region
    new_transform = transform * Affine.translation(c0, r0)

    old_h, old_w = array.shape
    new_h, new_w = trimmed.shape
    if old_h != new_h or old_w != new_w:
        print(f"  Trimmed nodata edges: {old_w}x{old_h} -> {new_w}x{new_h}")

    return trimmed, new_transform, new_w, new_h


def _reproject_chunked(src_array, src_transform, src_crs,
                       dst_transform, dst_crs, dst_width, dst_height,
                       nodata, chunk_rows=512):
    """Reproject in row chunks with progress feedback."""
    dst_array = np.empty((dst_height, dst_width), dtype=src_array.dtype)
    dst_array[:] = nodata if nodata is not None else 0

    n_chunks = math.ceil(dst_height / chunk_rows)

    iter_range = range(n_chunks)
    if tqdm is not None:
        iter_range = tqdm(iter_range, desc="  Reprojecting", unit="chunk")
    else:
        print(f"  Reprojecting in {n_chunks} chunks...")

    for i in iter_range:
        row_start = i * chunk_rows
        row_end = min(row_start + chunk_rows, dst_height)
        chunk_h = row_end - row_start

        # Shift the destination transform down to this chunk's row offset
        chunk_transform = dst_transform * Affine.translation(0, row_start)

        chunk_buf = np.empty((chunk_h, dst_width), dtype=src_array.dtype)
        chunk_buf[:] = nodata if nodata is not None else 0

        reproject(
            source=src_array,
            destination=chunk_buf,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=chunk_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=nodata,
            dst_nodata=nodata,
        )

        dst_array[row_start:row_end, :] = chunk_buf

    return dst_array


def download_dem(
    bbox: list[float],
    output_path: str,
    crs: str | int | CRS | None = None,
    epsg: int | None = None,
    source: str = "glo_30",
    resolution: float | None = None,
) -> Path:
    """
    Download DEM for a bounding box and save as GeoTIFF.

    Parameters
    ----------
    bbox : list[float]
        Bounding box as [west, south, east, north] in WGS84 (EPSG:4326)
    output_path : str
        Output GeoTIFF path
    crs : str, int, CRS, or None
        Target CRS as EPSG code, proj4, WKT, preset name, or CRS object.
        Presets: 'kaz_albers', 'kaz_north_albers', 'kaz_lcc', 'central_asia_albers'
    epsg : int, optional
        Target EPSG code (alternative to crs parameter)
    source : str
        DEM source: 'glo_30', 'glo_90', 'srtm_v3', 'nasadem', '3dep' (default: glo_30)
    resolution : float, optional
        Target resolution in CRS units. If None, uses native resolution.

    Returns
    -------
    Path
        Path to the output GeoTIFF
    """
    output_path = Path(output_path)

    # Parse CRS
    if isinstance(crs, CRS):
        target_crs = crs
    else:
        target_crs = parse_crs(crs, epsg)

    # Auto-switch glo_30 -> glo_90 when target resolution is coarse
    if source == "glo_30" and resolution is not None and resolution >= 100:
        print(f"  Target resolution ({resolution}m) >= 100m: switching source glo_30 -> glo_90 to save memory")
        source = "glo_90"

    # Estimate download size
    w_px, h_px, n_tiles, gb_raw = _estimate_download(bbox, source)
    print(f"Downloading DEM from {source}...")
    print(f"  Bbox (EPSG:4326): {bbox}")
    print(f"  Target CRS: {target_crs.to_string()}")
    print(f"  Estimated: ~{n_tiles} tiles, ~{w_px}x{h_px} px, ~{gb_raw:.2f} GB raw")

    if gb_raw > 8:
        print(f"  WARNING: estimated raw array is {gb_raw:.1f} GB — this may cause OOM. "
              "Consider using glo_90 (--source glo_90) or a smaller bbox.")

    # Download and stitch DEM tiles
    dem_array, dem_profile = stitch_dem(
        bbox,
        dem_name=source,
        dst_ellipsoidal_height=False,  # Keep orthometric heights
        dst_area_or_point="Point",
    )

    print(f"  Downloaded shape: {dem_array.shape}")
    print(f"  Native CRS: {dem_profile['crs']}")

    # Check if reprojection is needed
    if target_crs != dem_profile["crs"]:
        # Calculate transform for target CRS
        dst_transform, dst_width, dst_height = calculate_default_transform(
            dem_profile["crs"],
            target_crs,
            dem_profile["width"],
            dem_profile["height"],
            *rasterio.transform.array_bounds(
                dem_profile["height"], dem_profile["width"], dem_profile["transform"]
            ),
            resolution=resolution,
        )

        nodata = dem_profile.get("nodata")

        # Reproject in chunks with progress
        dst_array = _reproject_chunked(
            src_array=dem_array,
            src_transform=dem_profile["transform"],
            src_crs=dem_profile["crs"],
            dst_transform=dst_transform,
            dst_crs=target_crs,
            dst_width=dst_width,
            dst_height=dst_height,
            nodata=nodata,
        )

        # Trim nodata edges (reprojection of a lat/lon rect into a
        # non-geographic CRS produces a non-rectangular valid region
        # with nodata fill in the corners/edges)
        dst_array, dst_transform, dst_width, dst_height = _trim_nodata(
            dst_array, dst_transform, nodata)

        # Update profile
        out_profile = dem_profile.copy()
        out_profile.update(
            crs=target_crs,
            transform=dst_transform,
            width=dst_width,
            height=dst_height,
        )
        out_array = dst_array
    else:
        out_profile = dem_profile
        out_array = dem_array

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write GeoTIFF
    out_profile.update(
        driver="GTiff",
        dtype=out_array.dtype,
        count=1,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    _write_geotiff_chunked(output_path, out_array, out_profile, source)

    # Print summary
    bounds = rasterio.transform.array_bounds(
        out_profile["height"], out_profile["width"], out_profile["transform"]
    )
    print("\nOutput summary:")
    print(f"  File: {output_path}")
    print(f"  Size: {out_profile['width']} x {out_profile['height']} pixels")
    print(f"  CRS: {out_profile['crs'].to_string()}")
    print(f"  Bounds: [{bounds[0]:.2f}, {bounds[1]:.2f}, {bounds[2]:.2f}, {bounds[3]:.2f}]")
    print(f"  Resolution: {out_profile['transform'][0]:.2f} x {abs(out_profile['transform'][4]):.2f} m")
    print(f"  Elevation range: {np.nanmin(out_array):.1f} to {np.nanmax(out_array):.1f} m")

    return output_path


def _write_geotiff_chunked(output_path, out_array, out_profile, source,
                           chunk_rows=512):
    """Write GeoTIFF in row chunks with progress feedback."""
    height = out_array.shape[0]
    n_chunks = math.ceil(height / chunk_rows)

    print(f"  Writing to {output_path}...")
    with rasterio.open(output_path, "w", **out_profile) as dst:
        if n_chunks <= 2:
            # Small file — write in one go, no progress bar needed
            dst.write(out_array, 1)
        else:
            iter_range = range(n_chunks)
            if tqdm is not None:
                iter_range = tqdm(iter_range, desc="  Writing GeoTIFF", unit="chunk")

            for i in iter_range:
                row_start = i * chunk_rows
                row_end = min(row_start + chunk_rows, height)
                window = rasterio.windows.Window(0, row_start,
                                                 out_array.shape[1],
                                                 row_end - row_start)
                dst.write(out_array[row_start:row_end, :], 1, window=window)

        dst.update_tags(source=f"DEM source: {source}")


def list_sources():
    """List available DEM sources."""
    print("Available DEM sources:")
    for name in sorted(DATASETS.keys()):
        print(f"  - {name}")


def list_presets():
    """List available CRS presets."""
    print("Available CRS presets:\n")
    for name, proj4 in CRS_PRESETS.items():
        print(f"  {name}:")
        print(f"    {proj4}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Download DEM for a bounding box and save as GeoTIFF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download using Kazakhstan Albers preset
  python download_dem.py --bbox 68 50 78 55 --crs kaz_albers -o dem_kaz.tif

  # Download using custom proj4 string
  python download_dem.py --bbox 68 50 78 55 \\
    --crs "+proj=aea +lat_1=45 +lat_2=51 +lat_0=48 +lon_0=68 +ellps=WGS84 +units=m" \\
    -o dem_custom.tif

  # Download using EPSG code (UTM 43N)
  python download_dem.py --bbox 76.5 42.0 77.5 43.0 --epsg 32643 -o dem.tif

  # Keep in WGS84
  python download_dem.py --bbox 76.5 42.0 77.5 43.0 -o dem_wgs84.tif

  # List available presets
  python download_dem.py --list-presets

CRS Presets:
  kaz_albers          - Albers Equal Area for all Kazakhstan
  kaz_north_albers    - Albers optimized for Northern Kazakhstan
  kaz_lcc             - Lambert Conformal Conic for Kazakhstan
  central_asia_albers - Albers for broader Central Asia region
        """,
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("WEST", "SOUTH", "EAST", "NORTH"),
        help="Bounding box in WGS84 (EPSG:4326)",
    )
    parser.add_argument(
        "--crs",
        type=str,
        default=None,
        help="Target CRS: preset name, EPSG code, proj4, or WKT",
    )
    parser.add_argument(
        "--epsg",
        type=int,
        default=None,
        help="Target EPSG code (shorthand for --crs)",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="dem.tif",
        help="Output GeoTIFF path (default: dem.tif)",
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="glo_30",
        choices=["glo_30", "glo_90", "srtm_v3", "nasadem", "3dep", "glo_90_missing"],
        help="DEM source (default: glo_30)",
    )
    parser.add_argument(
        "--resolution", "-r",
        type=float,
        default=None,
        help="Target resolution in CRS units (default: native)",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="List available DEM sources and exit",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available CRS presets and exit",
    )

    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    if args.list_presets:
        list_presets()
        return

    if args.bbox is None:
        parser.error("--bbox is required")

    download_dem(
        bbox=args.bbox,
        output_path=args.output,
        crs=args.crs,
        epsg=args.epsg,
        source=args.source,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
