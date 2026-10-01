"""DEM processing: acquisition, terrain analysis (slope, aspect, SVF, horizon angles).

Includes D-infinity flow routing and catchment delineation for hydrological applications.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

log = logging.getLogger(__name__)
import rasterio
import xarray as xr
from rasterio.crs import CRS

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Fallback: prange is just range
    prange = range

    def njit(*args, **kwargs):
        """No-op decorator when numba is not available."""
        def decorator(func):
            return func
        return decorator

from topopyscale2.config.schema import DomainConfig, PolygonConfig, WindConfig

# =============================================================================
# Numba-accelerated horizon computation
# =============================================================================

@njit(parallel=True, cache=True)
def _compute_horizon_numba(
    elevation: np.ndarray,
    azimuths: np.ndarray,
    res: float,
    max_pixels: int,
) -> np.ndarray:
    """JIT-compiled horizon angle computation.

    Computes horizon angles for all directions in parallel.

    Parameters
    ----------
    elevation : np.ndarray
        2D elevation array (ny, nx).
    azimuths : np.ndarray
        1D array of azimuth angles in radians.
    res : float
        Grid resolution in meters.
    max_pixels : int
        Maximum number of pixels to search.

    Returns
    -------
    np.ndarray
        3D horizon array (n_directions, ny, nx) in degrees.
    """
    ny, nx = elevation.shape
    n_dirs = len(azimuths)
    horizon = np.zeros((n_dirs, ny, nx))

    for i in prange(n_dirs):
        az = azimuths[i]
        d_row = -np.cos(az)
        d_col = np.sin(az)

        for step in range(1, max_pixels + 1):
            r_offset = d_row * step
            c_offset = d_col * step
            dist = step * res

            for iy in range(ny):
                for ix in range(nx):
                    # Target coordinates
                    tr = iy + r_offset
                    tc = ix + c_offset

                    # Bounds check
                    if tr < 0 or tr >= ny - 1 or tc < 0 or tc >= nx - 1:
                        continue

                    # Bilinear interpolation
                    r0 = int(np.floor(tr))
                    c0 = int(np.floor(tc))
                    r1 = min(r0 + 1, ny - 1)
                    c1 = min(c0 + 1, nx - 1)
                    fr = tr - r0
                    fc = tc - c0

                    z_target = (
                        elevation[r0, c0] * (1 - fr) * (1 - fc)
                        + elevation[r0, c1] * (1 - fr) * fc
                        + elevation[r1, c0] * fr * (1 - fc)
                        + elevation[r1, c1] * fr * fc
                    )

                    elev_angle = np.degrees(np.arctan2(z_target - elevation[iy, ix], dist))
                    if elev_angle > horizon[i, iy, ix]:
                        horizon[i, iy, ix] = elev_angle

    return horizon

if TYPE_CHECKING:
    import pyflwdir


def _check_pyflwdir_available() -> "pyflwdir":
    """Check if pyflwdir is available and return the module.

    Raises:
        ImportError: If pyflwdir is not installed with installation instructions.
    """
    try:
        import pyflwdir

        return pyflwdir
    except ImportError as e:
        raise ImportError(
            "pyflwdir is required for flow routing and catchment delineation. "
            "Install it with: pip install pyflwdir"
        ) from e


def resolve_dem_crs(dem, override: str | None = None) -> str | None:
    """CRS of a processed DEM, wherever it happens to be recorded.

    `process()` sets the dataset-level attr only when the config names a CRS
    explicitly. Otherwise the DEM's own CRS -- read off the source raster --
    sits on the elevation variable instead. Consumers that checked only the
    dataset attr therefore failed on every config without `domain.crs`:
    glacier_source="rgi" did, and so did the MODIS processor, which raised a
    bare "Missing dst_crs" from deep inside rasterio.

    Returns None when nothing records a CRS, so callers can raise with their
    own context.
    """
    if override:
        return override
    crs = getattr(dem, "attrs", {}).get("crs")
    if crs:
        return crs
    for var in ("elevation", "slope", "aspect"):
        if hasattr(dem, "data_vars") and var in dem.data_vars:
            crs = dem[var].attrs.get("crs")
            if crs:
                return crs
    return None


class DEMProcessor:
    """Acquire and process DEM data for topographic downscaling."""

    def __init__(
        self,
        config: DomainConfig,
        wind_config: WindConfig | None = None,
        polygon_config: PolygonConfig | None = None,
    ):
        self.config = config
        self.wind_config = wind_config
        self.polygon_config = polygon_config

    def acquire(self) -> Path:
        """Acquire DEM: verify existing path or download via dem-downloader."""
        if self.config.dem is not None:
            path = Path(self.config.dem)
            if not path.exists():
                raise FileNotFoundError(f"DEM file not found: {path}")
            return path

        if self.config.bbox is None:
            raise ValueError("Either dem path or bbox must be provided")

        from topopyscale2.inputs.dem_download import download_dem

        cache_dir = Path("./dem_cache/")
        cache_dir.mkdir(parents=True, exist_ok=True)
        output_path = cache_dir / "dem.tif"

        if output_path.exists():
            log.info("Using cached DEM: %s", output_path)
            return output_path

        kwargs = {
            "bbox": self.config.bbox,
            "output_path": str(output_path),
            "source": self.config.dem_source,
        }
        if self.config.crs is not None:
            kwargs["crs"] = self.config.crs
        if self.config.grid_resolution is not None:
            kwargs["resolution"] = self.config.grid_resolution

        download_dem(**kwargs)
        return output_path

    def load(self, path: Path) -> xr.DataArray:
        """Load DEM GeoTIFF into xarray DataArray."""
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float64)
            nodata = src.nodata
            transform = src.transform
            crs = src.crs

            ny, nx = data.shape
            x = np.array([transform * (col + 0.5, 0) for col in range(nx)])[:, 0]
            y = np.array([transform * (0, row + 0.5) for row in range(ny)])[:, 1]

        if nodata is not None:
            data[data == nodata] = np.nan

        da = xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"y": y, "x": x},
            attrs={"crs": str(crs), "resolution_m": self._resolution_meters_from_transform(transform, crs, y)},
        )
        return da

    def compute_slope_aspect(
        self, dem: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Compute slope and aspect using Horn's method.

        Returns slope in degrees and aspect (0=N, 90=E, 180=S, 270=W).

        For large DEMs (>50M pixels), uses chunked processing to limit memory
        usage to ~2 GB regardless of DEM size.
        """
        ny, nx = dem.shape
        n_pixels = ny * nx
        print(f"  Computing slope/aspect ({ny}x{nx} = {n_pixels:,} pixels)...", flush=True)

        elevation = dem.values
        res = float(dem.attrs.get("resolution_m", abs(float(dem.x[1] - dem.x[0]))))

        # Threshold: 50M pixels ≈ 400 MB per array. Above this, use chunked.
        if n_pixels > 50_000_000:
            slope_deg, aspect_deg = self._compute_slope_aspect_chunked(
                elevation, res, ny, nx
            )
        else:
            dy, dx = np.gradient(elevation, res)
            slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
            slope_deg = np.degrees(slope_rad)
            # Aspect: convention 0=N, 90=E, 180=S, 270=W.
            # Validated: atan2(-dx, -dy) produces correct radiation pattern
            # (more snow on north faces, less on south) for standard GeoTIFFs.
            aspect_rad = np.arctan2(-dx, -dy)
            aspect_deg = np.degrees(aspect_rad)
            aspect_deg = (aspect_deg + 360.0) % 360.0

        slope_da = xr.DataArray(slope_deg, dims=dem.dims, coords=dem.coords)
        aspect_da = xr.DataArray(aspect_deg, dims=dem.dims, coords=dem.coords)
        print("  Slope/aspect complete.", flush=True)
        return slope_da, aspect_da

    @staticmethod
    def _compute_slope_aspect_chunked(
        elevation: np.ndarray, res: float, ny: int, nx: int,
        chunk_rows: int = 2000,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Chunked slope/aspect for large DEMs.

        Processes the DEM in horizontal strips with 1-pixel overlap for
        correct gradient computation at strip boundaries. Peak memory is
        ~6 arrays × chunk_rows × nx × 8 bytes (e.g. ~300 MB for 2000×30000).
        """
        slope_deg = np.empty((ny, nx), dtype=np.float32)
        aspect_deg = np.empty((ny, nx), dtype=np.float32)

        n_chunks = (ny + chunk_rows - 1) // chunk_rows
        print(f"  Chunked processing: {n_chunks} strips of ~{chunk_rows} rows", flush=True)

        for i in range(n_chunks):
            # Row range for this chunk (with 1-pixel overlap on each side)
            r0 = i * chunk_rows
            r1 = min((i + 1) * chunk_rows, ny)
            # Overlap: include 1 extra row on each side for gradient edge accuracy
            pad_top = 1 if r0 > 0 else 0
            pad_bot = 1 if r1 < ny else 0
            chunk = elevation[r0 - pad_top : r1 + pad_bot, :]

            dy, dx = np.gradient(chunk, res)

            # Compute slope and aspect in-place to minimize temporaries
            np.multiply(dx, dx, out=dx)
            np.multiply(dy, dy, out=dy)
            np.add(dx, dy, out=dx)
            np.sqrt(dx, out=dx)
            np.arctan(dx, out=dx)  # dx now holds slope_rad

            # Write slope (trimming overlap rows)
            slope_deg[r0:r1, :] = np.degrees(dx[pad_top : chunk.shape[0] - pad_bot])

            # Recompute dx for aspect (need original sign)
            dy_chunk, dx_chunk = np.gradient(chunk, res)
            np.arctan2(-dx_chunk, -dy_chunk, out=dx_chunk)
            np.degrees(dx_chunk, out=dx_chunk)
            dx_chunk += 360.0
            np.remainder(dx_chunk, 360.0, out=dx_chunk)
            aspect_deg[r0:r1, :] = dx_chunk[pad_top : chunk.shape[0] - pad_bot]

        return slope_deg, aspect_deg

    def compute_horizon_angles(
        self,
        dem: xr.DataArray,
        n_directions: int = 8,
        max_distance_m: float = 3000.0,
    ) -> xr.DataArray:
        """Compute horizon angles for each azimuth direction.

        For each pixel, traces rays in n_directions azimuth directions and
        computes the maximum elevation angle to the horizon.

        Parameters
        ----------
        dem : xr.DataArray
            Digital elevation model.
        n_directions : int
            Number of azimuth directions (default 8 = every 45°).
        max_distance_m : float
            Maximum search distance in meters (default 3000).

        Returns
        -------
        xr.DataArray
            Horizon angles with shape (n_directions, ny, nx), values in degrees.
        """
        elevation = dem.values.astype(np.float64)
        ny, nx = elevation.shape
        res = float(dem.attrs.get("resolution_m", abs(float(dem.x[1] - dem.x[0]))))
        max_pixels = int(max_distance_m / res)

        azimuths = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)

        if NUMBA_AVAILABLE:
            print(f"  Computing horizon angles (numba): {n_directions} directions, {max_distance_m:.0f}m, {ny}x{nx} pixels...")
            horizon = _compute_horizon_numba(elevation, azimuths, res, max_pixels)
            print("  Horizon computation complete.")
        else:
            print(f"  Computing horizon angles (numpy): {n_directions} directions, {max_distance_m:.0f}m distance, {max_pixels} steps...")
            horizon = np.zeros((n_directions, ny, nx))

            for i, az in enumerate(azimuths):
                print(f"    Direction {i+1}/{n_directions} ({np.degrees(az):.0f}°)...", end=" ", flush=True)
                d_row = -np.cos(az)
                d_col = np.sin(az)

                for step in range(1, max_pixels + 1):
                    r_offset = d_row * step
                    c_offset = d_col * step

                    rows = np.arange(ny) + r_offset
                    cols = np.arange(nx) + c_offset

                    r0 = np.clip(np.floor(rows).astype(int), 0, ny - 1)
                    r1 = np.clip(r0 + 1, 0, ny - 1)
                    c0 = np.clip(np.floor(cols).astype(int), 0, nx - 1)
                    c1 = np.clip(c0 + 1, 0, nx - 1)

                    fr = rows - np.floor(rows)
                    fc = cols - np.floor(cols)

                    z_target = (
                        elevation[np.ix_(r0, c0)] * (1 - fr[:, None]) * (1 - fc[None, :])
                        + elevation[np.ix_(r0, c1)] * (1 - fr[:, None]) * fc[None, :]
                        + elevation[np.ix_(r1, c0)] * fr[:, None] * (1 - fc[None, :])
                        + elevation[np.ix_(r1, c1)] * fr[:, None] * fc[None, :]
                    )

                    dist = step * res
                    elev_angle = np.degrees(np.arctan2(z_target - elevation, dist))
                    horizon[i] = np.maximum(horizon[i], elev_angle)

                print("done", flush=True)

        azimuth_coords = np.degrees(azimuths)
        return xr.DataArray(
            horizon,
            dims=["azimuth", "y", "x"],
            coords={"azimuth": azimuth_coords, "y": dem.y, "x": dem.x},
        )

    def compute_horizon_at_points(
        self,
        dem: xr.DataArray,
        points: list[tuple[float, float]],
        n_directions: int = 8,
        max_distance_m: float = 3000.0,
    ) -> np.ndarray:
        """Compute horizon angles at specific points only.

        Much faster than full-raster horizon when only a few points are needed
        (e.g., cluster centroids).

        Parameters
        ----------
        dem : xr.DataArray
            Digital elevation model.
        points : list of (x, y) tuples
            Coordinates of points to compute horizon for.
        n_directions : int
            Number of azimuth directions.
        max_distance_m : float
            Maximum search distance in meters.

        Returns
        -------
        np.ndarray
            Horizon angles with shape (n_points, n_directions), values in degrees.
        """
        elevation = dem.values.astype(np.float64)
        ny, nx = elevation.shape
        res = float(dem.attrs.get("resolution_m", abs(float(dem.x[1] - dem.x[0]))))
        max_pixels = int(max_distance_m / res)

        # Convert points to pixel coordinates
        x_coords = dem.x.values
        y_coords = dem.y.values

        azimuths = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        n_points = len(points)
        horizon = np.zeros((n_points, n_directions))

        print(f"  Computing horizon at {n_points} points ({n_directions} directions, {max_distance_m:.0f}m)...")

        for pi, (px, py) in enumerate(points):
            # Find pixel indices for this point
            ix = np.argmin(np.abs(x_coords - px))
            iy = np.argmin(np.abs(y_coords - py))
            z_origin = elevation[iy, ix]

            for di, az in enumerate(azimuths):
                d_row = -np.cos(az)
                d_col = np.sin(az)

                max_angle = 0.0
                for step in range(1, max_pixels + 1):
                    tr = iy + d_row * step
                    tc = ix + d_col * step

                    # Bounds check
                    if tr < 0 or tr >= ny - 1 or tc < 0 or tc >= nx - 1:
                        break

                    # Bilinear interpolation
                    r0, c0 = int(np.floor(tr)), int(np.floor(tc))
                    r1, c1 = min(r0 + 1, ny - 1), min(c0 + 1, nx - 1)
                    fr, fc = tr - r0, tc - c0

                    z_target = (
                        elevation[r0, c0] * (1 - fr) * (1 - fc)
                        + elevation[r0, c1] * (1 - fr) * fc
                        + elevation[r1, c0] * fr * (1 - fc)
                        + elevation[r1, c1] * fr * fc
                    )

                    dist = step * res
                    elev_angle = np.degrees(np.arctan2(z_target - z_origin, dist))
                    max_angle = max(max_angle, elev_angle)

                horizon[pi, di] = max_angle

        print(f"  Horizon computation complete for {n_points} points.")
        return horizon

    def compute_svf_at_points(
        self,
        dem: xr.DataArray,
        points: list[tuple[float, float]],
        n_directions: int = 8,
        max_distance_m: float = 3000.0,
    ) -> np.ndarray:
        """Compute sky view factor at specific points.

        Parameters
        ----------
        dem : xr.DataArray
            Digital elevation model.
        points : list of (x, y) tuples
            Coordinates of points.
        n_directions : int
            Number of azimuth directions.
        max_distance_m : float
            Maximum search distance.

        Returns
        -------
        np.ndarray
            SVF values for each point, shape (n_points,).
        """
        horizon = self.compute_horizon_at_points(dem, points, n_directions, max_distance_m)
        horizon_rad = np.radians(np.maximum(horizon, 0.0))
        svf = np.mean(np.cos(horizon_rad) ** 2, axis=1)
        return svf

    def compute_sky_view_factor(self, horizon_angles: xr.DataArray) -> xr.DataArray:
        """Compute sky view factor from horizon angles.

        SVF = (1/N) * sum(cos^2(horizon_angle_i))
        """
        horizon_rad = np.radians(horizon_angles.values)
        # Clamp negative horizon angles to 0 (horizon below the horizontal plane)
        horizon_rad = np.maximum(horizon_rad, 0.0)
        svf = np.mean(np.cos(horizon_rad) ** 2, axis=0)

        return xr.DataArray(svf, dims=["y", "x"], coords={"y": horizon_angles.y, "x": horizon_angles.x})

    def compute_winstral_sx(
        self,
        dem: xr.DataArray,
        n_directions: int = 36,
        search_distance_m: float = 300.0,
    ) -> xr.DataArray:
        """Compute Winstral Sx (maximum upwind slope) for wind exposure.

        For each wind direction (azimuth), traces upwind from each pixel and
        computes the maximum slope angle to any point within the search distance.

        Sx is defined as:
            Sx = max(atan2(z_upwind - z_current, distance))

        where we look in the upwind direction (opposite to wind direction).

        Positive Sx indicates exposed terrain (upwind is lower = ridgeline).
        Negative Sx indicates sheltered terrain (upwind is higher = valley/lee).

        Reference: Winstral, A., Elder, K., & Davis, R. E. (2002). Spatial snow
        modeling of wind-redistributed snow using terrain-based parameters.
        Journal of Hydrometeorology, 3(5), 524-538.

        Args:
            dem: Digital elevation model as xarray DataArray with 'y', 'x' dims
            n_directions: Number of azimuth directions (default 36 = 10° spacing)
            search_distance_m: Maximum distance to search upwind in meters

        Returns:
            DataArray with shape (n_directions, ny, nx), values in degrees.
            Coordinate 'azimuth' gives the wind direction (where wind comes FROM).
        """
        elevation = dem.values
        ny, nx = elevation.shape

        # Get resolution - handle single-pixel DEMs
        if "resolution_m" in dem.attrs:
            res = float(dem.attrs["resolution_m"])
        elif len(dem.x) > 1:
            res = abs(float(dem.x[1] - dem.x[0]))
        else:
            # Single pixel DEM - use a default resolution
            res = 30.0

        max_pixels = int(search_distance_m / res)

        # Azimuths represent wind direction (where wind comes FROM)
        # We look in that direction (upwind) to find sheltering/exposure
        azimuths = np.linspace(0, 2 * np.pi, n_directions, endpoint=False)
        sx = np.full((n_directions, ny, nx), -np.inf)

        for i, az in enumerate(azimuths):
            # Direction vector in pixel coordinates (row, col)
            # Azimuth 0 = North = wind from north = look northward (upwind)
            # In the DEM array, row index increases with y (northward), so:
            # - Looking north (az=0): d_row = +cos(0) = +1 (increasing row)
            # - Looking east (az=90): d_col = +sin(90) = +1 (increasing col)
            d_row = np.cos(az)
            d_col = np.sin(az)

            for step in range(1, max_pixels + 1):
                r_offset = d_row * step
                c_offset = d_col * step

                # Target pixel coordinates (upwind location)
                rows = np.arange(ny) + r_offset
                cols = np.arange(nx) + c_offset

                # Check bounds - mark out-of-bounds as invalid
                valid_rows = (rows >= 0) & (rows <= ny - 1)
                valid_cols = (cols >= 0) & (cols <= nx - 1)

                # Bilinear interpolation indices (clamped for interpolation)
                r0 = np.clip(np.floor(rows).astype(int), 0, ny - 1)
                r1 = np.clip(r0 + 1, 0, ny - 1)
                c0 = np.clip(np.floor(cols).astype(int), 0, nx - 1)
                c1 = np.clip(c0 + 1, 0, nx - 1)

                # Fractional parts for interpolation
                fr = rows - np.floor(rows)
                fc = cols - np.floor(cols)

                # Sample elevation at upwind points via bilinear interpolation
                z_upwind = (
                    elevation[np.ix_(r0, c0)] * (1 - fr[:, None]) * (1 - fc[None, :])
                    + elevation[np.ix_(r0, c1)] * (1 - fr[:, None]) * fc[None, :]
                    + elevation[np.ix_(r1, c0)] * fr[:, None] * (1 - fc[None, :])
                    + elevation[np.ix_(r1, c1)] * fr[:, None] * fc[None, :]
                )

                # Distance in meters
                dist = step * res

                # Slope angle: positive if upwind is lower (exposed)
                # negative if upwind is higher (sheltered)
                # Sx = atan2(z_current - z_upwind, distance)
                #    = atan2(-(z_upwind - z_current), distance)
                # Positive = we are higher than upwind = exposed/ridgeline
                # Negative = we are lower than upwind = sheltered/lee
                slope_angle = np.degrees(np.arctan2(elevation - z_upwind, dist))

                # Apply boundary mask - only update where upwind point is valid
                valid_mask = valid_rows[:, None] & valid_cols[None, :]
                slope_angle = np.where(valid_mask, slope_angle, -np.inf)

                # Keep maximum slope angle (most exposed/least sheltered)
                sx[i] = np.maximum(sx[i], slope_angle)

        # Replace -inf (no valid upwind pixels) with 0
        sx = np.where(np.isinf(sx), 0.0, sx)

        azimuth_coords = np.degrees(azimuths)
        return xr.DataArray(
            sx,
            dims=["wind_direction", "y", "x"],
            coords={"wind_direction": azimuth_coords, "y": dem.y, "x": dem.x},
            attrs={
                "long_name": "Winstral Sx wind exposure index",
                "units": "degrees",
                "description": "Maximum upwind slope angle. Positive=exposed, Negative=sheltered",
                "search_distance_m": search_distance_m,
            },
        )

    def compute_flow_directions(self, dem: xr.DataArray) -> xr.DataArray:
        """Compute D8/D-infinity flow directions using pyflwdir.

        Uses the LDD (Local Drain Direction) algorithm to compute flow directions.
        Handles flat areas and pits using the fill_depressions and resolve_flats
        options in pyflwdir.

        Args:
            dem: Digital elevation model as xarray DataArray with 'y', 'x' dims.
                 NaN values are treated as nodata.

        Returns:
            DataArray with flow direction codes (pyflwdir d8 power-of-2 encoding).
            Values are powers of 2:
                - 1: E, 2: SE, 4: S, 8: SW, 16: W, 32: NW, 64: N, 128: NE
                - 0: pit/outlet, 247: nodata
            Also includes pyflwdir-compatible attributes for reconstruction.

        Raises:
            ImportError: If pyflwdir is not installed.
        """
        pyflwdir = _check_pyflwdir_available()

        elevation = dem.values.astype(np.float32)

        # Create a mask for valid (non-NaN) data
        mask = ~np.isnan(elevation)

        # Replace NaN with a nodata value for pyflwdir
        nodata = -9999.0
        elevation_filled = np.where(mask, elevation, nodata)

        # Get resolution
        res = float(dem.attrs.get("resolution_m", abs(float(dem.x[1] - dem.x[0])) if len(dem.x) > 1 else 30.0))

        # Compute flow directions using pyflwdir
        # Use d8 algorithm which is more stable for most applications
        # latlon=False assumes projected coordinates (meters)
        transform = rasterio.transform.from_bounds(
            float(dem.x.min()) - res / 2,
            float(dem.y.min()) - res / 2,
            float(dem.x.max()) + res / 2,
            float(dem.y.max()) + res / 2,
            len(dem.x),
            len(dem.y),
        )

        # from_dem fills depressions and resolves flats internally
        flw = pyflwdir.from_dem(
            data=elevation_filled,
            nodata=nodata,
            latlon=False,
            transform=transform,
        )

        # Get the flow direction raster in d8 format
        flwdir = flw.to_array(ftype="d8")

        return xr.DataArray(
            flwdir,
            dims=["y", "x"],
            coords={"y": dem.y, "x": dem.x},
            attrs={
                "long_name": "D8 flow direction",
                "description": "Flow direction codes (1=NE, 2=E, 3=SE, 4=S, 5=pit, 6=N, 7=NW, 8=W, 9=SW)",
                "_FillValue": 247,
                "resolution_m": res,
            },
        )

    def delineate_catchments(
        self,
        dem: xr.DataArray,
        flow_dir: xr.DataArray,
        min_area_m2: float = 1e6,
    ) -> xr.DataArray:
        """Delineate sub-catchments using flow directions.

        Identifies basins (catchments) from the flow direction raster and filters
        by minimum area threshold.

        Args:
            dem: Digital elevation model (used for metadata consistency).
            flow_dir: Flow direction DataArray from compute_flow_directions().
            min_area_m2: Minimum catchment area in square meters. Smaller catchments
                         are merged with their downstream neighbor or marked as nodata.
                         Default is 1e6 m^2 (1 km^2).

        Returns:
            DataArray with integer catchment IDs. Each connected drainage basin
            has a unique positive integer ID. NoData areas have value 0.

        Raises:
            ImportError: If pyflwdir is not installed.
        """
        pyflwdir = _check_pyflwdir_available()

        res = float(flow_dir.attrs.get("resolution_m", 30.0))
        cell_area_m2 = res * res

        # Reconstruct FlwdirRaster from the flow direction array
        transform = rasterio.transform.from_bounds(
            float(dem.x.min()) - res / 2,
            float(dem.y.min()) - res / 2,
            float(dem.x.max()) + res / 2,
            float(dem.y.max()) + res / 2,
            len(dem.x),
            len(dem.y),
        )

        flw = pyflwdir.from_array(
            data=flow_dir.values.astype(np.uint8),
            ftype="d8",
            transform=transform,
            latlon=False,
        )

        # Delineate all basins
        basins = flw.basins()

        # Get unique basin IDs and their areas
        unique_ids, counts = np.unique(basins, return_counts=True)

        # Filter by minimum area
        min_cells = int(min_area_m2 / cell_area_m2)

        # Create mapping: small basins -> 0 (nodata)
        # Could also merge with downstream, but setting to nodata is simpler
        id_map = {}
        valid_id = 1
        for basin_id, count in zip(unique_ids, counts):
            if basin_id == 0:
                # nodata stays nodata
                id_map[basin_id] = 0
            elif count >= min_cells:
                id_map[basin_id] = valid_id
                valid_id += 1
            else:
                # Small catchment - mark as nodata
                id_map[basin_id] = 0

        # Apply remapping
        catchments = np.vectorize(id_map.get)(basins).astype(np.int32)

        return xr.DataArray(
            catchments,
            dims=["y", "x"],
            coords={"y": dem.y, "x": dem.x},
            attrs={
                "long_name": "Catchment IDs",
                "description": f"Sub-catchment delineation (min_area={min_area_m2} m^2)",
                "_FillValue": 0,
                "min_area_m2": min_area_m2,
                "n_catchments": valid_id - 1,
            },
        )

    def compute_flow_accumulation(
        self,
        dem: xr.DataArray,
        flow_dir: xr.DataArray,
    ) -> xr.DataArray:
        """Compute flow accumulation (upstream contributing area).

        For each cell, computes the total number of upstream cells that drain
        through it. Useful for avalanche routing, stream network extraction,
        and other hydrological applications.

        Args:
            dem: Digital elevation model (used for metadata consistency).
            flow_dir: Flow direction DataArray from compute_flow_directions().

        Returns:
            DataArray with flow accumulation values (upstream cell count).
            Can be converted to area by multiplying by cell_area_m2.
            Includes 'upstream_area_m2' coordinate transformation in attrs.

        Raises:
            ImportError: If pyflwdir is not installed.
        """
        pyflwdir = _check_pyflwdir_available()

        res = float(flow_dir.attrs.get("resolution_m", 30.0))
        cell_area_m2 = res * res

        # Reconstruct FlwdirRaster from the flow direction array
        transform = rasterio.transform.from_bounds(
            float(dem.x.min()) - res / 2,
            float(dem.y.min()) - res / 2,
            float(dem.x.max()) + res / 2,
            float(dem.y.max()) + res / 2,
            len(dem.x),
            len(dem.y),
        )

        flw = pyflwdir.from_array(
            data=flow_dir.values.astype(np.uint8),
            ftype="d8",
            transform=transform,
            latlon=False,
        )

        # Compute upstream area (in number of cells)
        acc = flw.upstream_area(unit="cell")

        return xr.DataArray(
            acc,
            dims=["y", "x"],
            coords={"y": dem.y, "x": dem.x},
            attrs={
                "long_name": "Flow accumulation",
                "units": "cells",
                "description": "Upstream contributing cell count",
                "cell_area_m2": cell_area_m2,
                "resolution_m": res,
            },
        )

    def process(
        self,
        compute_sx: bool | None = None,
        compute_catchments: bool | None = None,
        skip_horizon: bool = False,
    ) -> xr.Dataset:
        """Full DEM processing pipeline.

        Args:
            compute_sx: Whether to compute Winstral Sx. If None, automatically
                determined from wind_config (True if wind method is 'winstral').
            compute_catchments: Whether to compute flow directions, flow accumulation,
                and catchment delineation. If None, automatically determined from
                polygon_config (True if respect_catchments is True).
            skip_horizon: If True, skip horizon angle and SVF computation. Use this
                for large domains where you want to compute SVF only at cluster
                centroids (via compute_svf_at_points) after clustering.

        Returns:
            Dataset with: elevation, slope, aspect.
            If skip_horizon is False: also includes svf, horizon_angles.
            If compute_sx is True, also includes winstral_sx.
            If compute_catchments is True, also includes flow_direction,
            flow_accumulation, and catchments.
        """
        path = self.acquire()
        dem = self.load(path)
        slope, aspect = self.compute_slope_aspect(dem)

        # For large DEMs (>50M pixels), convert to float32 immediately to
        # halve memory. Delete the float64 DEM before creating the dataset
        # so both copies don't coexist.
        n_pixels = dem.shape[0] * dem.shape[1]
        if n_pixels > 50_000_000:
            import gc
            # slope/aspect already float32 from chunked code; convert elevation
            elev_f32 = dem.astype(np.float32)
            del dem
            gc.collect()
            dem = elev_f32

        ds = xr.Dataset(
            {
                "elevation": dem,
                "slope": slope,
                "aspect": aspect,
            }
        )

        # Store CRS in dataset attributes for downstream coordinate transforms
        if self.config.crs is not None:
            ds.attrs["crs"] = self.config.crs

        # Compute horizon and SVF unless skipped
        if not skip_horizon:
            n_dirs = getattr(self.config, "horizon_n_directions", 8)
            max_dist = getattr(self.config, "horizon_max_distance_m", 3000.0)
            horizon = self.compute_horizon_angles(dem, n_directions=n_dirs, max_distance_m=max_dist)
            svf = self.compute_sky_view_factor(horizon)
            ds["svf"] = svf
            ds["horizon_angles"] = horizon
        else:
            print("  Skipping horizon/SVF (will compute at cluster centroids).")

        # Determine whether to compute Sx
        if compute_sx is None:
            compute_sx = (
                self.wind_config is not None
                and self.wind_config.method == "winstral"
            )

        if compute_sx:
            # Use wind config parameters if available, otherwise use defaults
            n_directions = 36
            search_distance_m = 300.0
            if self.wind_config is not None:
                n_directions = self.wind_config.sx_n_directions
                search_distance_m = self.wind_config.sx_search_distance_m

            sx = self.compute_winstral_sx(
                dem,
                n_directions=n_directions,
                search_distance_m=search_distance_m,
            )
            ds["winstral_sx"] = sx

        # Determine whether to compute catchments
        if compute_catchments is None:
            compute_catchments = (
                self.polygon_config is not None
                and self.polygon_config.respect_catchments
            )

        if compute_catchments:
            flow_dir = self.compute_flow_directions(dem)
            flow_acc = self.compute_flow_accumulation(dem, flow_dir)

            # Get min area from polygon_config or use default
            min_area_m2 = 1e6
            if self.polygon_config is not None:
                min_area_m2 = self.polygon_config.min_area_m2

            catchments = self.delineate_catchments(dem, flow_dir, min_area_m2=min_area_m2)

            ds["flow_direction"] = flow_dir
            ds["flow_accumulation"] = flow_acc
            ds["catchments"] = catchments

        return ds

    def _resolution_meters_from_transform(self, transform, crs, y_coords) -> float:
        """Extract pixel resolution in meters."""
        if crs and CRS(crs).is_geographic:
            # Approximate meters from degrees at mean latitude
            mean_lat = np.mean(y_coords)
            lat_rad = np.radians(mean_lat)
            m_per_deg = 111320.0 * np.cos(lat_rad)
            return abs(transform.a) * m_per_deg
        return abs(transform.a)
