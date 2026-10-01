"""Raster map plotting with hillshade backgrounds.

Generates publication-quality PNG maps of model output and forcing variables
mapped to the DEM grid. Uses matplotlib with headless Agg backend.
"""

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

if TYPE_CHECKING:
    import xarray as xr

    from topopyscale2.domain import Domain

log = logging.getLogger(__name__)

# Directory containing custom colormaps (JSON files)
_COLORMAP_DIR = Path(__file__).parent / "colormaps"

# Default rendering settings per variable
_VAR_DEFAULTS = {
    "snow_depth": {"cmap": "snow_purple", "units": "m", "label": "Snow Depth"},
    "swe": {"cmap": "snow_purple", "units": "mm w.e.", "label": "SWE"},
    "surface_temperature": {"cmap": "RdBu_r", "units": "K", "label": "Surface Temperature"},
    "albedo": {"cmap": "Greys_r", "units": "", "label": "Albedo"},
    "runoff": {"cmap": "Blues", "units": "mm", "label": "Runoff"},
    "temperature": {"cmap": "RdBu_r", "units": "K", "label": "Temperature"},
    "precipitation": {"cmap": "YlGnBu", "units": "mm/h", "label": "Precipitation"},
    "shortwave_direct": {"cmap": "YlOrRd", "units": "W/m\u00b2", "label": "Direct SW"},
    "wind_speed": {"cmap": "viridis", "units": "m/s", "label": "Wind Speed"},
}


def hillshade(
    dem: np.ndarray,
    dx_m: float,
    dy_m: float,
    azimuth: float = 315.0,
    altitude: float = 45.0,
) -> np.ndarray:
    """Compute hillshade illumination from a DEM.

    Parameters
    ----------
    dem : np.ndarray
        2D elevation array (y, x) in metres.
    dx_m : float
        Grid spacing in x direction (metres).
    dy_m : float
        Grid spacing in y direction (metres).
    azimuth : float
        Sun azimuth in degrees clockwise from north. Default 315 (NW).
    altitude : float
        Sun altitude above horizon in degrees. Default 45.

    Returns
    -------
    np.ndarray
        Hillshade intensity array in [0, 1], same shape as *dem*.
        NaN pixels in *dem* produce NaN in output.
    """
    dem_f = dem.astype(np.float64, copy=True)
    nan_mask = np.isnan(dem_f)

    dzdx = np.gradient(dem_f, dx_m, axis=1)
    dzdy = np.gradient(dem_f, dy_m, axis=0)

    slope = np.arctan(np.sqrt(dzdx**2 + dzdy**2))
    aspect = np.arctan2(-dzdy, dzdx)

    az = np.radians(azimuth)
    alt = np.radians(altitude)

    hs = (
        np.cos(alt) * np.cos(slope)
        + np.sin(alt) * np.sin(slope) * np.cos(az - aspect)
    )
    hs = np.clip(hs, 0.0, 1.0)
    hs[nan_mask] = np.nan
    return hs


def load_colormap(name: str) -> mcolors.Colormap:
    """Load a colormap by name.

    Tries custom colormaps from ``topopyscale2/outputs/colormaps/{name}.json``
    first, then falls back to matplotlib built-ins.

    Parameters
    ----------
    name : str
        Colormap name (e.g. ``"snow_purple"`` or ``"viridis"``).

    Returns
    -------
    matplotlib.colors.Colormap
    """
    json_path = _COLORMAP_DIR / f"{name}.json"
    if json_path.exists():
        with open(json_path) as f:
            spec = json.load(f)
        mpl = spec.get("matplotlib", {})
        colors = mpl.get("colors", [c["hex"] for c in spec.get("colors", [])])
        n = mpl.get("N", 256)
        if colors:
            return mcolors.LinearSegmentedColormap.from_list(name, colors, N=n)

    # Fallback to matplotlib built-in
    return plt.get_cmap(name)


def _estimate_cell_metres(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[float, float]:
    """Estimate grid cell size in metres from coordinate arrays.

    For geographic (degree) coordinates, converts using approximate scale
    factors at the domain centre latitude.

    Parameters
    ----------
    x_coords, y_coords : np.ndarray
        1-D coordinate arrays.

    Returns
    -------
    dx_m, dy_m : float
        Approximate cell size in metres.
    """
    dx_native = float(np.abs(np.median(np.diff(x_coords)))) if len(x_coords) > 1 else 1.0
    dy_native = float(np.abs(np.median(np.diff(y_coords)))) if len(y_coords) > 1 else 1.0

    # Heuristic: if coordinates look like degrees (small magnitude), convert
    if abs(x_coords.mean()) <= 360 and dx_native < 1.0:
        lat_centre = float(np.mean(y_coords))
        dx_m = dx_native * 111_320.0 * np.cos(np.radians(lat_centre))
        dy_m = dy_native * 110_540.0
    else:
        dx_m = dx_native
        dy_m = dy_native

    return max(dx_m, 1e-6), max(dy_m, 1e-6)


def plot_raster_map(
    data: np.ndarray,
    dem: np.ndarray,
    output_path: Union[str, Path],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    dx_m: Optional[float] = None,
    dy_m: Optional[float] = None,
    title: str = "",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    units_label: str = "",
    contour_levels: Optional[list[float]] = None,
    figsize: tuple[float, float] = (10, 8),
    dpi: int = 150,
) -> Path:
    """Render a raster map with hillshade background.

    Parameters
    ----------
    data : np.ndarray
        2D data array (y, x) to plot.
    dem : np.ndarray
        2D elevation array (y, x) for hillshade.
    output_path : str or Path
        Output PNG file path.
    x_coords, y_coords : np.ndarray
        1-D coordinate arrays for axis labelling.
    dx_m, dy_m : float, optional
        Grid spacing in metres. Estimated from coordinates if not given.
    title : str
        Plot title.
    cmap : str
        Colormap name (custom or matplotlib built-in).
    vmin, vmax : float, optional
        Colour scale limits. Auto-scaled from data if not given.
    units_label : str
        Label for the colorbar.
    contour_levels : list[float], optional
        Elevation contour levels. If None, auto-selected.
    figsize : tuple
        Figure size in inches.
    dpi : int
        Output resolution.

    Returns
    -------
    Path
        Path to the written PNG file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if dx_m is None or dy_m is None:
        dx_m, dy_m = _estimate_cell_metres(x_coords, y_coords)

    # Compute hillshade
    hs = hillshade(dem, dx_m, dy_m)

    # Resolve colormap
    cmap_obj = load_colormap(cmap)
    cmap_obj = cmap_obj.copy()
    cmap_obj.set_bad(alpha=0)  # transparent for NaN

    # Auto-scale if needed
    valid = data[np.isfinite(data)]
    if vmin is None:
        vmin = float(np.nanmin(valid)) if len(valid) > 0 else 0.0
    if vmax is None:
        vmax = float(np.nanmax(valid)) if len(valid) > 0 else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    extent = [
        float(x_coords[0]),
        float(x_coords[-1]),
        float(y_coords[-1]),  # bottom
        float(y_coords[0]),   # top
    ]

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Hillshade background
    ax.imshow(hs, cmap="gray", extent=extent, aspect="auto", vmin=0, vmax=1)

    # NaN-aware Gaussian smoothing to reduce cluster-boundary noise
    valid_mask = np.isfinite(data)
    if valid_mask.any():
        filled = np.where(valid_mask, data, 0.0)
        smoothed_vals = gaussian_filter(filled, sigma=2.0)
        smoothed_weight = gaussian_filter(valid_mask.astype(float), sigma=2.0)
        smoothed_weight = np.where(smoothed_weight > 0, smoothed_weight, 1.0)
        data = np.where(valid_mask, smoothed_vals / smoothed_weight, np.nan)

    # Data overlay with transparency
    im = ax.imshow(
        data,
        cmap=cmap_obj,
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        alpha=0.7,
        interpolation="bilinear",
    )

    # Elevation contours
    if contour_levels is None and dem is not None:
        dem_valid = dem[np.isfinite(dem)]
        if len(dem_valid) > 0:
            lo, hi = float(np.nanmin(dem_valid)), float(np.nanmax(dem_valid))
            span = hi - lo
            if span > 0:
                step = max(50, round(span / 8 / 50) * 50)
                contour_levels = list(
                    np.arange(np.ceil(lo / step) * step, hi, step)
                )

    if contour_levels and dem is not None:
        ax.contour(
            dem,
            levels=contour_levels,
            colors="k",
            linewidths=0.3,
            alpha=0.4,
            extent=extent,
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    if units_label:
        cbar.set_label(units_label)

    ax.set_title(title)
    ax.set_xlabel("Longitude" if abs(x_coords.mean()) <= 360 else "X")
    ax.set_ylabel("Latitude" if abs(y_coords.mean()) <= 360 else "Y")

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    log.info("Saved map: %s", output_path)
    return output_path


def plot_model_maps(
    model_output: "xr.Dataset",
    domain: "Domain",
    output_dir: Union[str, Path],
    time_idx: int = -1,
    variables: Optional[list[str]] = None,
) -> list[Path]:
    """Generate raster maps of model output variables.

    Uses ``RasterMapper`` to map (time, unit_id) data to the DEM grid,
    then calls :func:`plot_raster_map` for each variable.

    Parameters
    ----------
    model_output : xr.Dataset
        Model output with (time, unit_id) dimensions.
    domain : Domain
        Domain with loaded DEM and membership.
    output_dir : str or Path
        Directory for output PNGs.
    time_idx : int
        Time index to plot. Default -1 (last timestep).
    variables : list[str], optional
        Variables to plot. If None, plots all variables that have defaults.

    Returns
    -------
    list[Path]
        Paths to generated PNG files.
    """
    import pandas as pd

    from topopyscale2.outputs.raster_mapper import RasterMapper

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build mapper
    mapper = RasterMapper.from_domain(domain)

    # Select variables to plot
    if variables is None:
        variables = [v for v in model_output.data_vars if v in _VAR_DEFAULTS]
        if not variables:
            variables = list(model_output.data_vars)

    # Get DEM elevation for hillshade
    dem = domain.dem_data["elevation"].values if domain.dem_data is not None else None
    if dem is None:
        log.warning("No DEM data available for hillshade; skipping plots")
        return []

    x_coords = domain.dem_data["elevation"].x.values
    y_coords = domain.dem_data["elevation"].y.values

    # Resolve time label
    time_label = ""
    if "time" in model_output.dims and len(model_output.time) > 0:
        ts = pd.Timestamp(model_output.time.values[time_idx])
        time_label = ts.strftime("%Y-%m-%d %H:%M")

    paths = []
    for var in variables:
        if var not in model_output:
            continue

        # Slice the requested timestep BEFORE rasterising — otherwise the
        # mapper materialises a full (time, ny, nx) cube which blows memory
        # on large domains (e.g. 591 × 3639 × 6854 × 8 B ≈ 118 GB for Kaz).
        if "time" in model_output[var].dims:
            to_map = model_output[[var]].isel(time=time_idx)
        else:
            to_map = model_output[[var]]

        raster_ds = mapper.map_to_raster(to_map, [var])
        data_2d = raster_ds[var].values

        # Get plotting defaults
        defaults = _VAR_DEFAULTS.get(var, {})
        cmap = defaults.get("cmap", "viridis")
        units_label = defaults.get("units", "")
        label = defaults.get("label", var)

        title = f"{label}"
        if time_label:
            title += f" — {time_label}"

        out_path = output_dir / f"{var}_map.png"
        path = plot_raster_map(
            data=data_2d,
            dem=dem,
            output_path=out_path,
            x_coords=x_coords,
            y_coords=y_coords,
            title=title,
            cmap=cmap,
            units_label=units_label,
        )
        paths.append(path)

    return paths
