"""Self-contained forcing page: one offline HTML file showing a TPS2 downscaling run.

The page shows what the downscaling engine does. Pick a variable (temperature,
precipitation, snowfall, shortwave, longwave, wind, humidity) and it is drawn day by day on
shaded relief for every terrain unit. Beside the map are that variable against unit
elevation (the lapse the downscaling produces), an elevation-by-time view of the season, and
the season at any unit you click. Everything is embedded, so the file opens offline.

Inputs, all from a simulation directory:

- ``dem_cache/dem.tif`` and ``dem_cache/cluster_map.tif`` (written by ``tps2 setup``/``run``)
- the forcing: ``output/forcing.nc`` or ``output/forcing.zarr`` (hourly, from ``tps2 run``),
  or a daily summary ``output/forcing_daily.nc`` written by :func:`write_daily_summary`,
  which is small enough to ship with an example
- ``config.yaml`` (embedded for provenance, optional)
- optionally a page YAML with title, lede, bullet lists, credits and logo (``PageText``)
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from topopyscale2.outputs.base import installed_engine_ref

TEMPLATE = Path(__file__).with_name("forcing_page_template.html")
DAILY_FILE = "forcing_daily.nc"
MAX_PIXELS = 1_500_000      # larger rasters are strided down to keep the page small
SCHEMA_VERSION = 1          # bump when the payload changes shape; the template checks it

# Daily layers: name -> (label, units, int16 scale, colour ramp id, fixed display range).
# Values are stored as round(value * scale) in int16.
LAYERS = {
    "t_mean": ("Air temperature, daily mean", "°C", 10, "temp", (-20.0, 20.0)),
    "precip": ("Precipitation, daily total", "mm", 10, "precip", (0.0, 40.0)),
    "snowfall": ("Snowfall, daily total (water equivalent)", "mm", 10, "snow", (0.0, 40.0)),
    "sw": ("Incoming shortwave, daily mean", "W m⁻²", 1, "sun", (0.0, 400.0)),
    "lw": ("Incoming longwave, daily mean", "W m⁻²", 1, "lw", (150.0, 350.0)),
    "wind": ("Wind speed, daily mean", "m s⁻¹", 10, "wind", (0.0, 12.0)),
    "rh": ("Relative humidity, daily mean", "%", 1, "rh", (0.0, 100.0)),
}

_DEFAULT_WHAT = [
    "<b>Terrain:</b> the domain DEM, grouped into terrain units of similar elevation, slope, "
    "aspect and sky view. Switch to “Terrain units” to see them.",
    "<b>Downscaling:</b> TPS2 turns coarse reanalysis (ERA5, ~31 km) into hourly forcing for "
    "every unit: temperature and humidity from the pressure levels at the unit's elevation, "
    "shortwave corrected for slope, aspect, shading and sky view, precipitation split into "
    "rain and snow by wet-bulb temperature.",
    "<b>The page</b> shows daily values; the run itself is hourly.",
]


@dataclass
class PageText:
    """Wording and branding for the page. Every field is optional."""

    title: str = "Downscaled climate forcing"
    lede: str = ""
    what: list[str] = field(default_factory=lambda: list(_DEFAULT_WHAT))
    limitations: list[str] = field(default_factory=list)
    credits: str = ""
    reproduce: str = ""
    tiles: list[dict] = field(default_factory=list)   # extra tiles: {k, v, s}
    logo: str | None = None          # path to an SVG/PNG, relative to the page YAML
    logo_dark: str | None = None     # variant for dark backgrounds
    logo_link: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "PageText":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        unknown = set(raw) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown page keys in {path}: {sorted(unknown)}")
        spec = cls(**raw)
        base = Path(path).resolve().parent
        for attr in ("logo", "logo_dark"):
            value = getattr(spec, attr)
            if value:
                setattr(spec, attr, str((base / value).resolve()))
        return spec


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _data_uri(path: str | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    mime = {".svg": "image/svg+xml", ".png": "image/png"}.get(p.suffix.lower())
    if mime is None:
        raise ValueError(f"logo must be .svg or .png: {p}")
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode("ascii")


# ---------------------------------------------------------------- daily summary


def _amount_per_step(da, step_hours: float):
    """Precipitation-like variable as an amount [mm] per time step, from its units."""
    units = str(da.attrs.get("units", "")).replace(" ", "").lower()
    if units in ("kgm-2", "mm"):                    # already an amount per step
        return da
    if units in ("kgm-2s-1", "mms-1", "mm/s"):
        return da * 3600.0 * step_hours
    if units in ("mm/hr", "mm/h", "mmh-1", "mmhr-1", "kgm-2h-1"):
        return da * step_hours
    raise ValueError(f"precipitation units '{da.attrs.get('units')}' are not recognised")


def daily_summary(ds):
    """Daily per-unit layers (see ``LAYERS``) from TPS2 forcing on (time, unit)."""
    import pandas as pd
    import xarray as xr

    unit_dim = "unit" if "unit" in ds.dims else "unit_id"
    times = pd.DatetimeIndex(ds["time"].values)
    if len(times) < 2:
        raise ValueError("forcing needs at least two time steps")
    step_hours = float(np.median(np.diff(times.values).astype("timedelta64[s]").astype(float))) / 3600.0

    def need(name):
        if name not in ds:
            raise ValueError(f"forcing has no '{name}' variable")
        return ds[name]

    temperature = need("temperature")
    t_c = temperature - 273.15 if str(temperature.attrs.get("units", "K")) == "K" else temperature
    rh = need("humidity_relative")
    # humidity_relative is a fraction (units "1"); older files may already hold percent.
    rh_pct = rh * 100.0 if float(rh.max()) <= 1.5 else rh
    sw = need("shortwave_direct") + need("shortwave_diffuse")
    out = xr.Dataset({
        "t_mean": t_c.resample(time="1D").mean(),
        "precip": _amount_per_step(need("precipitation"), step_hours).resample(time="1D").sum(),
        "snowfall": _amount_per_step(need("snowfall"), step_hours).resample(time="1D").sum(),
        "sw": sw.resample(time="1D").mean(),
        "lw": need("longwave").resample(time="1D").mean(),
        "wind": need("wind_speed").resample(time="1D").mean(),
        "rh": rh_pct.resample(time="1D").mean(),
    }).transpose("time", unit_dim)
    if unit_dim != "unit":
        out = out.rename({unit_dim: "unit"})
    for name, (label, units, *_rest) in LAYERS.items():
        out[name].attrs = {"long_name": label, "units": units}
    out.attrs = {k: v for k, v in ds.attrs.items() if k.startswith("tps2_")}
    return out


def _find_forcing(sim_dir: Path) -> Path:
    out = sim_dir / "output"
    for name in (DAILY_FILE, "forcing.nc", "forcing.zarr"):
        if (out / name).exists():
            return out / name
    raise FileNotFoundError(f"no forcing in {out}: expected {DAILY_FILE}, forcing.nc or forcing.zarr "
                            f"(run `tps2 run` first)")


def load_daily(sim_dir: Path):
    """The daily layers for ``sim_dir``: the shipped summary, or computed from the forcing."""
    import xarray as xr

    path = _find_forcing(Path(sim_dir))
    if path.name == DAILY_FILE:
        return xr.load_dataset(path)
    opener = xr.open_zarr if path.suffix == ".zarr" else xr.open_dataset
    with opener(path) as ds:
        return daily_summary(ds).load()


def write_daily_summary(sim_dir: Path, *, engine_ref: str | None = None,
                        run_date: str | None = None) -> Path:
    """Write ``output/forcing_daily.nc`` (compact, int16-packed) from the hourly forcing.

    A few hundred kB for a season of 150 units, so an example can ship it and rebuild its
    page offline. ``engine_ref``/``run_date`` are stamped for provenance.
    """
    sim_dir = Path(sim_dir)
    daily = load_daily(sim_dir)
    daily.attrs["tps2_engine"] = engine_ref or daily.attrs.get("tps2_engine") or installed_engine_ref()
    daily.attrs["tps2_run_date"] = run_date or daily.attrs.get("tps2_run_date") or "unknown"
    encoding = {name: {"dtype": "int16", "scale_factor": 1.0 / LAYERS[name][2], "_FillValue": -32768,
                       "zlib": True, "complevel": 9} for name in LAYERS}
    path = sim_dir / "output" / DAILY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_netcdf(path, encoding=encoding)
    return path


# ---------------------------------------------------------------- page payload


def _read_rasters(sim_dir: Path):
    import rasterio

    with rasterio.open(sim_dir / "dem_cache" / "dem.tif") as r:
        dem = r.read(1).astype("float32")
        if r.nodata is not None:
            dem[dem == r.nodata] = np.nan
        transform, crs = r.transform, r.crs
    with rasterio.open(sim_dir / "dem_cache" / "cluster_map.tif") as r:
        cmap = r.read(1)
        cmap_nodata = r.nodata
    if dem.shape != cmap.shape:
        raise ValueError(f"dem.tif {dem.shape} and cluster_map.tif {cmap.shape} differ in shape")
    return dem, cmap, cmap_nodata, transform, crs


def _pixel_size_m(transform, crs, lat_mid: float) -> tuple[float, float]:
    if crs is not None and crs.is_geographic:
        return (abs(transform.a) * 111_320 * np.cos(np.radians(lat_mid)), abs(transform.e) * 111_320)
    return abs(transform.a), abs(transform.e)


def build_page_data(sim_dir: Path, *, engine_ref: str | None = None, run_date: str | None = None) -> dict:
    """Assemble the JSON payload the page template renders."""
    sim_dir = Path(sim_dir).resolve()
    dem, cmap, cmap_nodata, transform, crs = _read_rasters(sim_dir)
    daily = load_daily(sim_dir)
    n_units, ndays = daily.sizes["unit"], daily.sizes["time"]

    # Terrain-unit map: ids outside [0, n_units) are no-data.
    cm = cmap.astype("int64")
    invalid = (cm < 0) | (cm >= n_units)
    if cmap_nodata is not None:
        invalid |= cmap == cmap_nodata
    nodata_id = 65535 if n_units > 254 else 255
    cm = np.where(invalid, nodata_id, cm).astype("uint16" if n_units > 254 else "uint8")

    stride = max(1, int(np.ceil(np.sqrt(dem.size / MAX_PIXELS))))
    full_transform = transform
    if stride > 1:
        dem, cm = dem[::stride, ::stride], cm[::stride, ::stride]
        transform = transform * transform.scale(stride, stride)
    h, w = dem.shape

    valid = cm != nodata_id
    sums = np.bincount(cm[valid].astype(np.int64), weights=np.nan_to_num(dem[valid]), minlength=n_units)
    counts = np.bincount(cm[valid].astype(np.int64), minlength=n_units)
    unit_elev = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    unit_area = counts.astype(float)

    lat_mid = (full_transform * (0, h * stride / 2))[1]
    if crs is not None and not crs.is_geographic:
        lat_mid = 45.0  # only used for geographic rasters
    dx, dy = _pixel_size_m(transform, crs, lat_mid)

    layers = {}
    for name, (label, units, scale, ramp, (lo, hi)) in LAYERS.items():
        v = daily[name].transpose("time", "unit").to_numpy().astype(float)
        packed = np.where(np.isfinite(v), np.clip(np.round(v * scale), -32767, 32767), -32768).astype("int16")
        layers[name] = dict(label=label, units=units, scale=scale, ramp=ramp, lo=lo, hi=hi,
                            data=_b64(packed))

    import pandas as pd

    start = pd.Timestamp(daily["time"].values[0]).strftime("%Y-%m-%d")
    config = sim_dir / "config.yaml"
    return dict(
        schema_version=SCHEMA_VERSION,
        w=w, h=h, dx=dx, dy=dy, nodata=nodata_id, cmap_bits=16 if n_units > 254 else 8,
        start=start, ndays=ndays, n_units=n_units,
        unit_elev=[None if np.isnan(z) else round(float(z)) for z in unit_elev],
        unit_area=[int(a) for a in unit_area],
        dem=_b64(np.where(np.isnan(dem), -32768, np.round(dem)).astype("int16")),
        cmap=_b64(cm),
        dem_min=round(float(np.nanmin(dem))), dem_max=round(float(np.nanmax(dem))),
        layers=layers, layer_order=list(LAYERS),
        stamp=dict(
            engine=engine_ref or daily.attrs.get("tps2_engine") or installed_engine_ref(),
            run_date=run_date or daily.attrs.get("tps2_run_date") or "unknown",
            config=config.read_text(encoding="utf-8") if config.exists() else "",
        ),
    )


def generate_forcing_page(
    sim_dir: Path,
    output_path: Path | None = None,
    *,
    page: PageText | None = None,
    engine_ref: str | None = None,
    run_date: str | None = None,
) -> Path:
    """Write the forcing page for ``sim_dir``; return its path."""
    sim_dir = Path(sim_dir).resolve()
    output_path = Path(output_path) if output_path else sim_dir / "output" / "forcing_page.html"
    page = page or PageText()
    data = build_page_data(sim_dir, engine_ref=engine_ref, run_date=run_date)
    data["page"] = dict(
        title=page.title, lede=page.lede, what=page.what, limitations=page.limitations,
        credits=page.credits, reproduce=page.reproduce, tiles=page.tiles,
        logo=_data_uri(page.logo), logo_dark=_data_uri(page.logo_dark) or _data_uri(page.logo),
        logo_link=page.logo_link,
    )
    html = TEMPLATE.read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    # Neutralise "</" so page text can never close the <script> element early.
    html = html.replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8", newline="\n")
    return output_path
