# TopoPyScale 2

Topographic downscaling of climate reanalysis onto real mountain terrain.

TopoPyScale 2 (TPS2) takes ERA5 reanalysis (about 31&nbsp;km per grid cell) and gives hourly
forcing to the terrain it does not resolve, in either of two ways:

- **an area:** a 30–90&nbsp;m DEM is grouped into terrain units of similar elevation, slope,
  aspect and sky view, and each unit gets its own forcing;
- **named points:** weather stations or field sites, each with its own forcing.

```
ERA5 (~31 km) ──► TPS2 ──► hourly forcing per terrain unit or point (NetCDF / Zarr, CF units)
                   │
                   ├── Temperature, humidity   interpolated from the pressure levels to each unit
                   ├── Shortwave               slope, aspect, terrain shading, sky view
                   ├── Longwave                elevation and sky view
                   ├── Precipitation           split into rain and snow by wet-bulb temperature
                   └── Wind                    log profile to the unit's height above ground
```

TPS2 is a ground-up rewrite of [TopoPyScale](https://github.com/ArcticSnow/TopoPyScale)
(Filhol et al.), with the physics kernels in Python and Rust.

## See it without installing

```bash
open examples/forcing_demo/demo.html        # macOS  (xdg-open on Linux, start on Windows)
```

One season (October 2023 to July 2024) above Davos, Switzerland: one ERA5 grid cell, which sees
smooth ground at about 2000&nbsp;m, downscaled onto 150 terrain units between 1041 and
3097&nbsp;m. Pick a variable and press play; beside the map, each unit's value against its
elevation shows what the downscaling does. Everything is embedded; it works offline.

## Install

```bash
# A release wheel (Linux, macOS, Windows; Python 3.11–3.13; no Rust needed)
pip install topopyscale2-*.whl

# From source with conda (conda-forge provides GDAL/rasterio and the Rust toolchain)
conda env create -f environment.yml && conda activate tps2

# From source with pip (needs a Rust toolchain), or Docker
pip install .
docker build -t tps2 .
```

If the compiled kernels are ever missing, TPS2 says so once and uses its Python kernels: same
results, slower.

## Run

The demo, step by step (set up → fetch ERA5 → downscale → results page):

```bash
examples/forcing_demo/run_demo.sh
```

With a local web page, to pick the area on a map, set the downscaling options, run, and view the
result:

```bash
tps2 ui ~/sim/davos
```

Or from the command line:

```bash
tps2 init ~/sim/davos --bbox 9.70,46.72,9.98,46.88 --time 2023-10-01,2023-10-31
tps2 run  --config ~/sim/davos/config.yaml      # DEM → terrain units → ERA5 → downscaling
tps2 view ~/sim/davos                           # the forcing page for your run
```

ERA5 comes from Google's public archive by default, with no account needed. The forcing lands
in `output/forcing.nc` (or `.zarr`) with CF units on every variable.

| Command | What it does |
|---|---|
| `tps2 ui <dir>` | Local page: configure, run, view |
| `tps2 init <dir>` | New simulation directory with a template `config.yaml` |
| `tps2 run -c <config>` | The whole pipeline |
| `tps2 setup` / `fetch-forcing` | Terrain units only / ERA5 only |
| `tps2 view <dir>` | Self-contained HTML page of the downscaled forcing |
| `tps2 info` / `preflight` / `evaluate-clusters` | Inspect a domain, check a config, compare cluster counts |
| `tps2 build-cache` | Build a local ERA5 Zarr cache for a region |

## Point downscaling: stations and sites

Instead of an area, list named locations. Each point gets its own forcing, with slope, aspect,
horizon and sky view taken from the DEM around it:

```yaml
domain:
  spatial_mode: points
  points:
    coordinates:
      - {name: davos, lon: 9.8458, lat: 46.8130}
      - {name: weissfluhjoch, lon: 9.8094, lat: 46.8297, elevation: 2536}
```

```bash
tps2 run --config examples/points/config.yaml     # two Swiss sites, two days, under a minute
```

```python
import xarray as xr
ds = xr.open_dataset("output/forcing.nc")
ds["temperature"].sel(unit="weissfluhjoch")       # one unit per point, named after it
```

- `elevation` is optional: give a station's surveyed height, or leave it out to take it from the DEM.
- Points can be far apart: TPS2 fetches a DEM patch and ERA5 around each group of nearby points,
  not one bounding box covering them all.
- Use this mode to compare with station measurements. An area run compares a station with the
  terrain unit it falls in, whose mean elevation and exposure can differ from the station's.

The complete config is [`examples/points/config.yaml`](examples/points/config.yaml).

## Documentation

The documentation site is built from `docs/` (`pip install -r docs/requirements.txt && mkdocs
serve`): getting started, the downscaling algorithms, the data model, **known limitations with
numbers**, and a reference for every command and configuration key generated from the code.
Read [`docs/known_limitations.md`](docs/known_limitations.md) before relying on a result:
precipitation amount, diurnal temperature range and wind have measured weaknesses.

## Roadmap

This is a static snapshot of the downscaling engine. Development continues privately: snow models,
station validation, data assimilation, forecasts and climate scenarios exist
and will be released as they mature. See [`docs/roadmap.md`](docs/roadmap.md).

## Contributing

Releases are snapshots of a private development repository, so **pull requests cannot be
merged**. Bug reports, feature requests and questions are very welcome as
[issues](../../issues/new/choose); drafting one with an AI assistant is fine (point it at
[`AGENTS.md`](AGENTS.md)). See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing TPS2

TPS2 is developed by Joel Fiddes, [Mountain Futures](https://mountainfutures.ch), and Simon
Filhol, Météo-France. If you use it
in research, please cite it using [`CITATION.cff`](CITATION.cff) (GitHub's "Cite this
repository" button), and for research that builds substantially on TPS2, please consider
getting in touch about co-authorship.

## Acknowledgements

TPS2 builds on the original [TopoPyScale](https://github.com/ArcticSnow/TopoPyScale) by Simon
Filhol, Joel Fiddes and contributors. The methods are described in Fiddes & Gruber
(2014, [TopoSCALE](https://doi.org/10.5194/gmd-7-387-2014)), Fiddes & Gruber (2012,
[TopoSUB](https://doi.org/10.5194/gmd-5-1245-2012)) and Filhol et al. (2023,
[TopoPyScale](https://doi.org/10.21105/joss.05059)); please cite these alongside TPS2 (all are
listed in [`CITATION.cff`](CITATION.cff)).

## License

MIT; see [LICENSE](LICENSE). [NOTICE](NOTICE) records what the licence does not cover (logos and
trademarks) and the third-party components and data shipped with the example.
