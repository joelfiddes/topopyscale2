# Get started

## See it without installing

The repository ships a finished run as a single page: water year 2024 (October 2023 to
July 2024) above Davos, Switzerland. One ERA5 grid cell, which sees smooth ground at about
2000&nbsp;m, is downscaled onto 150 terrain units between 1041 and 3097&nbsp;m.

```bash
open examples/forcing_demo/demo.html        # macOS
xdg-open examples/forcing_demo/demo.html    # Linux
start examples\forcing_demo\demo.html      # Windows (PowerShell or cmd)
```

Everything is embedded, so it works offline. Pick a variable and press play; the panel beside
the map shows each unit's value against its elevation for the day on screen.

## Install

=== "From a release (no Rust needed)"

    Each release has prebuilt wheels for Linux and macOS (x86_64 and arm64) and Windows
    (x86_64), for Python 3.11–3.13, with the Rust kernels compiled in. Download the one for your platform from
    the release page, then:

    ```bash
    pip install topopyscale2-*.whl
    ```

=== "conda (from source)"

    conda-forge provides the geospatial stack (GDAL, rasterio) and the Rust toolchain:

    ```bash
    conda env create -f environment.yml
    conda activate tps2
    ```

=== "pip (from source)"

    Needs a Rust toolchain on `PATH`; maturin compiles the kernels during the install.

    ```bash
    pip install .
    ```

=== "Docker"

    ```bash
    docker build -t tps2 .
    ```

    Usage examples are in the header of the `Dockerfile`.

Check the install:

```bash
tps2 --version
tps2 preflight --config path/to/config.yaml   # optional: checks a config before a run
```

!!! note "Rust is a speed-up, not a requirement"
    If the compiled kernels are ever missing, TPS2 says so once and uses its Python kernels.
    The results are the same; the Rust kernels are faster.

## Run the demo yourself, step by step

```bash
examples/forcing_demo/run_demo.sh      # into ~/sim/davos_forcing_demo
```

| Step | Command |
|---|---|
| 1. Set up the simulation: DEM, terrain analysis, terrain units | `tps2 setup --config config.yaml` |
| 2. Fetch ERA5 (the long step, about an hour) | `tps2 fetch-forcing --config config.yaml` |
| 3. Downscale: hourly forcing for every unit | `tps2 run --config config.yaml` |
| 4. Results: the forcing page | `tps2 view . --save-daily` |

On Windows, run the script from Git Bash or WSL, or type the four commands into PowerShell.
`tps2 run` reuses what steps 1 and 2 cached, so after changing the downscaling options you
re-run step 3 in seconds.

## Run a first simulation: the web page

`tps2 ui` opens a local page for one simulation directory:

```bash
tps2 ui ~/sim/davos
```

1. **Area:** click two corners on the map, or type the bounding box.
2. **Period and options:** dates, DEM, number of terrain units, output format, and where
   ERA5 comes from.
3. **Downscaling:** full or simple mode, pressure levels or lapse rate, the precipitation
   gradient (off by default), how rain and snow are split, and the wind method. Options
   that do not apply to your choices are greyed out. The page lists what it does not set;
   edit `config.yaml` for those and the page keeps them.
4. **Run:** starts `tps2 run` and streams its log; **View the result** opens the forcing page.

The page listens on `127.0.0.1` only and has no login; stop it with Ctrl+C.

## Run a first simulation: the command line

```bash
# 1. Create a simulation directory with a template config
tps2 init ~/sim/davos --bbox 9.70,46.72,9.98,46.88 --time 2023-10-01,2023-10-31

# 2. Review the config: domain, time range, number of terrain units, output format
$EDITOR ~/sim/davos/config.yaml

# 3. Run: DEM → terrain units → ERA5 download → downscaling → output
tps2 run --config ~/sim/davos/config.yaml

# 4. Look at it
tps2 view ~/sim/davos
```

The default `google` backend reads ERA5 from Google's public archive and needs no account.
The forcing is written to `output/forcing.nc` (or `.zarr`) with CF units on every variable:

```python
import xarray as xr

ds = xr.open_dataset("~/sim/davos/output/forcing.nc")
ds["temperature"].attrs["units"]          # "K"
ds["precipitation"].attrs["units"]        # "kg m-2", an amount per time step (hourly)
(ds["temperature"].isel(unit=0) - 273.15).plot()
```

## Export to model formats

`tps2 run` writes NetCDF (or Zarr). To drive an impact model, export the run to its input
format; the files land in `output/<format>/`:

```bash
tps2 export ~/sim/davos --format smet     # SNOWPACK / Alpine3D / MeteoIO
```

| `--format` | For | Files |
|---|---|---|
| `smet` | SNOWPACK, Alpine3D, MeteoIO | one `.smet` per unit |
| `fsm` | FSM (single point) | one `.txt` per unit |
| `fsm2` | FSM2 (multi-point) | one `met_input.txt` |
| `csv` | spreadsheets, R, pandas | one `.csv` per unit |
| `crocus` | Crocus / SURFEX | one `FORCING`-style `.nc` per unit |
| `cryogrid` | CryoGrid | one `.nc` per unit |
| `hbv` | HBV-style hydrological models | daily temperature, precipitation and PET (Hamon) |

Units are converted to what each model reads: precipitation to a rate or a total as the
format expects, relative humidity to %. Each unit's position and elevation come from the
forcing file, so an export needs nothing else from the run.

## Downscale to points: stations and sites

Instead of an area, give named locations. Each point gets its own forcing, with slope, aspect,
horizon and sky view taken from the DEM around it:

```yaml
domain:
  spatial_mode: points
  points:
    coordinates:
      - {name: davos, lon: 9.8458, lat: 46.8130}
      - {name: weissfluhjoch, lon: 9.8094, lat: 46.8297, elevation: 2536}
```

`elevation` is optional: give a station's surveyed height, or leave it out to take it from the
DEM. Points can be far apart: TPS2 fetches a DEM patch and ERA5 around each group of nearby
points, not one bounding box. A complete config is in `examples/points/config.yaml`:

```bash
tps2 run --config examples/points/config.yaml     # two Swiss sites, two days, a few minutes
```

The output has one `unit` per point, named after it:
`ds["temperature"].sel(unit="weissfluhjoch")`. This is the mode to use for comparing with
station measurements, since an area run compares a station with the terrain unit it falls in,
whose mean elevation can differ from the station's.

## Rebuild the demo page

The page is built from inputs that ship with it (556&nbsp;KB), so it rebuilds offline in
seconds:

```bash
tps2 view examples/forcing_demo/run --page examples/forcing_demo/page.yaml \
    -o examples/forcing_demo/demo.html
```

A test checks that this reproduces the committed page byte for byte, so the demo cannot
quietly drift away from the engine. `examples/forcing_demo/README.md` explains how to
repeat the whole run.

## Next

| You want to... | Read |
|---|---|
| Every configuration key | [Configuration reference](../reference/config.md) |
| Every command and option | [CLI reference](../reference/cli.md) |
| What the downscaling does | [Downscaling algorithms](../downscaling_algorithms.md) |
| Where it is wrong, and by how much | [Known limitations](../known_limitations.md) |
| What comes next | [Roadmap](../roadmap.md) |
