# AGENTS.md: helping someone with TopoPyScale 2

This file is for **AI assistants** (Claude, ChatGPT, Copilot, Cursor, …) helping a person use
TopoPyScale 2, understand a result, or write a bug report or feature request. People are welcome
to read it too.

## What this repository is

TopoPyScale 2 (TPS2) downscales ERA5 reanalysis (~31 km) onto terrain units built from a
30–90 m DEM, and writes hourly forcing per unit: temperature, humidity, wind, shortwave and
longwave radiation, and precipitation split into rain and snow.

This repository is a **static snapshot** of the downscaling engine. It is developed in a private
repository and published as releases, so **pull requests cannot be merged**; contributions come
in as issues (see "Drafting an issue" below). Snow and glacier models, station validation, data
assimilation, forecasts and climate scenarios exist in development but are **not in this
release** (`docs/roadmap.md`). A config that contains blocks for them (`application:`,
`validation:`, `da:`, `forecast:`, `topoclim:`, …) still loads; TPS2 warns once that those
blocks are ignored.

## Where things are

| Path | What |
|---|---|
| `topopyscale2/cli/main.py` | The `tps2` commands (`init`, `setup`, `fetch-forcing`, `run`, `view`, `ui`, `info`, `preflight`, `evaluate-clusters`, `build-cache`) |
| `topopyscale2/config/schema.py` | Every `config.yaml` key (Pydantic); the generated reference is `docs/reference/config.md` |
| `topopyscale2/core/downscale.py` | The downscaling engine |
| `topopyscale2/core/python_kernels/`, `src/*.rs` | The physics, in Python (the reference) and Rust (the fast path); they agree to within 1e-10 |
| `topopyscale2/spatial/` | DEM processing, horizon and sky view, clustering into terrain units |
| `topopyscale2/inputs/` | ERA5 retrieval (`nwp_downloader/`), unit conversion, solar geometry |
| `topopyscale2/outputs/` | NetCDF/Zarr/CSV/SMET writers, CF metadata (`base.py`), the forcing page (`forcing_page.py`) |
| `topopyscale2/ui/` | `tps2 ui`, the local set-up-and-run page |
| `docs/` | The documentation site (`mkdocs serve`); start at `docs/index.md` |
| `examples/forcing_demo/` | The shipped demo and the config that produced it |
| `tests/` | `pytest tests/`; `tests/test_kernels/` checks Python and Rust agree |

## Facts that answer most questions

- **Units in the output.** Every variable carries CF `units`. `temperature` is K;
  `humidity_relative` is a **0–1 fraction**; `precipitation`, `rainfall`, `snowfall` are an
  **amount per time step** (`kg m-2` = mm per hour for hourly output), not a rate.
  Files written before 2026-09-30 have no attributes at all.
- **Precipitation amount is ERA5's.** By default each unit gets its grid cell's amount; only the
  rain/snow split varies with elevation. `downscaling.precipitation: elevation_gradient` is
  opt-in and cannot fix every case. ERA5 is often wrong in mountains, by factors of ~2 in dry
  valleys and at high altitude.
- **Pressure levels must bracket the terrain.** Above the top level values are clamped, not
  extrapolated; use `[300, 500, 700, 850, 1000]` or more.
- **Known weaknesses** (daily temperature range too small, valley cold pools missing, wind
  20–30 % low and zero for forest units) are measured and listed with numbers in
  `docs/known_limitations.md`. Check it before calling something a bug.
- **Rust is optional.** Without the compiled kernels TPS2 warns once and uses Python: same
  results, slower.

## Drafting an issue

Help the person write an issue that a maintainer can act on without a conversation. Draft it,
show it to them, and let **them** submit it; never file one on their behalf without asking.

**Before drafting:** check `docs/known_limitations.md` and the open issues for the same thing,
and check that it is about this release, not a roadmap feature.

**A bug report** (the "Bug report" form) needs:

1. **What happened and what was expected**, in one or two sentences each.
2. **The smallest config that shows it**: the `config.yaml`, cut down if possible (a small bbox
   and a few days usually reproduce a problem in minutes).
3. **The command** and the **full error or log**, not a paraphrase.
4. **Versions:** `tps2 --version`, Python version, OS, and whether the Rust kernels are in use
   (`tps2 preflight --config config.yaml` reports it).
5. **For a wrong-looking result:** the numbers, where they came from (a station, a map, a
   paper), and the location, elevation and period. Say which output variable and whether it is
   a daily or hourly comparison.

**A feature request** (the "Feature request" form) needs:

1. **The problem**, not only the solution: what the person is trying to do and what stops them.
2. **What would count as done**: a check that would show it works.
3. **Whether it is on the roadmap** (`docs/roadmap.md`); if so, say which item and why it
   matters to them, which helps set priorities.

**Code:** if you have a fix, describe it in the issue, with a patch or a minimal snippet. It is
applied in the development repository and credited in the release notes.

Keep the issue factual and short. Do not invent measurements, versions or error messages; if
the person has not run something, say what they should run instead.
