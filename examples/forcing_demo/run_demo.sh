#!/usr/bin/env bash
# The Davos forcing demo from the command line, one step at a time.
#
#   examples/forcing_demo/run_demo.sh [SIM_DIR]     (default: ~/sim/davos_forcing_demo)
#
# 1. set up the simulation   DEM -> terrain (slope, aspect, horizon, sky view) -> 150 terrain units
# 2. fetch ERA5              surface + pressure levels for Oct 2023 - Jul 2024 (Google ARCO-ERA5, no account)
# 3. downscale               hourly forcing for every terrain unit -> output/forcing.nc
# 4. results                 output/forcing_page.html, the same page as demo.html, for your run
#
# Each step can be run on its own; `tps2 run` reuses what steps 1 and 2 cached, so re-running
# step 3 after changing the downscaling options in config.yaml takes seconds, not an hour.
# The ERA5 download in step 2 is most of the time (about an hour on a fast connection).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM="${1:-$HOME/sim/davos_forcing_demo}"
mkdir -p "$SIM"
[ -f "$SIM/config.yaml" ] || cp "$HERE/run/config.yaml" "$SIM/config.yaml"
cd "$SIM"
step() { printf '\n=== %s\n' "$*"; }

step "1/4  Set up the simulation: DEM, terrain analysis, terrain units"
tps2 setup --config config.yaml
tps2 info --config config.yaml

step "2/4  Fetch ERA5 (the long step; already-downloaded days are skipped)"
tps2 fetch-forcing --config config.yaml

step "3/4  Downscale: hourly forcing for every terrain unit"
tps2 run --config config.yaml

step "4/4  Results: the forcing page"
tps2 view . --page "$HERE/page.yaml" --save-daily
printf '\nDone. Open %s/output/forcing_page.html\n' "$SIM"
