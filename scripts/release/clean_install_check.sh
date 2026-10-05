#!/usr/bin/env bash
# Clean-install check: install a TPS2 tree into an EMPTY virtualenv using only its
# declared dependencies, then run the documented commands. Catches what every
# developer's environment hides — e.g. rioxarray was imported at run time but
# declared nowhere, so a fresh `pip install` crashed on the first regrid.
#
# Usage: scripts/release/clean_install_check.sh [TREE] [PYTHON]
#   TREE    a TPS2 source tree, e.g. the output of export_public.py (default: this repo)
#   PYTHON  interpreter to build the venv from (default: python3)
# Needs a Rust toolchain on PATH (maturin builds the kernels). Network is needed only
# for pip; the commands themselves run offline.
set -euo pipefail

TREE="$(cd "${1:-$(dirname "$0")/../..}" && pwd)"
PY="${2:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
step() { printf '\n== %s\n' "$*"; }

step "fresh virtualenv ($("$PY" --version 2>&1)) in $WORK/venv"
"$PY" -m venv "$WORK/venv"
# shellcheck disable=SC1091
. "$WORK/venv/bin/activate"
python -m pip install -q --upgrade pip

step "pip install $TREE (declared dependencies only, no extras)"
python -m pip install -q "$TREE"

cd "$WORK"   # run from outside the tree, so nothing is imported from the source dir

step "import every public module"
python - <<'EOF'
import importlib, pkgutil, sys
import topopyscale2
# Modules that belong to a declared optional extra (installed with pip install
# topopyscale2[<extra>]). Anything else that fails to import on a bare install is a
# missing dependency declaration.
skip = (
    "topopyscale2.ml",                        # [ml]
    "topopyscale2.core.jax_kernels",          # [jax]
    "topopyscale2.outputs.panel_dashboard",   # [explore]
    "topopyscale2.outputs.hydrology_dashboard",  # [explore]
)
bad = []
for m in pkgutil.walk_packages(topopyscale2.__path__, "topopyscale2."):
    if m.name.startswith(skip):
        continue
    try:
        importlib.import_module(m.name)
    except ImportError as e:            # a missing *declared* dependency surfaces here
        bad.append(f"{m.name}: {e}")
    except Exception:
        pass                            # import-time errors unrelated to packaging
if bad:
    print("modules that need an undeclared dependency:"); print("\n".join(bad)); sys.exit(1)
print("all modules import with the declared dependencies")
EOF

step "tps2 --help and every command's --help"
tps2 --help > /dev/null
# Command names from the CLI itself (click's public interface), not parsed help text.
cmds=$(python - <<'EOF2'
import click, typer
from topopyscale2.cli.main import app
root = typer.main.get_command(app)
ctx = click.Context(root)
for name in sorted(root.list_commands(ctx)):
    cmd = root.get_command(ctx, name)
    if hasattr(cmd, "list_commands"):          # a sub-group: check its commands too
        sub = click.Context(cmd, parent=ctx)
        print(name, end=" ")
        for s in sorted(cmd.list_commands(sub)):
            print(f"{name}:{s}", end=" ")
    else:
        print(name, end=" ")
EOF2
)
n=0
for c in $cmds; do
  tps2 ${c/:/ } --help > /dev/null || { echo "tps2 ${c/:/ } --help failed"; exit 1; }
  n=$((n + 1))
done
echo "ok: $n commands"

step "tps2 init (documented quick start)"
tps2 init "$WORK/sim" --bbox 9.70,46.72,9.98,46.88 --time 2023-10-01,2023-10-05 > /dev/null
test -f "$WORK/sim/config.yaml" && echo "ok: config.yaml written"

F="$TREE/examples/forcing_demo" D="$TREE/examples/davos_demo"
if [ -f "$F/page.yaml" ]; then
  step "rebuild the shipped forcing demo offline and compare byte for byte"
  tps2 view "$F/run" --page "$F/page.yaml" -o "$WORK/forcing_demo.html" > /dev/null
  cmp "$WORK/forcing_demo.html" "$F/demo.html" && echo "ok: forcing demo.html identical"
fi
if [ -f "$D/page.yaml" ] && tps2 snow-page --help > /dev/null 2>&1; then
  step "rebuild the shipped snow demo offline and compare byte for byte"
  tps2 snow-page "$D/run" --page "$D/page.yaml" \
      --stations "$D/obs_stations.csv" --obs "$D/obs_hs_daily.csv" -o "$WORK/demo.html" > /dev/null
  cmp "$WORK/demo.html" "$D/demo.html" && echo "ok: snow demo.html identical"
fi
if [ ! -f "$F/page.yaml" ] && [ ! -f "$D/page.yaml" ]; then
  echo "no shipped demo in this tree"; exit 1
fi

if [ "${TPS2_CHECK_ONLINE:-0}" = 1 ]; then
  # Everything above runs offline. This fetches real ERA5 the way a first-time user does
  # (default backend, declared dependencies only): v0.1.0 shipped without gcsfs and every
  # documented first run failed at the download, which no offline check could see.
  step "online: point downscaling with real ERA5 (examples/points, default backend)"
  P="$TREE/examples/points/config.yaml"
  [ -f "$P" ] || { echo "no examples/points/config.yaml in this tree"; exit 1; }
  mkdir -p "$WORK/points" && cp "$P" "$WORK/points/config.yaml"
  ( cd "$WORK/points" && NO_COLOR=1 tps2 run --config config.yaml > run.log 2>&1 ) \
    || { tail -40 "$WORK/points/run.log"; echo "the documented online run fails on a clean install"; exit 1; }
  python - "$WORK/points/output/forcing.nc" <<'EOF2'
import sys, xarray as xr
ds = xr.open_dataset(sys.argv[1])
names = [str(u) for u in ds["unit"].values]
assert names == ["davos", "weissfluhjoch"], names
t = (ds["temperature"] - 273.15).mean("time").values
assert t[0] > t[1], f"Davos (1560 m) should be warmer than Weissfluhjoch (2536 m): {t}"
print(f"ok: {len(names)} points, {ds.sizes['time']} hours, mean T {t.round(1)} degC")
EOF2
fi

step "the installed package carries the compiled Rust kernels"
python -c "from topopyscale2.core.dispatch import available_backends as a; b = a(); print(b); assert 'rust' in b" \
    || { echo "the install has no Rust kernels"; exit 1; }

step "the tree's own test suite, against the INSTALLED package"
# pytest is the only addition to the declared dependencies. The console script (not
# `python -m pytest`) keeps the source dir off sys.path, and --import-mode=append puts
# the tree's test packages after site-packages; TPS2_EXPECT_INSTALLED makes
# tests/conftest.py refuse to run if topopyscale2 still resolves to the source tree.
python -m pip install -q pytest
if ! ( cd "$TREE" && TPS2_EXPECT_INSTALLED=1 TPS2_FSM2_NO_COMPILE=1 NO_COLOR=1 COLUMNS=200 \
        "$WORK/venv/bin/pytest" tests -q -rs -p no:cacheprovider --import-mode=append ); then
  echo "the shipped tests fail on a clean install"; exit 1
fi

printf '\nCLEAN INSTALL CHECK PASSED for %s\n' "$TREE"
