"""``tps2 ui``: a local page to configure a run, start it and view the result.

A small standard-library HTTP server for ONE simulation directory, chosen on the command
line (the browser never names a path). It binds to the loopback interface only and has no
login, so it guards against the two ways another site could still reach it: every POST
needs the session token embedded in the page, and requests whose Host header is not a
loopback name are refused (DNS rebinding).

Endpoints (all JSON unless noted):

- ``GET /`` the app page (HTML)
- ``GET /api/state`` the managed config values, whether a run is going, what exists
- ``POST /api/config`` validate the form against the config schema and write config.yaml
- ``POST /api/run`` start ``tps2 run`` for the config (one run at a time)
- ``GET /api/log?offset=N`` run log from byte N, and whether the run has finished
- ``GET /view`` the forcing page for the finished run (HTML, built by ``tps2 view``)
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

APP_HTML = Path(__file__).with_name("app.html")
LOGO_SVG = Path(__file__).with_name("logo.svg")   # Mountain Futures, links to mountainfutures.ch
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
LOG_NAME = "ui_run.log"

# The keys the form edits, as dotted paths into config.yaml. Everything else is kept as is.
MANAGED = [
    "domain.bbox", "domain.dem_source",
    "inputs.time_range", "inputs.backend", "inputs.cache_path", "inputs.pressure_levels",
    "clustering.n_clusters",
    "downscaling.mode", "downscaling.lapse_rate",
    "downscaling.precipitation", "downscaling.precip_gradient",
    "downscaling.phase_method", "downscaling.t_snow_threshold_c",
    "downscaling.t_rain_threshold_c", "downscaling.t_air_max_c",
    "downscaling.wind_config.method",
    "output.format",
]
NOT_IN_FORM = (
    "clustering features, surface types, terrain/horizon settings, wind roughness, "
    "points or polygon modes, and output variables"
)


class UIState:
    """Everything the handlers share: the simulation directory, the token, the run."""

    def __init__(self, sim_dir: Path, run_command: list[str] | None = None):
        self.sim_dir = Path(sim_dir).resolve()
        self.token = secrets.token_urlsafe(24)
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        # Overridable for tests; by default this Python runs the TPS2 CLI.
        self.run_command = run_command or [sys.executable, "-m", "topopyscale2.cli.main", "run", "--config"]

    @property
    def config_path(self) -> Path:
        return self.sim_dir / "config.yaml"

    @property
    def log_path(self) -> Path:
        return self.sim_dir / LOG_NAME

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


# ---------------------------------------------------------------- config


def _base_config(state: UIState) -> dict:
    """The existing config.yaml, else the `tps2 init` template, as a dict."""
    import yaml

    if state.config_path.exists():
        return yaml.safe_load(state.config_path.read_text(encoding="utf-8")) or {}
    from topopyscale2.cli.main import _CONFIG_TEMPLATE

    return yaml.safe_load(_CONFIG_TEMPLATE)


def _get(cfg: dict, path: str):
    node = cfg
    for part in path.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(part)
    return node


def _set(cfg: dict, path: str, value) -> None:
    *parents, leaf = path.split(".")
    node = cfg
    for part in parents:
        node = node.setdefault(part, {})
    node[leaf] = value


def managed_values(cfg: dict) -> dict:
    return {path: _get(cfg, path) for path in MANAGED}


def _number(form: dict, key: str, default: float, lo: float, hi: float, what: str) -> float:
    raw = form.get(key)
    try:
        v = float(default if raw in (None, "") else raw)
    except (TypeError, ValueError):
        raise ValueError(f"{what} must be a number") from None
    if not lo <= v <= hi:
        raise ValueError(f"{what} must be between {lo} and {hi}")
    return v


def apply_form(cfg: dict, form: dict) -> dict:
    """``cfg`` with the managed keys set from ``form`` (validated), others untouched."""
    import copy

    out = copy.deepcopy(cfg)
    try:
        bbox = [float(x) for x in form["domain.bbox"]]
        if len(bbox) != 4:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise ValueError("bbox must be four numbers: west, south, east, north") from None
    w, s, e, n = bbox
    if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
        raise ValueError("bbox must satisfy west < east and south < north, in degrees")
    times = form.get("inputs.time_range") or []
    if len(times) != 2 or not all(times) or times[0] > times[1]:
        raise ValueError("time range needs a start and an end date, start first")
    backend = form.get("inputs.backend") or "google"

    levels_raw = form.get("inputs.pressure_levels")
    if levels_raw in (None, "", []):
        levels = None
    else:
        try:
            items = levels_raw if isinstance(levels_raw, list) else str(levels_raw).replace(" ", "").split(",")
            levels = sorted({int(x) for x in items if str(x) != ""})
        except ValueError:
            raise ValueError("pressure levels must be whole numbers in hPa, e.g. 300, 500, 700, 850, 1000") from None
        if not levels or not all(1 <= p <= 1000 for p in levels):
            raise ValueError("pressure levels must be between 1 and 1000 hPa")

    precipitation = form.get("downscaling.precipitation") or "none"
    if precipitation not in ("none", "elevation_gradient"):
        raise ValueError("precipitation must be none or elevation_gradient")
    phase = form.get("downscaling.phase_method") or "wet_bulb"
    if phase not in ("wet_bulb", "air_temperature"):
        raise ValueError("rain/snow method must be wet_bulb or air_temperature")
    t_snow = _number(form, "downscaling.t_snow_threshold_c", -0.5, -10, 10, "all-snow threshold")
    t_rain = _number(form, "downscaling.t_rain_threshold_c", 2.5, -10, 10, "all-rain threshold")
    if t_snow >= t_rain:
        raise ValueError("the all-snow threshold must be below the all-rain threshold")
    wind = form.get("downscaling.wind_config.method") or "log_profile"
    if wind not in ("log_profile", "winstral"):
        raise ValueError("wind method must be log_profile or winstral")

    values = {
        "domain.bbox": bbox,
        "domain.dem_source": form.get("domain.dem_source") or "glo_90",
        "inputs.time_range": [str(times[0]), str(times[1])],
        "inputs.backend": backend,
        "clustering.n_clusters": int(form.get("clustering.n_clusters") or 50),
        "downscaling.mode": form.get("downscaling.mode") or "full",
        "downscaling.lapse_rate": _number(form, "downscaling.lapse_rate", 0.0065, -0.02, 0.02, "lapse rate (K/m)"),
        "downscaling.precipitation": precipitation,
        "downscaling.precip_gradient": _number(form, "downscaling.precip_gradient", 0.0003, 0, 0.002,
                                               "precipitation gradient (per metre)"),
        "downscaling.phase_method": phase,
        "downscaling.t_snow_threshold_c": t_snow,
        "downscaling.t_rain_threshold_c": t_rain,
        "downscaling.t_air_max_c": _number(form, "downscaling.t_air_max_c", 4.0, -10, 99, "all-rain air temperature"),
        "downscaling.wind_config.method": wind,
        "output.format": form.get("output.format") or "netcdf",
    }
    if levels is not None:
        values["inputs.pressure_levels"] = levels
    if backend == "local":
        if not form.get("inputs.cache_path"):
            raise ValueError("the local backend needs the path of an ERA5 Zarr store")
        values["inputs.cache_path"] = str(form["inputs.cache_path"])
    elif "inputs" in out:
        out["inputs"].pop("cache_path", None)
    for path, v in values.items():
        _set(out, path, v)
    return out


def write_config(state: UIState, form: dict) -> Path:
    """Validate the form with the real schema, back up any existing file, write config.yaml."""
    import yaml

    from topopyscale2.config.schema import TPS2Config

    cfg = apply_form(_base_config(state), form)
    TPS2Config.model_validate(cfg)            # raises with the schema's own message
    state.sim_dir.mkdir(parents=True, exist_ok=True)
    if state.config_path.exists():
        shutil.copy2(state.config_path, state.config_path.with_suffix(".yaml.bak"))
    header = ("# TPS2 configuration, written by `tps2 ui`. Keys the page does not show are kept;\n"
              "# comments are not (the previous file is config.yaml.bak).\n")
    state.config_path.write_text(header + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return state.config_path


# ---------------------------------------------------------------- run


def start_run(state: UIState) -> None:
    with state.lock:
        if state.running():
            raise RuntimeError("a run is already going")
        if not state.config_path.exists():
            raise RuntimeError("write the configuration first")
        log = open(state.log_path, "wb")   # noqa: SIM115  (owned by the child process)
        state.proc = subprocess.Popen(
            [*state.run_command, str(state.config_path)], cwd=state.sim_dir,
            stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env={**os.environ, "NO_COLOR": "1", "COLUMNS": "120"},
        )
        log.close()


def read_log(state: UIState, offset: int) -> dict:
    data = b""
    if state.log_path.exists():
        with open(state.log_path, "rb") as f:
            f.seek(max(0, offset))
            data = f.read(256_000)
    rc = None if state.proc is None else state.proc.poll()
    return {"text": data.decode("utf-8", errors="replace"), "offset": max(0, offset) + len(data),
            "running": state.running(), "returncode": rc}


def forcing_available(state: UIState) -> bool:
    out = state.sim_dir / "output"
    return any((out / n).exists() for n in ("forcing_daily.nc", "forcing.nc", "forcing.zarr"))


# ---------------------------------------------------------------- HTTP


def make_handler(state: UIState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "tps2-ui"

        def log_message(self, fmt, *args):  # keep the terminal quiet
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
            return host in LOOPBACK_HOSTS

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj: dict) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "not a loopback host"})
            url = urlsplit(self.path)
            if url.path == "/":
                html = APP_HTML.read_text(encoding="utf-8").replace("__TOKEN__", state.token)
                return self._send(HTTPStatus.OK, html.encode(), "text/html; charset=utf-8")
            if url.path == "/logo.svg":
                return self._send(HTTPStatus.OK, LOGO_SVG.read_bytes(), "image/svg+xml")
            if url.path == "/api/state":
                cfg = _base_config(state)
                return self._json(HTTPStatus.OK, {
                    "sim_dir": str(state.sim_dir), "config_exists": state.config_path.exists(),
                    "values": managed_values(cfg), "not_in_form": NOT_IN_FORM,
                    "running": state.running(), "forcing": forcing_available(state),
                })
            if url.path == "/api/log":
                q = parse_qs(url.query)
                return self._json(HTTPStatus.OK, read_log(state, int((q.get("offset") or ["0"])[0])))
            if url.path == "/view":
                if not forcing_available(state):
                    return self._json(HTTPStatus.NOT_FOUND, {"error": "no forcing yet: run first"})
                from topopyscale2.outputs.forcing_page import generate_forcing_page

                page = generate_forcing_page(state.sim_dir)
                return self._send(HTTPStatus.OK, page.read_bytes(), "text/html; charset=utf-8")
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "not a loopback host"})
            if not secrets.compare_digest(self.headers.get("X-TPS2-Token") or "", state.token):
                return self._json(HTTPStatus.FORBIDDEN, {"error": "missing or wrong session token"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                return self._json(HTTPStatus.BAD_REQUEST, {"error": "body is not JSON"})
            path = urlsplit(self.path).path
            try:
                if path == "/api/config":
                    p = write_config(state, body)
                    return self._json(HTTPStatus.OK, {"written": str(p)})
                if path == "/api/run":
                    start_run(state)
                    return self._json(HTTPStatus.OK, {"started": True})
            except (ValueError, RuntimeError) as e:   # pydantic's ValidationError is a ValueError
                return self._json(HTTPStatus.BAD_REQUEST, {"error": str(e)})
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    return Handler


def make_server(sim_dir: Path, host: str = "127.0.0.1", port: int = 8765,
                run_command: list[str] | None = None) -> tuple[ThreadingHTTPServer, UIState]:
    """Build (not start) the server; refuses any host that is not loopback."""
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"tps2 ui serves on the loopback interface only (got {host!r}); it has no login")
    state = UIState(sim_dir, run_command=run_command)
    server = ThreadingHTTPServer((host, port), make_handler(state))
    return server, state
