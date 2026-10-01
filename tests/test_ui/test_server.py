"""`tps2 ui` server: loopback only, token-guarded, writes a valid config, runs, views."""

import http.client
import json
import sys
import threading
import time

import pytest
import yaml

from tests.test_outputs.test_forcing_page import sim_dir as forcing_sim_dir  # noqa: F401  (fixture)
from topopyscale2.config.schema import TPS2Config
from topopyscale2.ui.server import make_server

FORM = {
    "domain.bbox": [9.70, 46.72, 9.98, 46.88], "domain.dem_source": "glo_90",
    "inputs.time_range": ["2023-10-01", "2023-10-05"], "inputs.backend": "google",
    "inputs.cache_path": "", "clustering.n_clusters": 20,
    "downscaling.mode": "full", "output.format": "netcdf",
}


def fake_run(seconds: float = 0.0) -> list[str]:
    """A stand-in for `tps2 run --config`: prints its argument, optionally sleeps."""
    return [sys.executable, "-c",
            f"import sys, time; print('fake run', sys.argv[-1], flush=True); time.sleep({seconds})"]


@pytest.fixture
def ui(tmp_path):
    def start(sim_dir=None, seconds=0.0):
        server, state = make_server(sim_dir or tmp_path / "sim", port=0, run_command=fake_run(seconds))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        return server.server_address[1], state
    started = []
    yield start
    for s in started:
        s.shutdown()
        s.server_close()


def req(port, method, path, body=None, token=None, host="127.0.0.1"):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    headers = {"Host": f"{host}:{port}", "Content-Type": "application/json"}
    if token:
        headers["X-TPS2-Token"] = token
    c.request(method, path, body=None if body is None else json.dumps(body), headers=headers)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except json.JSONDecodeError:
        return r.status, data.decode()


def test_refuses_to_bind_beyond_loopback(tmp_path):
    with pytest.raises(ValueError, match="loopback"):
        make_server(tmp_path, host="0.0.0.0", port=0)


def test_page_embeds_the_session_token(ui):
    port, state = ui()
    status, html = req(port, "GET", "/")
    assert status == 200 and state.token in html and "__TOKEN__" not in html


def test_foreign_host_header_is_refused(ui):
    port, _ = ui()
    assert req(port, "GET", "/api/state", host="evil.example")[0] == 403


def test_post_without_the_token_is_refused(ui):
    port, state = ui()
    assert req(port, "POST", "/api/config", FORM)[0] == 403
    assert req(port, "POST", "/api/config", FORM, token="wrong")[0] == 403
    assert not state.config_path.exists()


def test_writes_a_valid_config_and_keeps_other_keys(ui):
    port, state = ui()
    status, body = req(port, "POST", "/api/config", FORM, token=state.token)
    assert status == 200, body
    cfg = TPS2Config.from_yaml(state.config_path)
    assert cfg.domain.bbox == FORM["domain.bbox"] and cfg.clustering.n_clusters == 20
    # an unmanaged key survives a rewrite, and the previous file is backed up
    raw = yaml.safe_load(state.config_path.read_text())
    raw["inputs"]["pressure_levels"] = [500, 700]
    state.config_path.write_text(yaml.safe_dump(raw))
    assert req(port, "POST", "/api/config", {**FORM, "clustering.n_clusters": 30}, token=state.token)[0] == 200
    again = yaml.safe_load(state.config_path.read_text())
    assert again["inputs"]["pressure_levels"] == [500, 700] and again["clustering"]["n_clusters"] == 30
    assert state.config_path.with_suffix(".yaml.bak").exists()
    st = req(port, "GET", "/api/state")[1]
    assert st["config_exists"] and st["values"]["clustering.n_clusters"] == 30


@pytest.mark.parametrize("bad,match", [
    ({"domain.bbox": [10, 46, 9, 47]}, "west < east"),
    ({"domain.bbox": [1, 2, 3]}, "four numbers"),
    ({"inputs.time_range": ["2024-02-01", "2024-01-01"]}, "start first"),
    ({"inputs.backend": "local", "inputs.cache_path": ""}, "Zarr store"),
])
def test_invalid_forms_are_rejected_with_a_reason(ui, bad, match):
    port, state = ui()
    status, body = req(port, "POST", "/api/config", {**FORM, **bad}, token=state.token)
    assert status == 400 and match in body["error"]
    assert not state.config_path.exists()


def test_run_streams_its_log_and_finishes(ui):
    port, state = ui()
    assert req(port, "POST", "/api/run", {}, token=state.token)[0] == 400   # no config yet
    req(port, "POST", "/api/config", FORM, token=state.token)
    assert req(port, "POST", "/api/run", {}, token=state.token)[0] == 200
    deadline, text, offset = time.time() + 20, "", 0
    while time.time() < deadline:
        r = req(port, "GET", f"/api/log?offset={offset}")[1]
        text += r["text"]
        offset = r["offset"]
        if not r["running"]:
            break
        time.sleep(0.1)
    assert r["returncode"] == 0 and "fake run" in text and str(state.config_path) in text


def test_only_one_run_at_a_time(ui):
    port, state = ui(seconds=3)
    req(port, "POST", "/api/config", FORM, token=state.token)
    assert req(port, "POST", "/api/run", {}, token=state.token)[0] == 200
    status, body = req(port, "POST", "/api/run", {}, token=state.token)
    assert status == 400 and "already" in body["error"]
    state.proc.kill()


def test_view_needs_forcing_then_serves_the_page(ui, tmp_path, forcing_sim_dir):  # noqa: F811
    port, _ = ui(sim_dir=tmp_path / "empty")
    assert req(port, "GET", "/view")[0] == 404
    port, _ = ui(sim_dir=forcing_sim_dir)
    status, html = req(port, "GET", "/view")
    assert status == 200 and "const D = {" in html


DOWNSCALING = {
    "downscaling.mode": "simple", "downscaling.lapse_rate": 0.0058,
    "inputs.pressure_levels": "1000, 300, 500,700, 850",
    "downscaling.precipitation": "elevation_gradient", "downscaling.precip_gradient": 0.0002,
    "downscaling.phase_method": "air_temperature",
    "downscaling.t_snow_threshold_c": 0.0, "downscaling.t_rain_threshold_c": 2.0,
    "downscaling.t_air_max_c": 5.0, "downscaling.wind_config.method": "winstral",
}


def test_downscaling_options_are_written_and_validated(ui):
    port, state = ui()
    status, body = req(port, "POST", "/api/config", {**FORM, **DOWNSCALING}, token=state.token)
    assert status == 200, body
    d = TPS2Config.from_yaml(state.config_path).downscaling
    assert (d.mode, d.lapse_rate, d.precipitation, d.precip_gradient) == ("simple", 0.0058, "elevation_gradient", 0.0002)
    assert (d.phase_method, d.t_snow_threshold_c, d.t_rain_threshold_c, d.t_air_max_c) == ("air_temperature", 0.0, 2.0, 5.0)
    assert d.wind_config.method == "winstral"
    assert TPS2Config.from_yaml(state.config_path).inputs.pressure_levels == [300, 500, 700, 850, 1000]
    values = req(port, "GET", "/api/state")[1]["values"]
    assert values["downscaling.wind_config.method"] == "winstral"


@pytest.mark.parametrize("bad,match", [
    ({"inputs.pressure_levels": "500, abc"}, "whole numbers"),
    ({"inputs.pressure_levels": "500, 1200"}, "between 1 and 1000"),
    ({"downscaling.t_snow_threshold_c": 3, "downscaling.t_rain_threshold_c": 1}, "below the all-rain"),
    ({"downscaling.lapse_rate": 0.5}, "lapse rate"),
    ({"downscaling.phase_method": "magic"}, "wet_bulb or air_temperature"),
    ({"downscaling.wind_config.method": "gust"}, "log_profile or winstral"),
])
def test_bad_downscaling_options_are_rejected(ui, bad, match):
    port, state = ui()
    status, body = req(port, "POST", "/api/config", {**FORM, **bad}, token=state.token)
    assert status == 400 and match in body["error"], body


def _wait_done(port, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = req(port, "GET", "/api/log?offset=0")[1]
        if not r["running"]:
            return r
        time.sleep(0.1)
    raise AssertionError("run did not finish")


def test_status_finished_and_failed(tmp_path):
    for code, expected in ((0, "finished"), (3, "failed")):
        server, state = make_server(tmp_path / f"s{code}", port=0,
                                    run_command=[sys.executable, "-c", f"import sys; sys.exit({code})"])
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.server_address[1]
        try:
            req(port, "POST", "/api/config", FORM, token=state.token)
            req(port, "POST", "/api/run", {}, token=state.token)
            r = _wait_done(port)
            assert r["status"] == expected and r["returncode"] == code
        finally:
            server.shutdown()
            server.server_close()


def test_status_stopped_when_killed(ui):
    port, state = ui(seconds=30)
    req(port, "POST", "/api/config", FORM, token=state.token)
    req(port, "POST", "/api/run", {}, token=state.token)
    state.proc.kill()
    state.proc.wait()
    assert _wait_done(port)["status"] == "stopped"


@pytest.mark.parametrize("log,expected", [
    ("Step 2: Fetch Forcing\nDownloading ERA5 ...\n", "interrupted"),
    ("...\nPipeline complete in 43.4s\n", "finished"),
])
def test_status_after_a_server_restart_comes_from_the_log(ui, tmp_path, log, expected):
    sim = tmp_path / "restarted"
    sim.mkdir()
    (sim / "ui_run.log").write_text(log)
    port, _ = ui(sim_dir=sim)
    st = req(port, "GET", "/api/state")[1]
    assert st["run_status"] == expected and not st["running"] and st["log_size"] == len(log)
    assert req(port, "GET", "/api/log?offset=0")[1]["status"] == expected


def test_status_none_before_any_run(ui):
    port, _ = ui()
    assert req(port, "GET", "/api/state")[1]["run_status"] == "none"
