"""Rust is a speed-up, not a requirement: missing Rust falls back to Python."""

import logging

from topopyscale2.core import dispatch


def test_rust_request_falls_back_to_python_when_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(dispatch, "available_backends", lambda: ["python"])
    monkeypatch.setattr(dispatch, "_WARNED_RUST_FALLBACK", False)
    with caplog.at_level(logging.WARNING, logger=dispatch.logger.name):
        d = dispatch.KernelDispatcher("rust")
        d2 = dispatch.KernelDispatcher("rust")
    assert d.backend_name == "python" and d.requested_backend == "rust"
    assert d2.backend_name == "python"
    # one warning per process, not per dispatcher (a run builds many)
    assert sum("Python kernels" in r.message for r in caplog.records) == 1
    assert d.temperature is not None


def test_rust_used_when_available(monkeypatch):
    monkeypatch.setattr(dispatch, "available_backends", lambda: ["python", "rust"])
    sentinel = {"temperature": object(), "radiation": object(), "precipitation": object(),
                "humidity": object(), "wind": object(), "redistribution": object(),
                "interpolation": object()}
    monkeypatch.setattr(dispatch, "get_backend", lambda name: sentinel if name == "rust" else None)
    d = dispatch.KernelDispatcher("rust")
    assert d.backend_name == "rust" and d.temperature is sentinel["temperature"]


def test_python_request_unchanged():
    d = dispatch.KernelDispatcher("python")
    assert d.backend_name == "python" and d.requested_backend == "python"
