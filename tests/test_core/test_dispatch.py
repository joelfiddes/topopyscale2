"""Tests for backend dispatch."""

import pytest

from topopyscale2.core.dispatch import KernelDispatcher, available_backends, get_backend


class TestAvailableBackends:
    def test_python_always_available(self):
        backends = available_backends()
        assert "python" in backends

    def test_returns_list(self):
        backends = available_backends()
        assert isinstance(backends, list)


class TestGetBackend:
    def test_python_backend(self):
        modules = get_backend("python")
        assert "temperature" in modules
        assert "radiation" in modules
        assert "precipitation" in modules
        assert "humidity" in modules

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")


class TestKernelDispatcher:
    def test_python_dispatcher(self):
        kd = KernelDispatcher("python")
        assert kd.backend_name == "python"
        assert hasattr(kd.temperature, "lapse_rate_correction")
        assert hasattr(kd.radiation, "partition_shortwave")
        assert hasattr(kd.precipitation, "elevation_gradient")
        assert hasattr(kd.humidity, "adjust_humidity")

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError):
            KernelDispatcher("nonexistent")
