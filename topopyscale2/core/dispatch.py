"""Backend dispatch for physics kernels."""

import logging
from types import ModuleType

logger = logging.getLogger(__name__)
_WARNED_RUST_FALLBACK = False


def available_backends() -> list[str]:
    """Return list of available kernel backends."""
    backends = ["python"]

    try:
        from topopyscale2.core import rust_kernels

        if rust_kernels.RUST_AVAILABLE:
            backends.append("rust")
    except ImportError:
        pass

    try:
        from topopyscale2.core.jax_kernels import JAX_AVAILABLE

        if JAX_AVAILABLE:
            backends.append("jax")
    except ImportError:
        pass

    return backends


def get_backend(name: str) -> dict[str, ModuleType]:
    """Get kernel modules for a specific backend.

    Returns a dict mapping module names to module objects.
    """
    if name == "python":
        from topopyscale2.core.python_kernels import (
            humidity,
            interpolation,
            precipitation,
            radiation,
            redistribution,
            temperature,
            wind,
        )

        return {
            "temperature": temperature,
            "radiation": radiation,
            "precipitation": precipitation,
            "humidity": humidity,
            "wind": wind,
            "redistribution": redistribution,
            "interpolation": interpolation,
        }

    elif name == "rust":
        try:
            from topopyscale2.core._rust_kernels import (
                humidity,
                interpolation,
                precipitation,
                radiation,
                redistribution,
                temperature,
                wind,
            )

            return {
                "temperature": temperature,
                "radiation": radiation,
                "precipitation": precipitation,
                "humidity": humidity,
                "wind": wind,
                "redistribution": redistribution,
                "interpolation": interpolation,
            }
        except ImportError:
            raise ImportError(
                "Rust backend not available. Build with: maturin develop"
            )

    elif name == "jax":
        try:
            from topopyscale2.core.jax_kernels import (
                humidity,
                interpolation,
                precipitation,
                radiation,
                redistribution,
                temperature,
                wind,
            )

            return {
                "temperature": temperature,
                "radiation": radiation,
                "precipitation": precipitation,
                "humidity": humidity,
                "wind": wind,
                "redistribution": redistribution,
                "interpolation": interpolation,
            }
        except (ImportError, AttributeError):
            raise ImportError(
                "JAX backend not available. Install with: pip install topopyscale2[jax]"
            )

    else:
        raise ValueError(
            f"Unknown backend '{name}'. Available: {available_backends()}"
        )


class KernelDispatcher:
    """Dispatch kernel calls to the selected backend."""

    def __init__(self, backend: str = "python"):
        global _WARNED_RUST_FALLBACK
        self.requested_backend = backend
        # Rust is a speed-up, not a requirement: the Python kernels are the canonical
        # reference and the cross-backend tests hold them to the same results. If the
        # compiled extension is missing (e.g. a source install without a Rust
        # toolchain), run on Python rather than fail. An explicit "jax" request still
        # errors -- that is a deliberate research choice, not a default.
        if backend == "rust" and "rust" not in available_backends():
            if not _WARNED_RUST_FALLBACK:
                logger.warning(
                    "Rust kernels are not available in this installation; using the Python "
                    "kernels (identical results, slower). Install a prebuilt wheel or build "
                    "with a Rust toolchain to enable them."
                )
                _WARNED_RUST_FALLBACK = True
            backend = "python"
        self.backend_name = backend
        self._modules = get_backend(backend)

    @property
    def temperature(self) -> ModuleType:
        return self._modules["temperature"]

    @property
    def radiation(self) -> ModuleType:
        return self._modules["radiation"]

    @property
    def precipitation(self) -> ModuleType:
        return self._modules["precipitation"]

    @property
    def humidity(self) -> ModuleType:
        return self._modules["humidity"]

    @property
    def wind(self) -> ModuleType:
        return self._modules["wind"]

    @property
    def redistribution(self) -> ModuleType:
        return self._modules["redistribution"]

    @property
    def interpolation(self) -> ModuleType:
        return self._modules["interpolation"]
