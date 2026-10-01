"""Rust high-performance implementations of physics kernels (via PyO3)."""

try:
    from topopyscale2.core._rust_kernels import (
        humidity,
        precipitation,
        radiation,
        redistribution,
        temperature,
        wind,
    )

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

if RUST_AVAILABLE:
    __all__ = ["temperature", "radiation", "precipitation", "humidity", "wind", "redistribution"]
else:
    __all__ = []
