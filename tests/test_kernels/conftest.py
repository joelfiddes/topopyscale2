"""Cross-backend kernel test fixtures."""

import functools

import numpy as np
import pytest

from topopyscale2.core.python_kernels import humidity as py_humidity
from topopyscale2.core.python_kernels import precipitation as py_precipitation
from topopyscale2.core.python_kernels import radiation as py_radiation
from topopyscale2.core.python_kernels import redistribution as py_redistribution
from topopyscale2.core.python_kernels import temperature as py_temperature
from topopyscale2.core.python_kernels import wind as py_wind


class _ArrayWrapper:
    """Wraps a Rust kernel module to auto-convert scalar inputs to arrays.

    The Rust (PyO3) kernels require numpy arrays (PyArrayDyn<f64> or PyArrayDyn<i64>)
    and cannot accept bare scalars. This wrapper transparently converts scalars to 1-d
    arrays before calling Rust, then extracts scalar results when the first
    input was scalar.

    For functions with optional float parameters (like sx_scale, sx_ref), these
    are passed through as-is since they're not array inputs.

    Integer arrays (like neighbor_indices) are preserved as int64.
    """

    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        fn = getattr(self._mod, name)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            # Track whether the first arg is scalar (0-d) to decide output shape
            first_is_scalar = np.ndim(args[0]) == 0

            # Convert array-like args to arrays, but leave plain floats/ints as-is
            # Array args: anything that looks like a numpy array or sequence
            arr_args = []
            for a in args:
                # Check if it's a bare Python float/int (not a numpy scalar)
                if isinstance(a, (int, float)) and not isinstance(a, np.generic):
                    arr_args.append(a)  # Pass through as scalar parameter
                else:
                    arr = np.asarray(a)
                    # Preserve integer dtypes (for neighbor_indices etc)
                    if np.issubdtype(arr.dtype, np.integer):
                        arr_args.append(np.atleast_1d(arr.astype(np.int64)))
                    else:
                        arr_args.append(np.atleast_1d(arr.astype(np.float64)))

            result = fn(*arr_args, **kwargs)
            if first_is_scalar:
                if isinstance(result, tuple):
                    return tuple(r.item() if hasattr(r, 'item') else r for r in result)
                return result.item() if hasattr(result, 'item') else result
            return result

        return wrapper


BACKENDS = {"python": {
    "temperature": py_temperature,
    "radiation": py_radiation,
    "precipitation": py_precipitation,
    "humidity": py_humidity,
    "wind": py_wind,
    "redistribution": py_redistribution,
}}

try:
    from topopyscale2.core.rust_kernels import (
        humidity as rs_humidity,
    )
    from topopyscale2.core.rust_kernels import (
        precipitation as rs_precipitation,
    )
    from topopyscale2.core.rust_kernels import (
        radiation as rs_radiation,
    )
    from topopyscale2.core.rust_kernels import (
        redistribution as rs_redistribution,
    )
    from topopyscale2.core.rust_kernels import (
        temperature as rs_temperature,
    )
    from topopyscale2.core.rust_kernels import (
        wind as rs_wind,
    )
    BACKENDS["rust"] = {
        "temperature": _ArrayWrapper(rs_temperature),
        "radiation": _ArrayWrapper(rs_radiation),
        "precipitation": _ArrayWrapper(rs_precipitation),
        "humidity": _ArrayWrapper(rs_humidity),
        "wind": _ArrayWrapper(rs_wind),
        "redistribution": _ArrayWrapper(rs_redistribution),
    }
except ImportError:
    pass

try:
    from topopyscale2.core.jax_kernels import (
        humidity as jax_humidity,
    )
    from topopyscale2.core.jax_kernels import (
        precipitation as jax_precipitation,
    )
    from topopyscale2.core.jax_kernels import (
        radiation as jax_radiation,
    )
    from topopyscale2.core.jax_kernels import (
        redistribution as jax_redistribution,
    )
    from topopyscale2.core.jax_kernels import (
        temperature as jax_temperature,
    )
    from topopyscale2.core.jax_kernels import (
        wind as jax_wind,
    )
    BACKENDS["jax"] = {
        "temperature": jax_temperature,
        "radiation": jax_radiation,
        "precipitation": jax_precipitation,
        "humidity": jax_humidity,
        "wind": jax_wind,
        "redistribution": jax_redistribution,
    }
except (ImportError, AttributeError):
    pass


def get_available_backends():
    return list(BACKENDS.keys())


@pytest.fixture(params=get_available_backends())
def backend_name(request):
    return request.param


@pytest.fixture
def temperature_mod(backend_name):
    return BACKENDS[backend_name]["temperature"]


@pytest.fixture
def radiation_mod(backend_name):
    return BACKENDS[backend_name]["radiation"]


@pytest.fixture
def precipitation_mod(backend_name):
    return BACKENDS[backend_name]["precipitation"]


@pytest.fixture
def humidity_mod(backend_name):
    return BACKENDS[backend_name]["humidity"]


@pytest.fixture
def wind_mod(backend_name):
    return BACKENDS[backend_name]["wind"]


@pytest.fixture
def redistribution_mod(backend_name):
    return BACKENDS[backend_name]["redistribution"]
