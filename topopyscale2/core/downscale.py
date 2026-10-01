"""Downscaling engine — orchestrates kernel application across spatial units.

Supports two modes:
- "full": TopoPyScale pressure-level interpolation (requires ERA5 pressure levels)
- "simple": Fixed lapse rate correction (ERA5-Land compatible, surface data only)

The full mode implements vertical interpolation: for each target elevation,
find the two pressure levels that bracket it and interpolate temperature,
humidity, and wind using inverse distance weighting.

The simple mode applies a fixed lapse rate correction from surface temperature,
suitable for quick runs or when pressure-level data is not available.
"""

import logging
from typing import Literal, Optional

import numpy as np
import xarray as xr

from topopyscale2.config.schema import WindConfig
from topopyscale2.core.dispatch import KernelDispatcher
from topopyscale2.spatial.units import SpatialUnit

log = logging.getLogger(__name__)

# ── Input variable requirements ──────────────────────────────────────────
# These define what the downscaler expects *after* unit conversion
# (i.e. after convert_era5() has run).  Units listed are post-conversion.

REQUIRED_SURFACE_VARS_FULL = {
    "t2m": "K",        # 2m air temperature
    "d2m": "K",        # 2m dewpoint temperature
    "sp": "Pa",        # Surface pressure
    "ssrd": "W m-2",   # Surface solar radiation downwards
    "strd": "W m-2",   # Surface thermal radiation downwards
    "tp": "mm",        # Total precipitation per timestep
}
# z_surf (or z) is also required but has an alias — checked separately.

REQUIRED_SURFACE_VARS_SIMPLE = {
    **REQUIRED_SURFACE_VARS_FULL,
    "u10": "m s-1",    # 10m u-wind component
    "v10": "m s-1",    # 10m v-wind component
}

REQUIRED_PLEV_VARS = {
    "t": "K",          # Temperature on pressure levels
    "z": "m",          # Geopotential height on pressure levels
    "u": "m s-1",      # U-wind component on pressure levels
    "v": "m s-1",      # V-wind component on pressure levels
    "q": "kg kg-1",    # Specific humidity on pressure levels
}

# ── Variable computation groups ──────────────────────────────────────────
# Each group declares its outputs, required inputs per mode, and dependencies.
# The downscaler resolves which groups can run based on available inputs +
# requested outputs, then only executes those groups.

VARIABLE_GROUPS = {
    "temperature": {
        "outputs": ["temperature", "pressure"],
        "surface_requires": ["t2m", "sp"],
        "plev_requires_full": ["t", "z"],
        "surface_requires_simple": ["t2m", "sp"],
        "needs_z_surf": True,
        "needs_derived": False,
        "depends_on": [],
    },
    "precipitation": {
        "outputs": ["precipitation", "rainfall", "snowfall"],
        "surface_requires": ["tp"],
        "plev_requires_full": [],
        "surface_requires_simple": ["tp"],
        "needs_z_surf": True,
        "needs_derived": False,
        "depends_on": ["temperature"],
    },
    "radiation": {
        "outputs": ["shortwave_direct", "shortwave_diffuse", "longwave"],
        "surface_requires": ["ssrd", "strd"],
        "plev_requires_full": [],
        "surface_requires_simple": ["ssrd", "strd"],
        "needs_z_surf": True,
        "needs_derived": True,  # solar_elevation, solar_azimuth, clearness_index
        "depends_on": ["temperature"],  # longwave needs t_source, t_target
    },
    "wind": {
        "outputs": ["wind_speed", "wind_direction"],
        "surface_requires": [],
        "plev_requires_full": ["u", "v", "z"],
        "surface_requires_simple": ["u10", "v10"],
        "needs_z_surf": False,
        "needs_derived": False,
        "depends_on": [],
    },
    "humidity": {
        "outputs": ["humidity_specific", "humidity_relative"],
        "surface_requires": ["d2m", "sp"],
        "plev_requires_full": ["q", "z"],
        "surface_requires_simple": ["d2m", "sp"],
        "needs_z_surf": False,
        "needs_derived": False,
        "depends_on": ["temperature"],
    },
}

# All valid output variable names across all groups
ALL_OUTPUT_VARIABLES = set()
for _g in VARIABLE_GROUPS.values():
    ALL_OUTPUT_VARIABLES.update(_g["outputs"])


def required_era5_variables(
    requested_outputs: list[str],
    mode: str,
) -> Optional[tuple[set[str], set[str]]]:
    """Return minimal ERA5 short-name sets needed to produce *requested_outputs*.

    Parameters
    ----------
    requested_outputs : list[str]
        Output variable names the user wants (e.g. ["temperature", "precipitation"]).
        Empty list means "all" — returns None to signal no filtering.
    mode : str
        Downscaling mode ("simple" or "full").

    Returns
    -------
    (surface_vars, plev_vars) : tuple[set[str], set[str]]
        Minimal sets of ERA5 short names needed.  Returns ``None`` when
        *requested_outputs* is empty (meaning "fetch everything").
    """
    if not requested_outputs:
        return None

    # 1. Find candidate groups from requested outputs
    candidates = set()
    for var in requested_outputs:
        for gname, ginfo in VARIABLE_GROUPS.items():
            if var in ginfo["outputs"]:
                candidates.add(gname)

    # 2. Expand with transitive dependencies
    expanded: set[str] = set()

    def _expand(g: str) -> None:
        if g in expanded:
            return
        expanded.add(g)
        for dep in VARIABLE_GROUPS[g]["depends_on"]:
            _expand(dep)

    for g in candidates:
        _expand(g)

    # 3. Union required inputs from all active groups
    surface_vars: set[str] = set()
    plev_vars: set[str] = set()
    needs_z_surf = False

    for gname in expanded:
        ginfo = VARIABLE_GROUPS[gname]
        if mode == "simple":
            surface_vars.update(
                ginfo.get("surface_requires_simple", ginfo["surface_requires"])
            )
        else:
            surface_vars.update(ginfo["surface_requires"])
        if mode == "full":
            plev_vars.update(ginfo["plev_requires_full"])
        if ginfo["needs_z_surf"]:
            needs_z_surf = True

    if needs_z_surf:
        surface_vars.add("z_surf")

    return surface_vars, plev_vars


def resolve_groups(
    available_surface: set[str],
    available_plev: set[str],
    requested_outputs: list[str],
    mode: str,
) -> list[str]:
    """Determine which computation groups can and should run.

    Parameters
    ----------
    available_surface : set[str]
        Variable names present in the surface forcing dataset.
    available_plev : set[str]
        Variable names present in the pressure-level dataset (empty for simple mode).
    requested_outputs : list[str]
        Output variable names the user wants. Empty list means "all computable".
    mode : str
        Downscaling mode ("simple" or "full").

    Returns
    -------
    list[str]
        Ordered group names that can be computed: temperature first, then the rest.
    """
    # Step 1: Determine candidate groups from requested outputs
    if not requested_outputs:
        candidates = set(VARIABLE_GROUPS.keys())
    else:
        candidates = set()
        for var in requested_outputs:
            for gname, ginfo in VARIABLE_GROUPS.items():
                if var in ginfo["outputs"]:
                    candidates.add(gname)

    # Step 2: Expand with dependencies (transitively)
    expanded = set()
    def _expand(g):
        if g in expanded:
            return
        expanded.add(g)
        for dep in VARIABLE_GROUPS[g]["depends_on"]:
            _expand(dep)
    for g in candidates:
        _expand(g)
    candidates = expanded

    # Step 3: Check input requirements for each group
    has_z_surf = bool({"z_surf", "z"} & available_surface)
    computable = set()
    for gname in candidates:
        ginfo = VARIABLE_GROUPS[gname]

        # Check z_surf requirement
        if ginfo["needs_z_surf"] and not has_z_surf:
            continue

        # Check surface variable requirements
        if mode == "simple":
            surf_req = set(ginfo.get("surface_requires_simple", ginfo["surface_requires"]))
        else:
            surf_req = set(ginfo["surface_requires"])
        if not surf_req.issubset(available_surface):
            continue

        # Check pressure-level requirements (full mode only)
        if mode == "full":
            plev_req = set(ginfo["plev_requires_full"])
            if plev_req and not plev_req.issubset(available_plev):
                continue

        computable.add(gname)

    # Step 4: Drop groups whose dependencies can't compute
    def _deps_met(g, visited=None):
        if visited is None:
            visited = set()
        if g in visited:
            return True
        visited.add(g)
        for dep in VARIABLE_GROUPS[g]["depends_on"]:
            if dep not in computable or not _deps_met(dep, visited):
                return False
        return True

    resolved = {g for g in computable if _deps_met(g)}

    # Step 5: Return in dependency order (temperature first, then rest alphabetically)
    order = []
    if "temperature" in resolved:
        order.append("temperature")
    for g in sorted(resolved):
        if g not in order:
            order.append(g)
    return order


def _validate_surface(ds: xr.Dataset, required: dict[str, str], mode: str) -> None:
    """Check that required surface variables are present."""
    missing = [v for v in required if v not in ds]
    if "z_surf" not in ds and "z" not in ds:
        missing.append("z_surf (or z)")
    if missing:
        raise ValueError(
            f"Downscaler ({mode} mode) missing surface variables: {missing}. "
            f"Expected after convert_era5_surface(): {list(required)}"
        )


def _validate_groups(
    ds_surf: xr.Dataset,
    ds_plev: Optional[xr.Dataset],
    groups: list[str],
    mode: str,
) -> None:
    """Validate that inputs for the resolved groups are present.

    This is a safety check — resolve_groups() already verified availability,
    but this catches any mismatch between resolve and actual data.
    """
    for gname in groups:
        ginfo = VARIABLE_GROUPS[gname]
        if mode == "simple":
            surf_req = ginfo.get("surface_requires_simple", ginfo["surface_requires"])
        else:
            surf_req = ginfo["surface_requires"]
        missing_surf = [v for v in surf_req if v not in ds_surf]
        if ginfo["needs_z_surf"] and "z_surf" not in ds_surf and "z" not in ds_surf:
            missing_surf.append("z_surf (or z)")
        if missing_surf:
            raise ValueError(
                f"Group '{gname}' missing surface inputs: {missing_surf}"
            )
        if mode == "full" and ds_plev is not None:
            plev_req = ginfo["plev_requires_full"]
            missing_plev = [v for v in plev_req if v not in ds_plev]
            if missing_plev:
                raise ValueError(
                    f"Group '{gname}' missing pressure-level inputs: {missing_plev}"
                )


def _validate_plev(ds: xr.Dataset) -> None:
    """Check that required pressure-level variables are present."""
    missing = [v for v in REQUIRED_PLEV_VARS if v not in ds]
    if missing:
        raise ValueError(
            f"Downscaler (full mode) missing pressure-level variables: {missing}. "
            f"Expected after convert_era5_pressure(): {list(REQUIRED_PLEV_VARS)}"
        )
    if "level" not in ds.dims:
        raise ValueError(
            "Pressure-level dataset must have a 'level' dimension."
        )


# Default roughness lengths [m] and displacement heights [m] by surface type
DEFAULT_ROUGHNESS_LENGTHS = {
    "glacier": 0.001,
    "open": 0.03,
    "rock": 0.01,
    "forest": 1.0,
}
DEFAULT_DISPLACEMENT_HEIGHTS = {
    "glacier": 0.0,
    "open": 0.0,
    "rock": 0.0,
    "forest": 10.0,
}

# ERA5 reference values (10m measurement height, open terrain)
ERA5_Z_REF = 10.0  # 10m wind measurement height
ERA5_Z0 = 0.03  # ERA5 effective roughness length (open terrain)
ERA5_D = 0.0  # ERA5 displacement height (open terrain)

# Gravitational acceleration [m/s²]
G = 9.80665

# Gas constant for dry air [J/(kg·K)]
R_DRY = 287.05


def _interpolate_pressure_levels_batch(
    values: np.ndarray,
    z_levels: np.ndarray,
    z_targets: np.ndarray,
) -> np.ndarray:
    """Interpolate pressure-level values at multiple target elevations.

    Same inverse-distance-weighted bracketing algorithm as the per-unit
    ``interpolate_pressure_levels`` kernel, but vectorized across all units
    within each timestep.  This is orchestration code, not a kernel — no
    three-backend sync needed.

    Parameters
    ----------
    values : np.ndarray
        Values at pressure levels, shape (n_time, n_levels, n_units).
    z_levels : np.ndarray
        Geopotential heights [m], shape (n_time, n_levels, n_units).
    z_targets : np.ndarray
        Target elevations [m], shape (n_units,).

    Returns
    -------
    np.ndarray
        Interpolated values, shape (n_time, n_units).
    """
    n_time, n_levels, n_units = values.shape
    result = np.empty((n_time, n_units), dtype=np.float64)

    for t in range(n_time):
        z = z_levels[t]  # (n_levels, n_units)
        v = values[t]    # (n_levels, n_units)

        # For each unit, find the closest level above and below z_target
        # diff[k, u] = z[k, u] - z_targets[u]
        diff = z - z_targets[np.newaxis, :]  # (n_levels, n_units)

        # Mask for levels above / below target
        above = diff > 0    # (n_levels, n_units)
        below = diff < 0    # (n_levels, n_units)
        has_above = above.any(axis=0)  # (n_units,)
        has_below = below.any(axis=0)  # (n_units,)

        # Default: if only above or only below, use nearest
        # Index of lowest (min z) and highest (max z) levels per unit
        i_lowest = np.argmin(z, axis=0)   # (n_units,)
        i_highest = np.argmax(z, axis=0)  # (n_units,)

        # Find closest level above target (smallest positive diff)
        diff_above = np.where(above, diff, np.inf)  # (n_levels, n_units)
        i_top = np.argmin(diff_above, axis=0)        # (n_units,)

        # Find closest level below target (largest negative diff → closest to 0)
        diff_below = np.where(below, diff, -np.inf)  # (n_levels, n_units)
        i_bot = np.argmax(diff_below, axis=0)         # (n_units,)

        # Gather values and heights at bracketing levels
        units_idx = np.arange(n_units)
        z_top = z[i_top, units_idx]
        z_bot = z[i_bot, units_idx]
        v_top = v[i_top, units_idx]
        v_bot = v[i_bot, units_idx]

        # Inverse distance weighting
        d_top = z_top - z_targets
        d_bot = z_targets - z_bot
        total_dist = d_top + d_bot
        safe_dist = np.where(total_dist > 0, total_dist, 1.0)
        w_top = d_bot / safe_dist
        w_bot = d_top / safe_dist
        interp_val = w_bot * v_bot + w_top * v_top

        # Handle edge cases: target below all levels or above all levels
        val_lowest = v[i_lowest, units_idx]
        val_highest = v[i_highest, units_idx]

        bracketed = has_above & has_below
        only_above = has_above & ~has_below
        only_below = ~has_above & has_below
        neither = ~has_above & ~has_below

        out = np.where(bracketed, interp_val, 0.0)
        out = np.where(only_above, val_lowest, out)
        out = np.where(only_below, val_highest, out)
        out = np.where(neither, v[0, units_idx], out)

        result[t] = out

    return result


class Downscaler:
    """Apply topographic downscaling from ERA5 to spatial units.

    Supports two modes:
    - "full": Pressure-level interpolation (TopoPyScale algorithm)
    - "simple": Fixed lapse rate correction (ERA5-Land compatible)
    """

    def __init__(
        self,
        backend: str = "python",
        wind_config: Optional[WindConfig] = None,
        mode: Literal["simple", "full"] = "full",
        lapse_rate: float = 0.0065,
        n_workers: int = 1,
        precip_gradient: float = 0.0,
        phase_method: str = "wet_bulb",
        t_snow: float = 272.65,  # -0.5 degC: all snow below
        t_rain: float = 275.65,  # +2.5 degC: all rain above
        t_air_max: float = 277.15,  # +4 degC: air-temp guard rail (all rain above)
    ):
        if mode not in ("simple", "full"):
            raise ValueError(f"mode must be 'simple' or 'full', got {mode!r}")
        if phase_method not in ("wet_bulb", "air_temperature"):
            raise ValueError(
                f"phase_method must be 'wet_bulb' or 'air_temperature', got {phase_method!r}"
            )
        # `not <=` (rather than `>`) also rejects NaN
        if not abs(lapse_rate) <= 0.02:
            raise ValueError(
                f"lapse_rate must be in [-0.02, 0.02] K/m (dry adiabatic is ~0.0098), "
                f"got {lapse_rate}"
            )
        if not abs(precip_gradient) <= 0.01:
            raise ValueError(
                f"precip_gradient must be in [-0.01, 0.01] per metre, got {precip_gradient}"
            )
        if not t_snow < t_rain:
            raise ValueError(
                f"t_snow must be strictly below t_rain (phase ramp), "
                f"got t_snow={t_snow}, t_rain={t_rain}"
            )

        self.kernels = KernelDispatcher(backend)
        self.wind_config = wind_config or WindConfig()
        self.mode = mode
        self.lapse_rate = lapse_rate  # K/m, default 6.5 K/km
        self.n_workers = n_workers
        self.precip_gradient = precip_gradient  # frac increase per m; 0.0 = off (opt-in)
        # Rain/snow phase partitioning: "wet_bulb" (default) partitions on the
        # psychrometric wet-bulb temperature; "air_temperature" on 2 m air temp.
        self.phase_method = phase_method
        self.t_snow = t_snow  # [K] all snow below
        self.t_rain = t_rain  # [K] all rain above
        # Air-temperature guard rail [K]: force all-rain above this air temperature
        # regardless of the partition (wet-bulb) temperature. Restores, for the
        # wet-bulb method, the air-temp ceiling that t_rain gives the air-temperature
        # method. No-op for "air_temperature" when t_air_max >= t_rain.
        self.t_air_max = t_air_max

        # Build roughness lookup dictionaries
        self.roughness_lengths = {
            **DEFAULT_ROUGHNESS_LENGTHS,
            **self.wind_config.roughness_lengths,
        }
        self.displacement_heights = {
            **DEFAULT_DISPLACEMENT_HEIGHTS,
            **self.wind_config.displacement_heights,
        }

    def _phase_temperature(self, t_target, p_target, q):
        """Temperature [K] fed to the rain/snow phase ramp.

        Returns the psychrometric wet-bulb temperature when phase_method is
        "wet_bulb" and the required moisture/pressure inputs are available;
        otherwise falls back to the air temperature ``t_target`` (e.g. when the
        humidity group is disabled). Works for both 1-D (per-unit) and 2-D
        (vectorized) arrays.
        """
        if self.phase_method == "wet_bulb" and p_target is not None and q is not None:
            return self.kernels.precipitation.wet_bulb_temperature(
                np.ascontiguousarray(np.asarray(t_target, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(p_target, dtype=np.float64)),
                np.ascontiguousarray(np.asarray(q, dtype=np.float64)),
            )
        return t_target

    def _air_temp_guard(self, rainfall, snowfall, p_precip, t_air):
        """Force all-rain where air temperature exceeds the guard ceiling.

        The wet-bulb ramp caps on wet-bulb temperature, which in dry air permits
        snow at high *air* temperatures; this re-imposes an air-temperature ceiling
        (the same bound t_rain gives the air-temperature method). Pure numpy so it
        works for any kernel backend; handles 1-D and 2-D arrays.
        """
        if self.t_air_max is None:
            return rainfall, snowfall
        over = np.asarray(t_air) > self.t_air_max
        rainfall = np.where(over, p_precip, rainfall)
        snowfall = np.where(over, 0.0, snowfall)
        return rainfall, snowfall

    def process_unit(
        self,
        unit: SpatialUnit,
        surf_forcing: xr.Dataset,
        plev_forcing: xr.Dataset,
        solar_elevation: Optional[xr.DataArray] = None,
        solar_azimuth: Optional[xr.DataArray] = None,
        clearness_index: Optional[xr.DataArray] = None,
        groups: Optional[set[str]] = None,
    ) -> xr.Dataset:
        """Downscale forcing for a single spatial unit using pressure-level interpolation.

        This is a pure function (no side effects) — parallelizable.
        Thin wrapper around the shared vectorized implementation
        (``_process_all``) with a batch of one unit.

        Parameters
        ----------
        unit : SpatialUnit
            Target spatial unit with elevation.
        surf_forcing : xr.Dataset
            ERA5 surface forcing at the nearest grid cell. Expected vars:
            t2m, d2m, sp, ssrd, strd, tp, z_surf (surface geopotential).
        plev_forcing : xr.Dataset
            ERA5 pressure-level forcing at the nearest grid cell. Expected vars:
            t, z, u, v, q with dimension 'level'.
        solar_elevation : xr.DataArray, optional
            Solar elevation [degrees], shape (time,). Required if radiation group active.
        solar_azimuth : xr.DataArray, optional
            Solar azimuth [degrees], shape (time,). Required if radiation group active.
        clearness_index : xr.DataArray, optional
            Clearness index kt [-], shape (time,). Required if radiation group active.
        groups : set[str], optional
            Computation groups to run. None = all groups (backward compatible).

        Returns
        -------
        xr.Dataset
            Downscaled forcing for this unit.
        """
        if groups is None:
            # Backward compatibility: validate all inputs, run all groups
            _validate_surface(surf_forcing, REQUIRED_SURFACE_VARS_FULL, "full")
            _validate_plev(plev_forcing)
            groups = set(VARIABLE_GROUPS.keys())

        return self._process_unit_via_batch(
            unit, surf_forcing, plev_forcing,
            solar_elevation, solar_azimuth, clearness_index,
            groups, mode="full",
        )

    def process_unit_simple(
        self,
        unit: SpatialUnit,
        surf_forcing: xr.Dataset,
        solar_elevation: Optional[xr.DataArray] = None,
        solar_azimuth: Optional[xr.DataArray] = None,
        clearness_index: Optional[xr.DataArray] = None,
        groups: Optional[set[str]] = None,
    ) -> xr.Dataset:
        """Downscale forcing using simple lapse rate (ERA5-Land compatible).

        This mode only requires surface data (t2m, sp, etc.) and applies a
        fixed lapse rate correction for temperature. Suitable for quick runs
        or when pressure-level data is not available.
        Thin wrapper around the shared vectorized implementation
        (``_process_all``) with a batch of one unit.

        Parameters
        ----------
        unit : SpatialUnit
            Target spatial unit with elevation.
        surf_forcing : xr.Dataset
            ERA5 surface forcing. Expected vars depend on active groups.
            Full set: t2m, d2m, sp, ssrd, strd, tp, z or z_surf, u10, v10.
        solar_elevation : xr.DataArray, optional
            Solar elevation [degrees], shape (time,). Required if radiation group active.
        solar_azimuth : xr.DataArray, optional
            Solar azimuth [degrees], shape (time,). Required if radiation group active.
        clearness_index : xr.DataArray, optional
            Clearness index kt [-], shape (time,). Required if radiation group active.
        groups : set[str], optional
            Computation groups to run. None = all groups (backward compatible).

        Returns
        -------
        xr.Dataset
            Downscaled forcing for this unit.
        """
        if groups is None:
            # Backward compatibility: validate all inputs, run all groups
            _validate_surface(surf_forcing, REQUIRED_SURFACE_VARS_SIMPLE, "simple")
            groups = set(VARIABLE_GROUPS.keys())

        return self._process_unit_via_batch(
            unit, surf_forcing, None,
            solar_elevation, solar_azimuth, clearness_index,
            groups, mode="simple",
        )

    def _process_unit_via_batch(
        self,
        unit: SpatialUnit,
        surf_forcing: xr.Dataset,
        plev_forcing: Optional[xr.Dataset],
        solar_elevation,
        solar_azimuth,
        clearness_index,
        groups: set[str],
        mode: Literal["simple", "full"],
    ) -> xr.Dataset:
        """Run one unit through the vectorized path as an n_units=1 batch."""
        time = surf_forcing.time if "time" in surf_forcing.dims else surf_forcing.coords.get("time")
        if "time" in surf_forcing.sizes:
            n_time = surf_forcing.sizes["time"]
        elif time is not None:
            n_time = int(np.asarray(time).size)
        else:
            first = next(iter(surf_forcing.data_vars), None)
            first_arr = np.asarray(surf_forcing[first].values) if first else np.empty(1)
            n_time = first_arr.shape[0] if first_arr.ndim > 0 else 1

        def col(values):
            """(time,) or scalar → C-contiguous (n_time, 1) float64 column."""
            arr = np.asarray(values, dtype=np.float64)
            if arr.ndim == 0:
                arr = np.broadcast_to(arr, (n_time,))
            return np.ascontiguousarray(arr.reshape(n_time, 1))

        surf = {v: col(surf_forcing[v].values) for v in surf_forcing.data_vars}
        z_key = "z_surf" if "z_surf" in surf_forcing else "z"
        if z_key not in surf and z_key in surf_forcing:
            surf[z_key] = col(surf_forcing[z_key].values)

        plev = None
        if mode == "full" and plev_forcing is not None:
            plev = {}
            for v in plev_forcing.data_vars:
                arr = np.asarray(plev_forcing[v].values, dtype=np.float64)
                if arr.ndim == 1:
                    # Constant-in-time profile: (n_levels,) → (n_time, n_levels)
                    arr = np.broadcast_to(arr, (n_time, arr.shape[0]))
                elif arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    # (n_levels, n_time) → (n_time, n_levels)
                    arr = arr.T
                plev[v] = np.ascontiguousarray(arr.reshape(n_time, -1, 1))

        z_target = np.array([unit.elevation], dtype=np.float64)
        slope = np.array([unit.attributes.get("slope", 0.0)], dtype=np.float64)
        aspect = np.array([unit.attributes.get("aspect", 0.0)], dtype=np.float64)
        svf = np.array([unit.attributes.get("svf", 1.0)], dtype=np.float64)
        sx_val = unit.attributes.get("sx", np.nan)
        sx = np.array([np.nan if sx_val is None else sx_val], dtype=np.float64)
        z0_target = np.array(
            [self.roughness_lengths.get(unit.surface_type, DEFAULT_ROUGHNESS_LENGTHS["open"])],
            dtype=np.float64,
        )
        d_target = np.array(
            [self.displacement_heights.get(
                unit.surface_type, DEFAULT_DISPLACEMENT_HEIGHTS["open"])],
            dtype=np.float64,
        )

        sol_elev = sol_az = kt = None
        if "radiation" in groups:
            sol_elev = col(solar_elevation)
            sol_az = col(solar_azimuth)
            kt = col(clearness_index)

        output_vars = [
            var
            for gname, ginfo in VARIABLE_GROUPS.items()
            if gname in groups
            for var in ginfo["outputs"]
        ]
        out = {var: np.empty((n_time, 1), dtype=np.float64) for var in output_vars}

        self._process_all(
            out, surf, plev, groups,
            z_target, surf.get(z_key), surf.get("t2m"), surf.get("sp"),
            slope, aspect, svf, z0_target, d_target, sx,
            sol_elev, sol_az, kt, surf.get("d2m"),
            mode,
        )

        data_vars = {var: (["time"], out[var][:, 0]) for var in output_vars}
        attrs = {"unit_id": unit.id}
        if mode == "simple":
            attrs["downscaling_mode"] = "simple"
        return xr.Dataset(
            data_vars,
            coords={"time": time} if time is not None else {},
            attrs=attrs,
        )

    @staticmethod
    def map_units_to_era5(
        units: list[SpatialUnit],
        era5_lat: np.ndarray,
        era5_lon: np.ndarray,
    ) -> dict[str, tuple[int, int]]:
        """Map spatial units to nearest ERA5 grid cell indices.

        Returns dict mapping unit_id → (lat_idx, lon_idx).

        .. deprecated::
            Prefer :meth:`compute_bilinear_weights` for horizontal interpolation
            instead of nearest-neighbor assignment.
        """
        # Vectorized: compute all lat/lon distances at once
        unit_lats = np.array([u.y for u in units])
        unit_lons = np.array([u.x for u in units])
        # |unit_lats[:, None] - era5_lat[None, :]| → (n_units, n_lat)
        lat_indices = np.argmin(np.abs(unit_lats[:, None] - era5_lat[None, :]), axis=1)
        lon_indices = np.argmin(np.abs(unit_lons[:, None] - era5_lon[None, :]), axis=1)
        return {
            unit.id: (int(lat_indices[i]), int(lon_indices[i]))
            for i, unit in enumerate(units)
        }

    @staticmethod
    def compute_bilinear_weights(
        units: list[SpatialUnit],
        era5_lat: np.ndarray,
        era5_lon: np.ndarray,
        source_crs: Optional[str] = None,
    ) -> dict[str, np.ndarray]:
        """Compute bilinear interpolation weights for all units.

        For each unit, finds the four surrounding ERA5 grid cells and computes
        bilinear weights. Works with both ascending and descending latitude.

        Parameters
        ----------
        units : list[SpatialUnit]
            Spatial units with x and y attributes (in source_crs or lat/lon).
        era5_lat : np.ndarray
            ERA5 latitude coordinate array (ascending or descending).
        era5_lon : np.ndarray
            ERA5 longitude coordinate array (ascending).
        source_crs : str, optional
            CRS of unit coordinates. If not None and not EPSG:4326,
            coordinates are reprojected to lat/lon for ERA5 grid matching.

        Returns
        -------
        dict with keys:
            lat_idx0, lat_idx1 : np.ndarray of int, shape (n_units,)
                Lower and upper latitude indices into the original array.
            lon_idx0, lon_idx1 : np.ndarray of int, shape (n_units,)
                Lower and upper longitude indices into the original array.
            w00, w01, w10, w11 : np.ndarray of float64, shape (n_units,)
                Bilinear weights for (lat0,lon0), (lat0,lon1), (lat1,lon0), (lat1,lon1).
        """
        n_units = len(units)
        unit_xs = np.array([u.x for u in units], dtype=np.float64)
        unit_ys = np.array([u.y for u in units], dtype=np.float64)

        # Convert from projected CRS to lat/lon if needed
        if source_crs is not None:
            from pyproj import CRS, Transformer
            crs_obj = CRS.from_user_input(source_crs)
            if not crs_obj.is_geographic:
                transformer = Transformer.from_crs(crs_obj, CRS.from_epsg(4326), always_xy=True)
                unit_lons, unit_lats = transformer.transform(unit_xs, unit_ys)
                log.info("Reprojected %d unit centroids from %s to EPSG:4326", n_units, source_crs)
            else:
                unit_lons, unit_lats = unit_xs, unit_ys
        else:
            unit_lons, unit_lats = unit_xs, unit_ys

        # Ensure latitude is ascending for searchsorted
        lat = np.asarray(era5_lat, dtype=np.float64)
        lon = np.asarray(era5_lon, dtype=np.float64)
        lat_ascending = lat[np.argsort(lat)]
        sort_idx = np.argsort(lat)

        n_lat = len(lat)
        n_lon = len(lon)

        # Find bounding indices in sorted (ascending) latitude
        j1_sorted = np.searchsorted(lat_ascending, unit_lats).clip(1, n_lat - 1)
        j0_sorted = j1_sorted - 1

        # Map back to original array indices
        lat_idx0 = sort_idx[j0_sorted]
        lat_idx1 = sort_idx[j1_sorted]

        # Longitude (assumed ascending)
        i1 = np.searchsorted(lon, unit_lons).clip(1, n_lon - 1)
        i0 = i1 - 1
        lon_idx0 = i0
        lon_idx1 = i1

        # Fractional positions
        lat0_vals = lat[lat_idx0]
        lat1_vals = lat[lat_idx1]
        lon0_vals = lon[lon_idx0]
        lon1_vals = lon[lon_idx1]

        # Handle exact grid points (avoid division by zero)
        dlat = lat1_vals - lat0_vals
        dlon = lon1_vals - lon0_vals
        # Where dlat or dlon is zero, the unit is exactly on a grid line
        dlat_safe = np.where(dlat == 0, 1.0, dlat)
        dlon_safe = np.where(dlon == 0, 1.0, dlon)

        t = np.where(dlat == 0, 0.0, (unit_lats - lat0_vals) / dlat_safe)
        s = np.where(dlon == 0, 0.0, (unit_lons - lon0_vals) / dlon_safe)

        # Bilinear weights
        w00 = (1 - t) * (1 - s)
        w01 = (1 - t) * s
        w10 = t * (1 - s)
        w11 = t * s

        return {
            "lat_idx0": lat_idx0,
            "lat_idx1": lat_idx1,
            "lon_idx0": lon_idx0,
            "lon_idx1": lon_idx1,
            "w00": w00,
            "w01": w01,
            "w10": w10,
            "w11": w11,
        }

    @staticmethod
    def interpolate_bilinear_all(
        field: np.ndarray,
        weights: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Bilinearly interpolate a field at all unit locations simultaneously.

        Parameters
        ----------
        field : np.ndarray
            3D array (time, lat, lon) or 4D array (time, level, lat, lon).
        weights : dict
            Output of :meth:`compute_bilinear_weights`.

        Returns
        -------
        np.ndarray
            Interpolated values: shape (time, n_units) for 3D input,
            (time, level, n_units) for 4D input.
        """
        li0 = weights["lat_idx0"]
        li1 = weights["lat_idx1"]
        lo0 = weights["lon_idx0"]
        lo1 = weights["lon_idx1"]
        w00 = weights["w00"]
        w01 = weights["w01"]
        w10 = weights["w10"]
        w11 = weights["w11"]

        if field.ndim == 3:
            # (time, lat, lon) → (time, n_units)
            return (
                w00 * field[:, li0, lo0]
                + w01 * field[:, li0, lo1]
                + w10 * field[:, li1, lo0]
                + w11 * field[:, li1, lo1]
            )
        elif field.ndim == 4:
            # (time, level, lat, lon) → (time, level, n_units)
            return (
                w00 * field[:, :, li0, lo0]
                + w01 * field[:, :, li0, lo1]
                + w10 * field[:, :, li1, lo0]
                + w11 * field[:, :, li1, lo1]
            )
        else:
            raise ValueError(f"Expected 3D or 4D field, got {field.ndim}D")

    @staticmethod
    def interpolate_bilinear_unit(
        field: np.ndarray,
        weights: dict[str, np.ndarray],
        unit_idx: int,
    ) -> np.ndarray:
        """Extract bilinearly-interpolated data for a single unit.

        Parameters
        ----------
        field : np.ndarray
            3D array (time, lat, lon) or 4D array (time, level, lat, lon).
        weights : dict
            Output of :meth:`compute_bilinear_weights`.
        unit_idx : int
            Index into the weights arrays for this unit.

        Returns
        -------
        np.ndarray
            Interpolated values: shape (time,) for 3D input,
            (time, level) for 4D input.
        """
        li0 = weights["lat_idx0"][unit_idx]
        li1 = weights["lat_idx1"][unit_idx]
        lo0 = weights["lon_idx0"][unit_idx]
        lo1 = weights["lon_idx1"][unit_idx]
        w00 = weights["w00"][unit_idx]
        w01 = weights["w01"][unit_idx]
        w10 = weights["w10"][unit_idx]
        w11 = weights["w11"][unit_idx]

        if field.ndim == 3:
            # (time, lat, lon) → (time,)
            return (
                w00 * field[:, li0, lo0]
                + w01 * field[:, li0, lo1]
                + w10 * field[:, li1, lo0]
                + w11 * field[:, li1, lo1]
            )
        elif field.ndim == 4:
            # (time, level, lat, lon) → (time, level)
            return (
                w00 * field[:, :, li0, lo0]
                + w01 * field[:, :, li0, lo1]
                + w10 * field[:, :, li1, lo0]
                + w11 * field[:, :, li1, lo1]
            )
        else:
            raise ValueError(f"Expected 3D or 4D field, got {field.ndim}D")

    # ── Vectorized batch processing ────────────────────────────────────

    def _process_all(
        self,
        out: dict[str, np.ndarray],
        surf: dict[str, np.ndarray],
        plev: Optional[dict[str, np.ndarray]],
        groups: set[str],
        z_target: np.ndarray,
        z_source: np.ndarray,
        t_source: np.ndarray,
        p_source: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        svf: np.ndarray,
        z0_target: np.ndarray,
        d_target: np.ndarray,
        sx: np.ndarray,
        sol_elev: Optional[np.ndarray],
        sol_az: Optional[np.ndarray],
        kt: Optional[np.ndarray],
        d2m: Optional[np.ndarray],
        mode: Literal["simple", "full"],
    ) -> None:
        """Vectorized downscaling on (n_time, n_units) arrays.

        Single implementation for both modes. They differ only in the
        *source* of temperature, wind, and specific humidity: "full"
        interpolates from pressure levels (via
        ``_interpolate_pressure_levels_batch``), "simple" derives them from
        surface fields with a fixed lapse rate. Pressure, radiation, and
        precipitation physics are mode-independent.

        Writes results directly into the pre-allocated ``out`` dict.
        """
        def _c(arr):
            """Ensure C-contiguous float64 (required by Rust kernels)."""
            return np.ascontiguousarray(arr, dtype=np.float64)

        full = mode == "full"
        n_time = t_source.shape[0] if t_source is not None else next(iter(surf.values())).shape[0]
        n_u = z_target.shape[0]

        # Pressure-level geopotential heights: (n_time, n_levels, n_units)
        z_plev = plev.get("z") if plev else None

        # --- Temperature ---
        t_target = None
        p_target = None
        if "temperature" in groups:
            if full:
                t_target = _interpolate_pressure_levels_batch(plev["t"], z_plev, z_target)
            else:
                # simple_lapse_rate: t - lr * (z_tgt - z_src)
                # Equivalent to lapse_rate_correction with gamma = -lr
                gamma = np.broadcast_to(-self.lapse_rate, z_target.shape).copy()
                t_target = _c(self.kernels.temperature.lapse_rate_correction(
                    _c(t_source), _c(gamma), _c(z_target), _c(z_source),
                ))

            # Surface pressure (hypsometric from downscaled temperature)
            dz = z_target[np.newaxis, :] - z_source
            t_mean = (t_source + t_target) / 2.0
            p_target = p_source * np.exp(-G * dz / (R_DRY * t_mean))

            out["temperature"][:] = t_target
            out["pressure"][:] = p_target

        # --- Wind ---
        if "wind" in groups:
            if full:
                u = _interpolate_pressure_levels_batch(plev["u"], z_plev, z_target)
                v = _interpolate_pressure_levels_batch(plev["v"], z_plev, z_target)
                # Pressure-level wind is free-air: bring down from ~50 m with
                # negligible source roughness/displacement
                z_from, z0_src, d_src = 50.0, 0.01, 0.0
            else:
                u = surf.get("u10", np.zeros((n_time, n_u)))
                v = surf.get("v10", np.zeros_like(u))
                z_from, z0_src, d_src = ERA5_Z_REF, ERA5_Z0, ERA5_D

            wind_speed = np.sqrt(u**2 + v**2)
            wind_dir = np.degrees(np.arctan2(-u, -v)) % 360

            # Broadcast per-unit scalars to (n_time, n_units) — copy for Rust
            z_from_b = np.broadcast_to(z_from, (n_time, n_u)).copy()
            z_to_b = np.broadcast_to(ERA5_Z_REF, (n_time, n_u)).copy()
            z0_src_b = np.broadcast_to(z0_src, (n_time, n_u)).copy()
            d_src_b = np.broadcast_to(d_src, (n_time, n_u)).copy()
            z0_t = np.broadcast_to(z0_target, (n_time, n_u)).copy()
            d_t = np.broadcast_to(d_target, (n_time, n_u)).copy()

            wind_speed_corr = _c(self.kernels.wind.log_profile_correction(
                _c(wind_speed), z_from_b, z_to_b, z0_src_b, z0_t, d_src_b, d_t,
            ))

            if self.wind_config.method == "winstral":
                # Only apply where sx is not NaN
                has_sx = ~np.isnan(sx)
                if has_sx.any():
                    sx_broad = np.broadcast_to(sx, (n_time, n_u)).copy()
                    sx_corr = _c(self.kernels.wind.winstral_wind_correction(
                        wind_speed_corr, sx_broad,
                    ))
                    # Blend: use corrected where has_sx, uncorrected otherwise
                    wind_speed_corr = np.where(has_sx, sx_corr, wind_speed_corr)

            out["wind_speed"][:] = wind_speed_corr
            out["wind_direction"][:] = wind_dir

        # --- Humidity ---
        if "humidity" in groups:
            if full:
                q_target = _interpolate_pressure_levels_batch(plev["q"], z_plev, z_target)
                # Vapor pressure from mixing ratio (Bolton 1980)
                mr = q_target / (1.0 - q_target)
                e_actual = mr * p_target / (0.62197 + mr)
            else:
                # Conserve specific humidity computed from surface dewpoint
                e_sat_d = 611.2 * np.exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
                q_target = 0.622 * e_sat_d / (p_source - 0.378 * e_sat_d)
                e_actual = q_target * p_target / (0.622 + 0.378 * q_target)

            e_sat_t = 611.2 * np.exp(17.67 * (t_target - 273.15) / (t_target - 29.65))
            rh_target = np.clip(e_actual / e_sat_t, 0.0, 1.0)

            out["humidity_specific"][:] = q_target
            out["humidity_relative"][:] = rh_target

        # --- Radiation ---
        if "radiation" in groups:
            sw_total = surf["ssrd"]  # (n_time, n_units)

            sw_direct, sw_diffuse = self.kernels.radiation.partition_shortwave(
                _c(sw_total), _c(sol_elev), _c(kt),
            )
            sw_direct = _c(sw_direct)
            sw_diffuse = _c(sw_diffuse)

            slope_b = np.broadcast_to(slope, (n_time, n_u)).copy()
            aspect_b = np.broadcast_to(aspect, (n_time, n_u)).copy()
            svf_b = np.broadcast_to(svf, (n_time, n_u)).copy()

            out["shortwave_direct"][:] = _c(self.kernels.radiation.slope_correction(
                sw_direct, _c(sol_elev), _c(sol_az), slope_b, aspect_b,
            ))
            out["shortwave_diffuse"][:] = _c(self.kernels.radiation.diffuse_correction(
                sw_diffuse, svf_b,
            ))

            lw_source = surf["strd"]  # (n_time, n_units)

            # Vapor pressure for Brutsaert emissivity model
            d2m_r = d2m if d2m is not None else surf.get("d2m")
            vp_src = 611.2 * np.exp(17.67 * (d2m_r - 273.15) / (d2m_r - 29.65))
            if "humidity" in groups:
                q_t = out["humidity_specific"]
                vp_tgt = q_t * p_target / (0.622 + 0.378 * q_t)
            else:
                vp_tgt = vp_src * (p_target / p_source) if p_target is not None else vp_src

            out["longwave"][:] = _c(self.kernels.radiation.longwave_correction(
                _c(lw_source), _c(t_source), _c(t_target),
                _c(vp_src), _c(vp_tgt), svf_b,
            ))

        # --- Precipitation ---
        if "precipitation" in groups:
            p_total = surf["tp"]  # (n_time, n_units)

            z_t_b = np.broadcast_to(z_target, (n_time, n_u)).copy()
            if z_source.ndim < 2:
                z_s_b = np.broadcast_to(z_source, (n_time, n_u)).copy()
            else:
                z_s_b = z_source.copy()
            grad_b = np.broadcast_to(self.precip_gradient, (n_time, n_u)).copy()

            p_corrected = _c(self.kernels.precipitation.elevation_gradient(
                _c(p_total), z_t_b, z_s_b, grad_b,
            ))

            q_phase = out["humidity_specific"] if "humidity" in groups else None
            t_phase = _c(self._phase_temperature(t_target, p_target, q_phase))
            t_rain = np.broadcast_to(self.t_rain, t_target.shape).copy()
            t_snow = np.broadcast_to(self.t_snow, t_target.shape).copy()
            rainfall, snowfall = self.kernels.precipitation.phase_partition(
                p_corrected, t_phase, t_rain, t_snow,
            )
            rainfall, snowfall = self._air_temp_guard(rainfall, snowfall, p_corrected, t_target)

            out["precipitation"][:] = p_corrected
            out["rainfall"][:] = _c(rainfall)
            out["snowfall"][:] = _c(snowfall)

    def process(
        self,
        units,
        forcing: xr.Dataset,
        dem_data: xr.Dataset,
        requested_outputs: Optional[list[str]] = None,
        progress_callback=None,
    ) -> xr.Dataset:
        """Downscale forcing for all spatial units.

        Parameters
        ----------
        units : SpatialUnitCollection or list[SpatialUnit]
            Spatial units to process.
        forcing : xr.Dataset
            ERA5 forcing with both surface and pressure-level variables.
            Surface: t2m, d2m, sp, ssrd, strd, tp, z_surf
            Pressure-level: t, z, u, v, q (with 'level' dimension)
            Derived: solar_elevation, clearness_index (only if radiation needed)
        dem_data : xr.Dataset
            DEM data (used for solar azimuth computation if needed).
        requested_outputs : list[str], optional
            Output variables the user wants. Empty or None = all computable.
        progress_callback : callable, optional
            Called as progress_callback(done, total) after each unit is processed.

        Returns
        -------
        xr.Dataset
            Downscaled forcing with dimensions (time, unit).
        """
        import time as time_module

        from topopyscale2.spatial.units import SpatialUnitCollection

        # Convert to list if SpatialUnitCollection
        if isinstance(units, SpatialUnitCollection):
            unit_list = list(units)
        else:
            unit_list = list(units)

        n_units = len(unit_list)
        log.info("Downscaling %d units (n_workers=%d)...", n_units, self.n_workers)
        start = time_module.time()

        # --- Resolve computation groups ---
        # Determine what variables are available in the forcing
        surf_var_names_all = [v for v in forcing.data_vars
                              if "level" not in forcing[v].dims
                              and v not in ("solar_elevation", "clearness_index")]
        plev_var_names_all = [v for v in forcing.data_vars if "level" in forcing[v].dims]
        available_surface = set(surf_var_names_all)
        available_plev = set(plev_var_names_all)

        active_groups = resolve_groups(
            available_surface, available_plev,
            requested_outputs or [], self.mode,
        )
        active_groups_set = set(active_groups)

        # Collect the output variables we'll produce
        output_vars = []
        for gname in active_groups:
            output_vars.extend(VARIABLE_GROUPS[gname]["outputs"])

        log.info("Active groups: %s → output vars: %s", active_groups, output_vars)
        if not active_groups:
            log.warning("No computation groups resolved — returning empty dataset")

        # Get ERA5 coordinates
        era5_lat = forcing.latitude.values
        era5_lon = forcing.longitude.values

        # Compute bilinear interpolation weights (once for all units)
        # Detect CRS from dem_data to reproject unit centroids to lat/lon
        source_crs = None
        if dem_data is not None and hasattr(dem_data, 'attrs'):
            source_crs = dem_data.attrs.get('crs', None)
        bilinear = self.compute_bilinear_weights(unit_list, era5_lat, era5_lon, source_crs=source_crs)
        log.info("Computed bilinear weights for %d units", n_units)

        # Extract derived quantities only if radiation group is active
        solar_elevation = None
        solar_azimuth = None
        clearness_index = None
        sol_elev_arr = None
        sol_az_arr = None
        kt_arr = None
        if "radiation" in active_groups_set:
            solar_elevation = forcing["solar_elevation"]
            clearness_index = forcing["clearness_index"]
            solar_azimuth = xr.zeros_like(solar_elevation)
            sol_elev_arr = solar_elevation.values
            sol_az_arr = solar_azimuth.values
            kt_arr = clearness_index.values

        # --- Extract numpy arrays from forcing (shares memory, no copy) ---
        # Surface variables: (time, lat, lon)
        surf_var_names = [v for v in forcing.data_vars
                          if "level" not in forcing[v].dims
                          and "latitude" in forcing[v].dims]
        surf_arrays = {v: forcing[v].values for v in surf_var_names}

        # Pressure-level variables: ensure (time, level, lat, lon) ordering
        plev_var_names = [v for v in forcing.data_vars if "level" in forcing[v].dims]
        plev_arrays = {}
        for v in plev_var_names:
            da = forcing[v]
            arr = da.values  # eagerly load numpy array first
            if da.dims[0] != "time":
                # numpy transpose is guaranteed; dask lazy transpose may not apply
                time_ax = da.dims.index("time")
                level_ax = da.dims.index("level")
                lat_ax = da.dims.index("latitude")
                lon_ax = da.dims.index("longitude")
                arr = np.ascontiguousarray(arr.transpose(time_ax, level_ax, lat_ax, lon_ax))
            plev_arrays[v] = arr

        # Handle scalar z_surf (no lat/lon dims) — broadcast to grid shape
        z_surf_key = "z_surf" if "z_surf" in forcing else "z"
        z_surf_scalar = None
        if z_surf_key in surf_arrays:
            pass  # already in surf_arrays with spatial dims
        elif z_surf_key in forcing:
            z_surf_scalar = forcing[z_surf_key].values

        # --- Pre-allocate output arrays ---
        n_time = forcing.sizes["time"]
        time_coord = forcing.time.values

        # Only allocate arrays for variables we'll actually compute
        out_arrays = {var: np.empty((n_time, n_units), dtype=np.float64) for var in output_vars}

        # --- Extract ALL unit attributes upfront ---
        z_target_all = np.array([u.elevation for u in unit_list], dtype=np.float64)
        slope_all = np.array([u.attributes.get("slope", 0.0) for u in unit_list], dtype=np.float64)
        aspect_all = np.array([u.attributes.get("aspect", 0.0) for u in unit_list], dtype=np.float64)
        svf_all = np.array([u.attributes.get("svf", 1.0) for u in unit_list], dtype=np.float64)
        sx_all = np.array([u.attributes.get("sx", np.nan) for u in unit_list], dtype=np.float64)
        z0_all = np.array(
            [self.roughness_lengths.get(u.surface_type, DEFAULT_ROUGHNESS_LENGTHS["open"]) for u in unit_list],
            dtype=np.float64,
        )
        d_all = np.array(
            [self.displacement_heights.get(u.surface_type, DEFAULT_DISPLACEMENT_HEIGHTS["open"]) for u in unit_list],
            dtype=np.float64,
        )

        # --- Chunked vectorized processing ---
        CHUNK_SIZE = 1000
        interp_all = self.interpolate_bilinear_all

        for chunk_start in range(0, n_units, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, n_units)
            cs = slice(chunk_start, chunk_end)

            # Slice bilinear weights for this chunk
            chunk_weights = {k: v[cs] for k, v in bilinear.items()}

            # Bilinear interpolation for surface variables → (n_time, n_chunk)
            surf_chunk = {}
            for var in surf_var_names:
                arr = surf_arrays[var]
                if arr.ndim >= 3:
                    surf_chunk[var] = interp_all(arr, chunk_weights)
                else:
                    # 1D time-only vars: broadcast to (n_time, n_chunk)
                    n_c = chunk_end - chunk_start
                    surf_chunk[var] = np.broadcast_to(
                        arr[:, np.newaxis] if arr.ndim == 1 else np.broadcast_to(arr, (n_time,))[:, np.newaxis],
                        (n_time, n_c),
                    ).copy()

            # Handle scalar z_surf
            n_c = chunk_end - chunk_start
            if z_surf_key not in surf_chunk:
                if z_surf_scalar is not None:
                    val = float(z_surf_scalar) if z_surf_scalar.ndim == 0 else z_surf_scalar
                    surf_chunk[z_surf_key] = np.broadcast_to(
                        np.float64(val), (n_time, n_c),
                    ).copy()

            # Bilinear interpolation for plev variables → (n_time, n_levels, n_chunk)
            plev_chunk = {}
            for var in plev_var_names:
                arr = plev_arrays[var]
                if arr.ndim == 4:
                    plev_chunk[var] = interp_all(arr, chunk_weights)

            # Derived fields for radiation → (n_time, n_chunk)
            sol_elev_chunk = None
            sol_az_chunk = None
            kt_chunk = None
            if "radiation" in active_groups_set:
                sol_elev_chunk = interp_all(sol_elev_arr, chunk_weights) if sol_elev_arr.ndim >= 3 else np.broadcast_to(sol_elev_arr[:, np.newaxis], (n_time, n_c)).copy()
                sol_az_chunk = interp_all(sol_az_arr, chunk_weights) if sol_az_arr.ndim >= 3 else np.broadcast_to(sol_az_arr[:, np.newaxis], (n_time, n_c)).copy()
                kt_chunk = interp_all(kt_arr, chunk_weights) if kt_arr.ndim >= 3 else np.broadcast_to(kt_arr[:, np.newaxis], (n_time, n_c)).copy()

            # Common scalars needed by groups
            z_source_chunk = surf_chunk.get(z_surf_key)
            t_source_chunk = surf_chunk.get("t2m")
            p_source_chunk = surf_chunk.get("sp")
            d2m_chunk = surf_chunk.get("d2m")

            # Slice output arrays — writes go directly into pre-allocated memory
            chunk_out = {var: out_arrays[var][:, cs] for var in output_vars}

            self._process_all(
                chunk_out, surf_chunk, plev_chunk, active_groups_set,
                z_target_all[cs], z_source_chunk, t_source_chunk, p_source_chunk,
                slope_all[cs], aspect_all[cs], svf_all[cs],
                z0_all[cs], d_all[cs], sx_all[cs],
                sol_elev_chunk, sol_az_chunk, kt_chunk, d2m_chunk,
                self.mode,
            )

            # Copy chunk results back (chunk_out is a view into out_arrays)
            for var in output_vars:
                out_arrays[var][:, cs] = chunk_out[var]

            if progress_callback is not None:
                progress_callback(chunk_end, n_units)

        # --- Build output dataset from pre-allocated arrays ---
        unit_ids = [u.id for u in unit_list]
        ds_out = xr.Dataset(
            {var: (["time", "unit"], out_arrays[var]) for var in output_vars},
            coords={
                "time": time_coord,
                "unit": unit_ids,
                "latitude": ("unit", [u.y for u in unit_list]),
                "longitude": ("unit", [u.x for u in unit_list]),
                "elevation": ("unit", [u.elevation for u in unit_list]),
            },
        )

        elapsed = time_module.time() - start
        log.info("Downscaled %d units in %.1f seconds (%.3f s/unit)", n_units, elapsed, elapsed / n_units)

        return ds_out


def downscale_ensemble(downscaler, units, forcing, dem_data, **kwargs):
    """Downscale a member-dimensioned (ENS) forecast forcing, member by member.

    If ``forcing`` has no ``member`` dimension this is just ``downscaler.process``
    (the deterministic/HRES path). Otherwise each member is downscaled
    independently — reusing the full vectorized pipeline — and the per-member
    outputs are concatenated along a ``member`` dimension for the probabilistic
    products in :mod:`topopyscale2.outputs.ensemble`.

    Returns an :class:`xarray.Dataset`; with a ``member`` dimension when the input
    has one.
    """
    if "member" not in getattr(forcing, "dims", {}):
        return downscaler.process(units, forcing, dem_data, **kwargs)
    outs = []
    members = [int(m) for m in forcing["member"].values]
    for i, m in enumerate(members):
        log.info("ENS member %d/%d (id=%d)", i + 1, len(members), m)
        out = downscaler.process(units, forcing.sel(member=m), dem_data, **kwargs)
        outs.append(out.assign_coords(member=m))
    return xr.concat(outs, dim="member")
