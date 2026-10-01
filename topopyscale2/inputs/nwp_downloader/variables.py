"""Variable registry and name mappings for ERA5 and IFS data."""

from __future__ import annotations

# ── ERA5 variable name mappings ──────────────────────────────────────────
# Maps CDS long names → short names used in output files

SURF_VARS_CDS = {
    "geopotential": "z",
    "2m_dewpoint_temperature": "d2m",
    "surface_thermal_radiation_downwards": "strd",
    "surface_solar_radiation_downwards": "ssrd",
    "surface_pressure": "sp",
    "total_precipitation": "tp",
    "2m_temperature": "t2m",
}

PLEV_VARS_CDS = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "relative_humidity": "r",
}

# Google ARCO-ERA5 uses the same long names for URI paths
SURF_VARS_GOOGLE = SURF_VARS_CDS.copy()
PLEV_VARS_GOOGLE = PLEV_VARS_CDS.copy()

# Default surface variables required for downscaling
DEFAULT_SURF_VARS = list(SURF_VARS_CDS.keys())

# Default pressure level variables required for downscaling
DEFAULT_PLEV_VARS = list(PLEV_VARS_CDS.keys())

# Google ARCO-ERA5: variables that exist as both surface and pressure level
VARS_BOTH_SURFACE_AND_LEVEL = {"geopotential"}

# ── IFS forecast variable mappings ───────────────────────────────────────

IFS_SURF_PARAMS = ["2t", "sp", "2d", "ssrd", "strd", "tp", "msl"]
IFS_PLEV_PARAMS = ["gh", "u", "v", "r", "q", "t"]
IFS_DEFAULT_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300]

# IFS → ERA5-compatible renaming
IFS_RENAME_MAP = {
    "2t": "t2m",
    "2d": "d2m",
    "lat": "latitude",
    "lon": "longitude",
    "plev": "level",
    "isobaricInhPa": "level",
}

# ── CDS new API renaming (post mid-2024) ────────────────────────────────

CDS_RENAME_SURF = {"valid_time": "time"}
CDS_RENAME_PLEV = {"valid_time": "time", "pressure_level": "level"}
CDS_DROP_VARS = ["number", "expver"]

# ── Standard dimension names ────────────────────────────────────────────

STANDARD_DIMS = {
    "time": "time",
    "latitude": "latitude",
    "longitude": "longitude",
    "level": "level",
}

# Accumulated variables (need special handling for deaccumulation)
ACCUMULATED_VARS = {"ssrd", "strd", "tp"}

# ── Time resolution mappings ────────────────────────────────────────────

TIME_RESOLUTION_HOURS = {
    "1H": list(range(24)),
    "2H": list(range(0, 24, 2)),
    "3H": list(range(0, 24, 3)),
    "6H": list(range(0, 24, 6)),
}

TIME_RESOLUTION_STRINGS = {
    "1H": [f"{h:02d}:00" for h in range(24)],
    "2H": [f"{h:02d}:00" for h in range(0, 24, 2)],
    "3H": [f"{h:02d}:00" for h in range(0, 24, 3)],
    "6H": [f"{h:02d}:00" for h in range(0, 24, 6)],
}
