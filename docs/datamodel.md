# TPS2 Data Model

This document describes how meteorological variables flow through TPS2, from raw NWP data to impact-model-ready output.

---

## Overview

TPS2 has a three-layer data model:

```
NWP native         Converted (internal)        Canonical output
(ERA5, IFS, ...)   (ERA5 short names,          (descriptive names,
                    SI units)                    SI units)
      │                    │                          │
  backends +           downscaler                 writers +
  units.py             consumes                   applications
                       these                      consume these
```

All NWP sources are converted to the internal representation before reaching the downscaler. The downscaler produces canonical output variables consumed by all writers and applications. This means:

- **Adding a new NWP source** (e.g., IFS forecasts, GFS) only requires mapping its native names/units to the internal representation in a backend + unit converter.
- **Adding a new output format** (e.g., a new snow model) only requires mapping from the canonical output variables.
- **The downscaler never changes.** It speaks one input dialect and one output dialect.

---

## Layer 1: NWP-Native Variables

Each backend downloads data in the NWP's native naming convention. Variable name mappings are defined in `topopyscale2/inputs/nwp_downloader/variables.py`.

### ERA5 / IFS surface variables

| NWP name | Short name | Description | Native units |
|----------|-----------|-------------|-------------|
| `2m_temperature` | `t2m` | 2m air temperature | K |
| `2m_dewpoint_temperature` | `d2m` | 2m dewpoint temperature | K |
| `surface_pressure` | `sp` | Surface pressure | Pa |
| `surface_solar_radiation_downwards` | `ssrd` | SW radiation (accumulated) | J/m² |
| `surface_thermal_radiation_downwards` | `strd` | LW radiation (accumulated) | J/m² |
| `total_precipitation` | `tp` | Precipitation (accumulated) | m |
| `geopotential` | `z` → `z_surf` | Surface geopotential | m²/s² |
| `10m u-wind` | `u10` | 10m u-wind (simple mode) | m/s |
| `10m v-wind` | `v10` | 10m v-wind (simple mode) | m/s |
| `toa_incident_solar_radiation` | `tisr` | TOA solar (optional) | J/m² |

### ERA5 / IFS pressure-level variables

| Short name | Description | Native units |
|-----------|-------------|-------------|
| `t` | Temperature | K |
| `z` | Geopotential | m²/s² |
| `u` | U-wind component | m/s |
| `v` | V-wind component | m/s |
| `q` | Specific humidity | kg/kg |
| `r` | Relative humidity | % or 0–1 |

### Accumulated variable handling

ERA5 `tp`, `ssrd`, `strd` are 1-hour accumulations regardless of the data's time resolution. When a backend subsamples to coarser resolution (e.g., 3H), it must scale accumulated precipitation by the step ratio so that `tp` represents the total for the timestep interval. Radiation is kept as a 1-hour snapshot (divided by 3600s to get W/m²), which is an acceptable approximation for sub-daily forcing.

---

## Layer 2: Converted Internal Variables

`topopyscale2/inputs/units.py` converts NWP-native units to standard SI units. **Variable names stay the same** (ERA5 short names); only values change.

| Variable | Conversion | Post-conversion units |
|----------|-----------|----------------------|
| `ssrd`, `strd`, `tisr` | ÷ 3600 | W/m² |
| `tp` | × 1000 | mm (total per timestep) |
| `z`, `z_surf` | ÷ 9.80665 | m (geopotential height) |
| `r` | ÷ 100 if in % | 0–1 (fractional) |
| `t2m`, `d2m`, `sp`, `u10`, `v10`, `t`, `u`, `v`, `q` | unchanged | K, Pa, m/s, kg/kg |

### Derived variables (computed before downscaling)

| Variable | Description | Units | Source |
|----------|-------------|-------|--------|
| `solar_elevation` | Solar elevation angle | degrees | `topopyscale2/inputs/derived.py` |
| `solar_azimuth` | Solar azimuth angle | degrees | `topopyscale2/inputs/derived.py` |
| `clearness_index` | Clearness index kt | – | `topopyscale2/inputs/derived.py` |

### Input validation

The downscaler validates inputs at entry. Required variables are defined as `REQUIRED_SURFACE_VARS_FULL`, `REQUIRED_SURFACE_VARS_SIMPLE`, and `REQUIRED_PLEV_VARS` in `topopyscale2/core/downscale.py`. A `ValueError` is raised with a clear message if any are missing.

---

## Layer 3: Canonical Output Variables

The downscaler produces these 12 variables, defined in the `OutputVariable` enum (`topopyscale2/outputs/base.py`). All output writers and application wrappers consume this format.

| Variable name | CF standard_name | Units | Description |
|--------------|------------------|-------|-------------|
| `temperature` | `air_temperature` | K | Near-surface air temperature |
| `precipitation` | `precipitation_amount` | kg/m² (= mm) | Total precipitation per timestep |
| `rainfall` | `rainfall_amount` | kg/m² | Liquid precipitation per timestep |
| `snowfall` | `snowfall_amount` | kg/m² | Solid precipitation per timestep |
| `shortwave_direct` | `surface_direct_downwelling_shortwave_flux_in_air` | W/m² | Direct SW on slope |
| `shortwave_diffuse` | `surface_diffuse_downwelling_shortwave_flux_in_air` | W/m² | Diffuse SW radiation |
| `longwave` | `surface_downwelling_longwave_flux_in_air` | W/m² | Downwelling LW radiation |
| `humidity_specific` | `specific_humidity` | kg/kg | Specific humidity |
| `humidity_relative` | `relative_humidity` | 0–1 | Relative humidity (fraction) |
| `wind_speed` | `wind_speed` | m/s | Wind speed |
| `wind_direction` | – | degrees | Wind direction (meteorological) |
| `pressure` | `surface_air_pressure` | Pa | Surface air pressure |

### Dimensions

- **`time`**: Temporal dimension, preserves the NWP time resolution.
- **`unit_id`**: Spatial dimension, one entry per spatial unit (cluster or polygon).

### CF attributes

Every output variable has `standard_name`, `long_name`, and `units` attributes defined in `CF_ATTRIBUTES` (`topopyscale2/outputs/base.py`). Generic writers (NetCDF, Zarr) apply these automatically.

---

## Output Writers

Writers consume canonical variables and either preserve them or map to model-specific conventions.

| Writer | Format | Variable handling |
|--------|--------|-------------------|
| `CFNetCDFWriter` | NetCDF-4 | Preserves canonical names + CF attributes |
| `ZarrWriter` | Zarr | Preserves canonical names + CF attributes |
| `CSVWriter` | CSV | Preserves canonical names |
| `FSMWriter` | Text (per unit) | Maps to: year month day hour SW LW Sf Rf Ta RH Ua Ps — used only for the **Fortran** FSM1 backend; the default **fsm-rs** (Rust) engine consumes the forcing xarray directly and skips this writer |
| `FSM2Writer` | Text (all units) | Maps to: year month day hour SW LW Sf Rf Ta RH Ua Ps — used for FSM2 (Fortran subprocess) |
| `SMETWriter` | SMET text | Maps to SNOWPACK variable names |
| `CrocusWriter` | NetCDF | Maps to SAFRAN/Crocus variable names |
| `CryoGridWriter` | NetCDF | Maps to CryoGrid variable names |
| `HBVWriter` | Text (daily) | Uses only temperature + precipitation |

### Writer-specific conversions

FSM writers convert precipitation from mm/timestep to kg/m²/s (rate) by dividing by the timestep in seconds. The timestep is inferred from the forcing data's time coordinate — not hardcoded.

---

## Adding a New NWP Source

To add a new NWP source (e.g., IFS forecasts):

1. **Create a backend** in `topopyscale2/inputs/nwp_downloader/` that fetches the data and renames variables to ERA5 short names.
2. **Add any needed renaming** to `variables.py` (see `IFS_RENAME_MAP` for an existing example).
3. **Ensure accumulated variables** are scaled correctly for the timestep interval.
4. **The existing `convert_era5_surface()` / `convert_era5_pressure()`** will handle the rest, since they operate on short names.

No changes needed to the downscaler, output writers, or applications.

---

## Adding a New Output Format

To add a new impact model writer:

1. **Create a writer class** in `topopyscale2/outputs/` following the `OutputWriter` protocol.
2. **Consume the canonical variable names** from the forcing dataset.
3. **Map to your model's conventions** (variable names, units, file format).
4. **Register in `topopyscale2/outputs/__init__.py`** if needed.

No changes needed to the downscaler or input pipeline.
