# Downscaling Algorithms: TopoPyScale vs TPS2

This document compares the downscaling algorithms in the original TopoPyScale with our TPS2 reimplementation, identifying gaps and proposing fixes.

## Summary of Differences

| Variable | TopoPyScale | TPS2 Full Mode | TPS2 Simple Mode | Status |
|----------|-------------|----------------|------------------|--------|
| Temperature | Pressure-level interpolation | Pressure-level interpolation | Fixed lapse rate | ✅ Implemented |
| Pressure | Hypsometric from upper level | Hypsometric from surface | Hypsometric from surface | ✅ OK |
| Wind | Pressure-level interpolation | Pressure-level + log-profile | 10m log-profile only | ✅ Implemented |
| SW Direct | Beer's law attenuation | Simple slope correction | Simple slope correction | **Missing attenuation** |
| SW Diffuse | Erbs + SVF | Erbs + SVF | Erbs + SVF | ✅ OK |
| Longwave | Emissivity model (clear+cloud) | T^4 ratio scaling | T^4 ratio scaling | **Simplified** |
| Precipitation | Monthly-varying gradient | Pass-through (gradient opt-in) | Pass-through (gradient opt-in) | **Simplified** |
| Humidity | Magnus formula + interpolation | Pressure-level interpolation | Conserve q from dewpoint | ✅ Implemented |

---

## Downscaling Modes

TPS2 supports two downscaling modes configured via `downscaling.mode` in the config:

### Full Mode (default)

**Requires**: ERA5 pressure-level data (t, z, u, v, q at multiple pressure levels)

```yaml
downscaling:
  mode: full
```

**Temperature**: Interpolates between pressure levels bracketing the target elevation using inverse distance weighting. Captures actual atmospheric profile including inversions.

```
For target elevation z_target:
1. Find levels above (z_top) and below (z_bot) the target
2. Compute weights: w_top = (z_target - z_bot) / (z_top - z_bot)
3. Interpolate: T_target = w_bot * T_bot + w_top * T_top
```

**Wind**: Interpolates u, v components from pressure levels, then applies log-profile correction for surface roughness.

**Humidity**: Interpolates specific humidity from pressure levels, recomputes relative humidity at target T and P.

### Simple Mode (ERA5-Land compatible)

**Requires**: Only surface data (t2m, sp, d2m, u10, v10, ssrd, strd, tp)

```yaml
downscaling:
  mode: simple
  lapse_rate: 0.0065  # K/m, default 6.5 K/km
```

**Temperature**: Applies a fixed environmental lapse rate correction:

```
T_target = T_surface - lapse_rate × (z_target - z_surface)
```

Where:
- `lapse_rate` = 0.0065 K/m (6.5 K/km) by default
- Configurable for different conditions (e.g., 0.005 for moist adiabatic)

**Wind**: Uses 10m wind (u10, v10) with log-profile correction for surface roughness differences.

**Humidity**: Computes specific humidity from dewpoint temperature, conserves q during vertical displacement, recomputes RH at target T and P.

### Mode Comparison

| Aspect | Full Mode | Simple Mode |
|--------|-----------|-------------|
| Data requirements | ERA5 pressure levels | ERA5-Land surface only |
| Temperature accuracy | High (captures inversions) | Moderate (assumes linear lapse) |
| Computational cost | Higher | Lower |
| Use case | Research, validation | Operational, large domains |

### When to Use Each Mode

**Use Full Mode when:**
- Accuracy is critical
- You have ERA5 pressure-level data
- Domain includes complex inversions (valleys, nocturnal)
- Validating against station observations

**Use Simple Mode when:**
- Using ERA5-Land (no pressure levels)
- Running large domains where speed matters
- Initial exploratory runs
- The domain has relatively uniform lapse rates

---

## 1. Temperature

### TopoPyScale Algorithm

**Approach**: Vertical interpolation between pressure levels bracketing the target elevation.

```python
# 1. Find pressure levels above and below target elevation
ind_z_bot = (plev.where(plev.z < target_elev).z - target_elev).argmax('level')
ind_z_top = (plev.where(plev.z > target_elev).z - target_elev).argmin('level')

top = plev.isel(level=ind_z_top)
bot = plev.isel(level=ind_z_bot)

# 2. Distance-weighted interpolation
dist = [|bot.z - target_elev|, |top.z - target_elev|]
weights = dist / sum(dist)  # Inverse distance weighting

T_target = bot.t * weights[1] + top.t * weights[0]
```

**Key properties**:
- Uses actual atmospheric profile at each timestep
- Adapts to inversions and non-linear lapse rates
- Falls back to nearest level when target is outside pressure level range

### TPS2 Current Algorithm (Full Mode) ✅ IMPLEMENTED

```python
# Pressure-level interpolation (same approach as TopoPyScale)
t_target = kernels.interpolation.interpolate_pressure_levels(t_plev, z_plev, z_target)
```

The `interpolate_pressure_levels` function:
1. Finds pressure levels above and below target elevation
2. Computes inverse-distance weights
3. Interpolates temperature between bracketing levels
4. Handles edge cases (extrapolation when outside level range)

### TPS2 Simple Mode Algorithm

```python
# Fixed lapse rate correction (ERA5-Land compatible)
T_target = T_surface - lapse_rate * (z_target - z_surface)
```

Where `lapse_rate` defaults to 0.0065 K/m (6.5 K/km).

### Reference Implementation

```python
def interpolate_temperature(t_plev, z_plev, z_target):
    """Interpolate temperature from pressure levels to target elevation.

    Parameters
    ----------
    t_plev : array (time, level)
        Temperature at pressure levels [K]
    z_plev : array (time, level)
        Geopotential height at pressure levels [m]
    z_target : float
        Target elevation [m]

    Returns
    -------
    t_target : array (time,)
        Interpolated temperature [K]
    """
    # For each timestep, find bracketing levels and interpolate
    for t in range(n_time):
        z = z_plev[t, :]
        above = z > z_target
        below = z < z_target

        if above.any() and below.any():
            # Bracketed: interpolate
            i_top = np.where(above, z - z_target, np.inf).argmin()
            i_bot = np.where(below, z_target - z, np.inf).argmin()

            d_top = z[i_top] - z_target
            d_bot = z_target - z[i_bot]
            w_top = d_bot / (d_top + d_bot)
            w_bot = d_top / (d_top + d_bot)

            t_target[t] = w_bot * t_plev[t, i_bot] + w_top * t_plev[t, i_top]
        else:
            # Extrapolate from nearest level
            i_nearest = np.abs(z - z_target).argmin()
            t_target[t] = t_plev[t, i_nearest]

    return t_target
```

---

## 2. Pressure

### TopoPyScale Algorithm

Uses hypsometric equation from the pressure level above:

```python
# p_target from level above using mean temperature
p_target = p_top * exp(-(z_target - z_top) / (0.5 * (T_top + T_target) * R / g))
```

Where:
- `p_top` = pressure of level above [Pa]
- `z_top` = geopotential height of level above [m]
- `R` = 287.05 J/(kg·K) (gas constant for dry air)
- `g` = 9.81 m/s²

### TPS2 Current Algorithm (Both Modes) ✅ IMPLEMENTED

Uses hypsometric equation from surface pressure with mean temperature:

```python
# Mean temperature between source and target for better accuracy
t_mean = (t_source + t_target) / 2
dz = z_target - z_source
p_target = p_source * exp(-g * dz / (R_DRY * t_mean))
```

Where:
- `t_source` = surface temperature (t2m) [K]
- `t_target` = downscaled temperature at target elevation [K]
- `R_DRY` = 287.05 J/(kg·K)
- `g` = 9.80665 m/s²

### Assessment

Both use the hypsometric equation. TPS2 uses the mean of source and target temperature for improved accuracy over large elevation differences. TopoPyScale's use of the nearest pressure level above may be marginally more accurate for very high-elevation targets, but the difference is small in practice.

---

## 3. Wind

### TopoPyScale Algorithm

Interpolates u, v components between pressure levels:

```python
# Same interpolation as temperature
u_target = bot.u * weights[1] + top.u * weights[0]
v_target = bot.v * weights[1] + top.v * weights[0]

ws = sqrt(u_target^2 + v_target^2)
wd = atan2(-u_target, -v_target)
```

### TPS2 Current Algorithm (Full Mode) ✅ IMPLEMENTED

Combines pressure-level interpolation with surface roughness correction:

```python
# 1. Interpolate u, v from pressure levels (same as TopoPyScale)
u_target, v_target = kernels.interpolation.interpolate_wind_components(
    u_plev, v_plev, z_plev, z_target
)

# 2. Convert to speed and direction
wind_speed = sqrt(u_target^2 + v_target^2)
wind_direction = atan2(-u_target, -v_target)

# 3. Apply log-profile correction from free-atmosphere to surface
# Assumes interpolated wind is at ~50m equivalent height
wind_speed_corr = kernels.wind.log_profile_correction(
    wind_speed,
    z_source=50.0,      # Interpolated wind height
    z_target=10.0,      # Target: 10m equivalent
    z0_source=0.01,     # Free-atmosphere roughness
    z0_target=z0_surface,  # Surface roughness by type
    d_source=0.0,
    d_target=d_surface,
)

# 4. Optional Winstral Sx correction for terrain exposure
if sx is not None and method == "winstral":
    wind_speed_corr = kernels.wind.winstral_wind_correction(wind_speed_corr, sx)
```

### TPS2 Simple Mode Algorithm

Uses 10m wind directly with surface roughness correction:

```python
# 1. Get 10m wind components from surface data
u10, v10 = surf_forcing["u10"], surf_forcing["v10"]
wind_speed = sqrt(u10^2 + v10^2)
wind_direction = atan2(-u10, -v10)

# 2. Log-profile correction for surface type
wind_speed_corr = kernels.wind.log_profile_correction(
    wind_speed,
    z_source=10.0,       # ERA5 10m
    z_target=10.0,       # Target 10m equivalent
    z0_source=0.03,      # ERA5 open terrain
    z0_target=z0_surface,
    d_source=0.0,
    d_target=d_surface,
)
```

### Assessment

✅ TPS2 Full Mode now matches TopoPyScale approach:
- Pressure-level interpolation captures wind speed increase with elevation
- Surface roughness correction adapts to local terrain
- Winstral Sx correction adds terrain exposure effects

---

## 4. Shortwave Radiation

### TopoPyScale Algorithm

**Direct radiation with Beer's law attenuation**:

```python
# 1. Compute atmospheric attenuation coefficient from source conditions
ka = (g * mu0 / p_source) * ln(SW_toa / SW_direct_source)

# 2. Apply Beer's law at target pressure
SW_direct_target = SW_toa * exp(-ka * p_target / (g * mu0))

# 3. Apply terrain correction
cos_illumination = (mu0 * cos(slope) +
                    sin(zenith) * sin(slope) * cos(azimuth - aspect))
SW_direct_target *= (cos_illumination / mu0) * (1 - shadow)
```

**Diffuse radiation**:

```python
# Erbs correlation for diffuse fraction
kd = 0.952 - 1.041 * exp(-exp(2.3 - 4.702 * kt))
SW_diffuse = svf * kd * SW_total
```

### TPS2 Current Algorithm (Both Modes)

```python
# Erbs partition (same as TopoPyScale)
kd = erbs_correlation(kt)
sw_diffuse = kd * sw_total
sw_direct = sw_total - sw_diffuse

# Slope correction without Beer's law
cos_incidence = sin(solar_elev) * cos(slope) +
                cos(solar_elev) * sin(slope) * cos(solar_az - aspect)
sw_direct_slope = sw_direct * (cos_incidence / sin(solar_elev))

# SVF correction
sw_diffuse_corr = sw_diffuse * svf
```

### Gap: Missing Beer's Law Attenuation (TODO)

TPS2 applies the same direct radiation intensity regardless of elevation. At higher elevations, the atmosphere is thinner and direct radiation should be stronger.

**Recommendation**: Add Beer's law elevation correction:

```python
def elevation_correction(sw_direct, p_source, p_target, solar_elev):
    """Correct direct SW for atmospheric path length difference."""
    mu0 = np.sin(np.radians(solar_elev))
    mu0 = np.maximum(mu0, 0.01)

    # Relative air mass correction
    # At higher elevation (lower pressure), less attenuation
    correction = (p_source / p_target) ** (1 / mu0)

    return sw_direct * np.minimum(correction, 2.0)  # Cap at 2x
```

---

## 5. Longwave Radiation

### TopoPyScale Algorithm

**Emissivity-based model**:

```python
# Clear-sky emissivity (Brutsaert 1975)
cse = 0.23 + 0.43 * (vp / T) ** (1/5.7)

# Cloud emissivity from surface LW measurement
cle = (LW_source / (sigma * T_source^4)) - cse_source

# All-sky emissivity at target
aef = cse_target + cle

# Final LW with terrain contribution
LW_target = (svf * aef * sigma * T_target^4 +
             (1 - svf) * 0.99 * sigma * T_terrain^4)
```

### TPS2 Current Algorithm (Both Modes)

```python
# Simple T^4 ratio for sky component (Stefan-Boltzmann scaling)
lw_sky = lw_source * (t_target / t_source)^4

# Terrain emission (surrounding terrain at local temperature)
lw_terrain = sigma * t_target^4

# SVF weighting: sky contribution + terrain contribution
lw_target = svf * lw_sky + (1 - svf) * lw_terrain
```

Where `sigma` = 5.67×10⁻⁸ W/(m²·K⁴) (Stefan-Boltzmann constant).

### Assessment

TPS2 is simplified but reasonable for most applications. The T^4 ratio approximates the emissivity change with temperature. The SVF weighting correctly accounts for reduced sky view in complex terrain.

For higher accuracy, could add clear-sky emissivity model (Brutsaert 1975), but current approach is acceptable for most applications.

---

## 6. Precipitation

### TopoPyScale Algorithm

**Monthly-varying elevation gradient**:

```python
monthly_coef = [0.35, 0.35, 0.35, 0.30, 0.25, 0.20,
                0.20, 0.20, 0.20, 0.25, 0.30, 0.35]

dz_km = (z_target - z_source) / 1000
lapse_factor = (1 + coef * dz_km) / (1 - coef * dz_km)
precip_target = precip_source * lapse_factor
```

### TPS2 Current Algorithm (Both Modes)

**Default: no elevation gradient.** A fixed precip-elevation gradient is poorly
constrained in high-relief terrain (the precip-elevation relationship is
non-monotonic and seasonally variable), so source precipitation is passed
through unchanged unless the gradient is explicitly enabled. To opt in, set
`downscaling.precipitation: elevation_gradient` and tune `downscaling.precip_gradient`.

```python
# Optional elevation gradient (opt-in via config; default precip_gradient = 0.0)
gradient = 0.0003  # +3%/100m = +30%/km (legacy default, applied only when enabled)
dz = z_target - z_source
precip_target = max(precip_source * (1 + gradient * dz), 0.0)

# Phase partition based on a partition temperature t_phase
rain_fraction = (t_phase - t_snow) / (t_rain - t_snow)  # Linear transition
rainfall = precip_target * rain_fraction
snowfall = precip_target * (1 - rain_fraction)
```

**Partition temperature.** By default `t_phase` is the **psychrometric
wet-bulb temperature** (`downscaling.phase_method: wet_bulb`), not the air
temperature. A falling hydrometeor equilibrates toward the wet-bulb temperature,
so in dry air snow can fall at air temperatures well above 0°C — the wet-bulb
solve (pressure-aware, so valid at high-elevation low pressures) captures this.
Set `phase_method: air_temperature` to partition on 2 m air temperature instead.

Default thresholds (configurable via `t_snow_threshold_c` / `t_rain_threshold_c`),
recentred for wet-bulb which runs below air temperature:
- `t_rain` = 275.65 K (+2.5°C) — all rain above this
- `t_snow` = 272.65 K (−0.5°C) — all snow below this

**Air-temperature guard rail.** The wet-bulb ramp caps on the wet-bulb
temperature, which in very dry air permits snow at high *air* temperatures (a
+2.5°C wet-bulb can correspond to a +7°C air temperature at low RH). To restore
the air-temperature ceiling that `t_rain` gives the air-temperature method,
precipitation is forced to all-rain above `t_air_max` (`t_air_max_c`, default
+4°C) regardless of wet-bulb. This is a no-op for the air-temperature method
(where `t_rain` already caps air temp) and for saturated air; it only clips the
implausible dry-air warm tail. Set `t_air_max_c` high (e.g. 99) to disable.

### Recommendation (TODO)

Add monthly-varying coefficients to precipitation kernel:

```python
MONTHLY_PRECIP_COEF = [0.35, 0.35, 0.35, 0.30, 0.25, 0.20,
                       0.20, 0.20, 0.20, 0.25, 0.30, 0.35]

def elevation_gradient_monthly(p_source, z_unit, z_source, month):
    coef = MONTHLY_PRECIP_COEF[month - 1]
    dz_km = (z_unit - z_source) / 1000
    lapse_factor = (1 + coef * dz_km) / (1 - coef * dz_km)
    return np.maximum(p_source * lapse_factor, 0.0)
```

---

## 7. Humidity

### TopoPyScale Algorithm

Uses Magnus formula to convert between dewpoint and specific humidity, then interpolates between pressure levels.

### TPS2 Current Algorithm (Full Mode) ✅ IMPLEMENTED

Interpolates specific humidity from pressure levels, then recomputes relative humidity:

```python
# 1. Interpolate specific humidity from pressure levels
q_target = kernels.interpolation.interpolate_pressure_levels(q_plev, z_plev, z_target)

# 2. Compute saturation vapor pressure at target temperature (Magnus formula)
e_sat = 611.2 * exp(17.67 * (t_target - 273.15) / (t_target - 29.65))

# 3. Compute actual vapor pressure from specific humidity and target pressure
e_actual = q_target * p_target / (0.622 + 0.378 * q_target)

# 4. Relative humidity
rh_target = e_actual / e_sat
```

### TPS2 Simple Mode Algorithm

Computes specific humidity from surface dewpoint, conserves it during vertical displacement:

```python
# 1. Compute specific humidity from dewpoint (Magnus formula)
e_sat_d = 611.2 * exp(17.67 * (d2m - 273.15) / (d2m - 29.65))
q_source = 0.622 * e_sat_d / (p_source - 0.378 * e_sat_d)

# 2. Assume specific humidity conserved
q_target = q_source

# 3. Recompute RH at target T and P
e_sat_t = 611.2 * exp(17.67 * (t_target - 273.15) / (t_target - 29.65))
e_actual = q_target * p_target / (0.622 + 0.378 * q_target)
rh_target = clip(e_actual / e_sat_t, 0, 1)
```

### Assessment

Both modes use physically reasonable approaches:
- Full mode: Interpolates q from atmospheric profile (captures moisture inversions)
- Simple mode: Conserves q from surface (approximation, but reasonable for most cases)

RH changes appropriately with temperature and pressure in both modes.

---

## Priority Fixes

1. ~~**HIGH**: Temperature - switch to pressure-level interpolation~~ ✅ DONE (Full mode)
2. **HIGH**: Shortwave direct - add Beer's law elevation correction
3. **MEDIUM**: Precipitation - add monthly-varying coefficients
4. **LOW**: Longwave - add clear-sky emissivity model
5. ~~**LOW**: Wind - combine pressure-level interpolation with surface correction~~ ✅ DONE (Full mode)

---

## Implementation Plan

### Step 1: Temperature Kernel Update ✅ DONE

Updated all three backends with two methods:
- `simple_lapse_rate(t_surface, z_surface, z_target, lapse_rate)` - Simple mode
- `interpolate_pressure_levels(t_plev, z_plev, z_target)` - Full mode (in interpolation module)

Files updated:
- `python_kernels/temperature.py` ✅
- `jax_kernels/temperature.py` ✅
- `rust_kernels/src/temperature.rs` ✅
- `python_kernels/interpolation.py` ✅

### Step 2: Wind Interpolation ✅ DONE

Full mode now interpolates u, v from pressure levels before applying log-profile correction.

### Step 3: Radiation Kernel Update (TODO)

Add to all three backends:
- `elevation_correction(sw_direct, p_source, p_target, solar_elev) -> sw_direct_corrected`

### Step 4: Precipitation Kernel Update (TODO)

Add to all three backends:
- `elevation_gradient_monthly(p_source, z_unit, z_source, month) -> p_corrected`

### Step 5: Downscaler Updates ✅ DONE

Modified `core/downscale.py`:
- Added `mode` parameter ("simple" or "full")
- Added `process_unit()` for full mode with pressure-level interpolation
- Added `process_unit_simple()` for simple mode with fixed lapse rate
