# Known limitations

TPS2 is used operationally, and it is wrong in known, measured ways. This page lists them so
that you can judge whether a result is fit for your purpose before you rely on it.

Every entry says **what goes wrong, by how much, where it was measured, and what to do about
it**. The numbers come from specific runs against specific observations. They show the size of
each effect, not a guaranteed error bound for your domain, and they change when the engine does.
Entries marked *open* are unfixed on the current release.

---

## Forcing (the downscaling engine)

### Precipitation amount is ERA5's, and ERA5 is often wrong in mountains

TPS2 partitions precipitation into rain and snow, but by default it does **not** change the
amount. Each unit receives the precipitation of the ERA5 cell it sits in (the
`elevation_gradient` option is off by default). ERA5's ~31 km cells cannot resolve the
precipitation structure of mountain terrain:

| Where | What | Size |
|---|---|---|
| Dry inner-alpine valley (Mattertal, Zermatt, 1600 m) | wet-side precipitation smeared into a rain-shadow valley | ~1760 mm in 10 months against ~650–700 mm/yr climatology: **~2.5× too wet**; 2 m of snow on the valley floor into July |
| High Himalaya (Khumbu, 2660–5035 m, 13 matched years) | vertical gradient far too weak | right at 2660 m (0.93×), **2.5–2.7× too wet above 4000 m**; modelled gradient −2.5 %/100 m against −6.8 %/100 m observed |
| Weissfluhjoch (2536 m, 25 winters) | too little in big winters | modelled peak SWE rises at **half** the observed interannual rate (slope 0.49); the deficit is already in the ERA5 precipitation, not in the snow model |

**The built-in `elevation_gradient` cannot fix the Himalayan case.** It scales by the
difference between the unit elevation and ERA5's *smoothed* orography, which can be badly
wrong: ERA5 puts Lukla (true 2660 m) at 4454 m. A **single precipitation multiplier cannot fix
the Weissfluhjoch case** either, because the error grows with the size of the winter.

**What to do:** treat snow amount as the least certain output. Check it against any
precipitation, snow-course or glacier winter-balance data you have before drawing conclusions
from absolute amounts. Relative patterns (timing, elevation structure) are more robust than totals.

### Pressure levels must bracket the terrain, and more levels help

In `full` mode, temperature, humidity and wind are interpolated between pressure levels, and
the interpolation **clamps** above the top level instead of extrapolating. With
`[700, 850, 1000]` everything above ~3100 m becomes isothermal and several K too warm (measured
on a 1600–4100 m glacier domain: identical JJA mean temperature at 3200–3600 m and 3600–4000 m).
Use `[300, 500, 700, 850, 1000]` or more; see `inputs.pressure_levels` in the
[configuration reference](reference/config.md).

Level *count* matters beyond bracketing. Twenty levels instead of five (same store, 17
stations, one year) improved relative humidity by 3.4–5.0 points RMSE at 2100–2700 m and wind by
0.3–0.7 m/s RMSE at high stations; temperature barely changed.

### Diurnal temperature range is too small, and valley cold pools are missing

Pressure-level interpolation follows the free atmosphere, not the surface boundary layer:

- At 5035 m (Pyramid, Khumbu, 2002–2025) the modelled daily range is **35 %** of observed
  (2.7 vs 7.7 °C), and the warm-season Tmax trend has the wrong sign. Night-time is fine.
- At 4200 m on a Pamir glacier the model is **−2.6 °C too cold on summer days** (−5.4 °C at
  09:00), with no bias at night. Glacier surface energy balance and melt are sensitive to exactly
  this.
- A valley station at 626 m is **+9.2 K too warm on winter nights**. The inversion is missing,
  and adding pressure levels does not bring it back (0.15 K).

**What to do:** be cautious with daily extremes, frost and inversion-dependent results, and
with melt at high elevation in summer.

### Wind is an hourly terrain-unit mean, 20–30 % low, and zero in forest (*open*)

`wind_speed` is an hourly mean over a terrain unit, not a gust, and runs about 20–30 % below
station measurements. Over a full Himalayan water year its domain maximum was 10.7 m/s, so any
absolute strong-wind threshold (~15 m/s and up) would never be reached.

*Open bug:* for units with surface type `forest` (roughness 2 m) the 10 m reference height falls
below the displacement height and `wind_log_profile` returns **0 for every hour**. This only
affects runs that stratify units by surface type.

**What to do:** compare wind with its own climatology ("windier than usual here"), never with
absolute thresholds. Check forest units if you stratify by land cover.

### Sub-daily precipitation timing is inherited

ERA5 and IFS time convective precipitation poorly. Over the South Asian monsoon the daily peak
arrives **6–12 h too early**, earlier than the afternoon peak at high stations and roughly
opposite to the night-time peak below ~1000 m. Because TPS2 splits rain and snow by the
temperature at the hour the precipitation falls, the timing error can bias that split.
Aggregate to daily totals where sub-daily timing isn't the question.

### Output metadata

- `humidity_relative` is a **0–1 fraction**, not a percentage, and its `units` attribute
  says so (`"1"`).
- Precipitation, rainfall and snowfall are an **amount per time step** (`kg m-2`, i.e. mm per
  hour for hourly forcing), not a rate. For hourly output the two are numerically equal; for
  any other step they are not.
- Since 2026-09-30, `tps2 run` writes CF `units` and names on every forcing variable and
  records the engine version and run date (`tps2_engine`, `tps2_run_date`). Forcing written
  before then has **no attributes**: apply the conventions above by hand.
- Forcing uses the dimension name `unit`; FSM output uses `unit_id`.

---

## Spatial units

TPS2 models **terrain units**: groups of pixels with similar elevation, slope, aspect and
sky-view. It does not model individual points. A station is compared with the unit it falls in,
whose mean elevation and exposure can differ from the station's. When you validate against point data,
check that the unit represents the point, or run in `points` mode.
