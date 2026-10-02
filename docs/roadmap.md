# Roadmap

Public releases of TopoPyScale 2 are static snapshots of the **downscaling engine**:
terrain units, ERA5 downscaling, the forcing output, and the tools to configure, run and look
at it. Development continues in a private repository, where the pieces below already exist in
some form and are used on real domains. They will be released as they become robust enough
to support without the people who wrote them in the room.

There are no dates here on purpose. To ask for something, or to say which of these matters
most to you, open an issue (see [Contributing](contributing.md)).

## See them in operation

Several of the features below already run every day in operational systems built on the
same engine:

- **[Snow forecasting for Central Asia](https://apps.mountainfutures.ch/ca-forecast/)**:
  the daily cycle from start to finish. ERA5 and ECMWF forecasts are downscaled onto the
  mountains of Central Asia, the FSM snow model runs on every terrain unit, and the results are
  published as maps and forecasts of snow depth and snow water equivalent, compared against
  the long-term climatology. It shows the snow model, forecasts and dashboards from this
  roadmap working together.

## Models driven by the forcing

- **Snow model (FSM).** The Factorial Snow Model run on every terrain unit: snow depth, SWE,
  melt and runoff, with a Rust core. Known issues are measured and being fixed first: snow-free
  ground too cold, no overburden compaction, late melt-out at mid elevations.
- **Glacier-enabled snow model.** Ice melt under the snowpack, so units above the
  equilibrium line lose mass instead of accumulating snow for ever.
- **More snow models.** SNOWPACK and others fed from the same forcing export.
- **Hydrology.** HBV and GR4J catchment models on the downscaled forcing.

## Observations and validation

- **Station data.** Fetching and quality-controlling station observations (global archives and
  national networks), with explicit timestamp and unit conventions.
- **A validation lab.** Scoring forcing and snow against stations, snow courses, satellite
  snow cover and glacier mass balance, with held-out data and reproducible scorecards.

## Better forcing

- **Data assimilation.** Ensemble methods that use satellite snow cover to correct the
  precipitation and temperature forcing.
- **Forecasts.** Downscaling ECMWF forecasts (deterministic and ensemble) and blending them
  onto the reanalysis.
- **Learned corrections.** Machine-learning precipitation corrections trained on gauges.
- **A differentiable backend.** JAX versions of the kernels, for calibration by gradient.

## Climate scenarios

- **TopoCLIM.** Per-unit quantile mapping of CMIP6 projections against the downscaled
  reanalysis, for future snow and ground temperature.

## Products

- **Dashboards and warnings.** Map-based snow and weather products built from the model
  output, as used in operational snow forecasting.
