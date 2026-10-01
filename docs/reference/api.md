# Python API

The public entry points for using TPS2 from Python. This page is read from the
docstrings; the full module tree is not listed on purpose, because most of it is
internal and may change between releases.

!!! note "Stability"
    Until the first tagged release, these signatures can change. The CLI and the
    configuration file are the more stable interfaces.

## Downscaling

::: topopyscale2.core.downscale.Downscaler
    options:
      members: [process_unit_simple]

## Spatial units

::: topopyscale2.spatial.units.SpatialUnit

## Forcing download

::: topopyscale2.inputs.nwp_downloader.reanalysis.loader.ERA5Loader
    options:
      members: [download]
