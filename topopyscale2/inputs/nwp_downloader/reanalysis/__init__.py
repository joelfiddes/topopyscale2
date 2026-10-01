"""Reanalysis data module.

Provides unified access to ERA5 reanalysis data from multiple backends:
- Google ARCO-ERA5 (Zarr)
- CDS API (NetCDF)
- S3 Zarr mirror
- Open-Meteo (REST API, 2022+)

Example::

    from topopyscale2.inputs.nwp_downloader.reanalysis import ERA5Loader

    loader = ERA5Loader(
        backend="google",
        bbox=(7.7, 45.95, 7.85, 46.05),
        start_date="2020-01-01",
        end_date="2020-12-31",
        pressure_levels=[300, 500, 700, 850, 1000],
        output_format="zarr",
        output_dir="./inputs/climate/",
    )
    loader.download()
"""

from topopyscale2.inputs.nwp_downloader.reanalysis.loader import ERA5Loader

__all__ = ["ERA5Loader"]
