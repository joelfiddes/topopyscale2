# TopoPyScale 2.0 — bulletproof, reproducible install.
#
# Bundles the geospatial stack (GDAL/rasterio/geopandas) and the Rust toolchain
# via conda-forge, so the image builds the same way on any host regardless of
# local OS / GDAL / Rust state. Use this if a native `pip install -e .` hits
# dependency friction.
#
# Build:
#   docker build -t tps2 .
#
# The shipped demo needs no container: open examples/forcing_demo/demo.html in a browser.
#
# Run a full pipeline (needs network for DEM + ERA5):
#   docker run --rm -v "$PWD/work:/work/run" tps2 \
#     bash -lc "tps2 init run/alps && tps2 run -c run/alps/config.yaml"
#
# Interactive shell:
#   docker run --rm -it tps2 bash

FROM condaforge/miniforge3:latest

WORKDIR /work

# Create the conda env from the same spec used for local installs.
COPY environment.yml /work/environment.yml

# Copy the source needed to build & install the package (env file references `-e .`).
COPY pyproject.toml Cargo.toml Cargo.lock README.md /work/
COPY src /work/src
COPY topopyscale2 /work/topopyscale2

# Build the env (resolves GDAL + Rust + maturin from conda-forge) and install TPS2.
# `conda env create` runs the `pip: -e .` line, which compiles the Rust kernels.
RUN conda env create -f environment.yml && conda clean -afy

# Make the `tps2` env the default for every subsequent command / `docker run`.
SHELL ["conda", "run", "--no-capture-output", "-n", "tps2", "/bin/bash", "-c"]
ENV PATH=/opt/conda/envs/tps2/bin:$PATH
RUN echo "conda activate tps2" >> ~/.bashrc

# Sanity check the install at build time.
RUN tps2 --version

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "tps2"]
CMD ["bash"]
