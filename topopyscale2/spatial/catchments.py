"""Catchment-based spatial units for hydrological modeling.

Loads pre-delineated catchment polygons (e.g. HydroSHEDS) and creates
SpatialUnits at catchment level. Each catchment's forcing is the
area-weighted average of the TPS2 clusters/HRUs that overlap it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
import xarray as xr

from topopyscale2.spatial.units import SpatialUnit


class CatchmentUnitGenerator:
    """Generate catchment-level SpatialUnits from a shapefile.

    Parameters
    ----------
    shapefile : Path
        Catchment polygon shapefile (e.g. HydroSHEDS).
    id_column : str
        Column with unique catchment identifier.
    downstream_column : str
        Column linking each catchment to its downstream neighbor.
    area_column : str
        Column with catchment area in km^2.
    """

    def __init__(
        self,
        shapefile: Path,
        id_column: str = "HYBAS_ID",
        downstream_column: str = "NEXT_DOWN",
        area_column: str = "SUB_AREA",
    ):
        self.shapefile = Path(shapefile)
        self.id_column = id_column
        self.downstream_column = downstream_column
        self.area_column = area_column
        self._gdf: Optional[gpd.GeoDataFrame] = None
        self._upstream_graph: Optional[dict[str, list[str]]] = None

    def load(self) -> gpd.GeoDataFrame:
        """Load and validate the catchment shapefile."""
        gdf = gpd.read_file(self.shapefile)

        for col in [self.id_column, self.downstream_column, self.area_column]:
            if col not in gdf.columns:
                raise ValueError(
                    f"Column '{col}' not found in {self.shapefile}. "
                    f"Available: {list(gdf.columns)}"
                )

        # Ensure string IDs for consistency
        gdf[self.id_column] = gdf[self.id_column].astype(str)
        gdf[self.downstream_column] = gdf[self.downstream_column].astype(str)

        self._gdf = gdf
        return gdf

    @property
    def gdf(self) -> gpd.GeoDataFrame:
        if self._gdf is None:
            self.load()
        return self._gdf

    def generate_units(
        self,
        dem_data: Optional[xr.Dataset] = None,
    ) -> list[SpatialUnit]:
        """Create a SpatialUnit for each catchment.

        Parameters
        ----------
        dem_data : xr.Dataset, optional
            DEM data for extracting mean elevation per catchment.
            If None, elevation is set from centroid z=0.

        Returns
        -------
        list[SpatialUnit]
            One SpatialUnit per catchment polygon.
        """
        gdf = self.gdf
        units = []

        for _, row in gdf.iterrows():
            cid = str(row[self.id_column])
            geom = row.geometry
            centroid = geom.centroid

            # Mean elevation from DEM if available, else 0
            elev = 0.0
            if dem_data is not None and "elevation" in dem_data:
                elev = self._mean_elevation_in_polygon(dem_data, geom)

            area_km2 = float(row[self.area_column])

            unit = SpatialUnit(
                id=cid,
                centroid=(centroid.x, centroid.y, elev),
                attributes={
                    "area_km2": area_km2,
                    "downstream_id": str(row[self.downstream_column]),
                },
                area_m2=area_km2 * 1e6,
                geometry=geom,
            )
            units.append(unit)

        return units

    def compute_cluster_weights(
        self,
        cluster_membership: np.ndarray,
        dem_data: xr.Dataset,
    ) -> dict[str, dict[str, float]]:
        """Compute area weights mapping clusters to catchments.

        For each catchment, determines what fraction of its area is
        covered by each TPS2 cluster.

        Parameters
        ----------
        cluster_membership : ndarray, shape (ny, nx)
            Raster of cluster IDs (from TopoSUBClustering).
        dem_data : xr.Dataset
            DEM dataset with 'x' and 'y' coordinates (for geotransform).

        Returns
        -------
        dict[str, dict[str, float]]
            {catchment_id: {cluster_id: weight}} where weights sum to 1.0.
        """
        from rasterio.features import geometry_mask
        from rasterio.transform import from_bounds

        gdf = self.gdf
        ny, nx = cluster_membership.shape

        # Build affine transform from DEM coordinates
        x = dem_data.x.values
        y = dem_data.y.values
        transform = from_bounds(
            x.min(), y.min(), x.max(), y.max(), nx, ny,
        )

        # Reproject catchments to DEM CRS if needed
        if hasattr(dem_data, "attrs") and "crs" in dem_data.attrs:
            dem_crs = dem_data.attrs["crs"]
            if gdf.crs is not None and str(gdf.crs) != str(dem_crs):
                gdf = gdf.to_crs(dem_crs)

        weights = {}
        for _, row in gdf.iterrows():
            cid = str(row[self.id_column])
            geom = row.geometry

            # Create mask for this catchment
            mask = geometry_mask(
                [geom], out_shape=(ny, nx), transform=transform, invert=True,
            )

            # Count pixels per cluster within this catchment
            pixels_in_catchment = cluster_membership[mask]
            if len(pixels_in_catchment) == 0:
                weights[cid] = {}
                continue

            unique, counts = np.unique(pixels_in_catchment, return_counts=True)
            total = counts.sum()
            cluster_weights = {
                str(int(uid)): float(cnt / total)
                for uid, cnt in zip(unique, counts)
                if not np.isnan(uid) and uid >= 0
            }
            weights[cid] = cluster_weights

        return weights

    def build_upstream_graph(self) -> dict[str, list[str]]:
        """Build upstream adjacency graph from NEXT_DOWN links.

        Returns
        -------
        dict[str, list[str]]
            {catchment_id: [list of direct upstream catchment_ids]}.
        """
        if self._upstream_graph is not None:
            return self._upstream_graph

        gdf = self.gdf
        graph: dict[str, list[str]] = {
            str(row[self.id_column]): []
            for _, row in gdf.iterrows()
        }

        for _, row in gdf.iterrows():
            cid = str(row[self.id_column])
            downstream = str(row[self.downstream_column])
            if downstream in graph:
                graph[downstream].append(cid)

        self._upstream_graph = graph
        return graph

    def find_all_upstream(self, catchment_id: str) -> set[str]:
        """Find all catchments upstream of a given catchment (recursive).

        Parameters
        ----------
        catchment_id : str
            Target catchment ID.

        Returns
        -------
        set[str]
            All upstream catchment IDs (not including the target itself).
        """
        graph = self.build_upstream_graph()
        upstream = set()
        stack = list(graph.get(catchment_id, []))

        while stack:
            cid = stack.pop()
            if cid not in upstream:
                upstream.add(cid)
                stack.extend(graph.get(cid, []))

        return upstream

    @staticmethod
    def _mean_elevation_in_polygon(
        dem_data: xr.Dataset,
        geom,
    ) -> float:
        """Compute mean elevation within a polygon from DEM grid."""
        from rasterio.features import geometry_mask
        from rasterio.transform import from_bounds

        elev = dem_data["elevation"].values
        ny, nx = elev.shape
        x = dem_data.x.values
        y = dem_data.y.values
        transform = from_bounds(x.min(), y.min(), x.max(), y.max(), nx, ny)

        mask = geometry_mask([geom], out_shape=(ny, nx), transform=transform, invert=True)
        masked_elev = elev[mask]
        if len(masked_elev) == 0:
            return 0.0
        return float(np.nanmean(masked_elev))
