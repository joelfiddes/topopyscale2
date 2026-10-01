"""Spatial unit data model for TopoPyScale 2.0."""

from dataclasses import dataclass, field
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon


@dataclass
class SpatialUnit:
    """A spatial unit (cluster centroid, polygon, or grid cell) for downscaling."""

    id: str
    centroid: tuple[float, float, float]  # (x, y, z)
    attributes: dict[str, float] = field(default_factory=dict)
    surface_type: str = "open"
    area_m2: float = 0.0
    geometry: Optional[Polygon] = None
    neighbors: Optional[list[str]] = None
    era5_grid_indices: Optional[tuple[int, ...]] = None

    @property
    def x(self) -> float:
        return self.centroid[0]

    @property
    def y(self) -> float:
        return self.centroid[1]

    @property
    def elevation(self) -> float:
        return self.centroid[2]


class SpatialUnitCollection:
    """Container for spatial units with convenience methods."""

    def __init__(self, units: list[SpatialUnit]):
        self._units = {u.id: u for u in units}

    def __len__(self) -> int:
        return len(self._units)

    def __iter__(self):
        return iter(self._units.values())

    def __getitem__(self, unit_id: str) -> SpatialUnit:
        return self._units[unit_id]

    @property
    def ids(self) -> list[str]:
        return list(self._units.keys())

    @property
    def units(self) -> list[SpatialUnit]:
        return list(self._units.values())

    def get_by_id(self, unit_id: str) -> SpatialUnit:
        """Get a spatial unit by its ID."""
        if unit_id not in self._units:
            raise KeyError(f"No spatial unit with id '{unit_id}'")
        return self._units[unit_id]

    def filter_by_surface_type(self, surface_type: str) -> "SpatialUnitCollection":
        """Return a new collection filtered by surface type."""
        filtered = [u for u in self._units.values() if u.surface_type == surface_type]
        return SpatialUnitCollection(filtered)

    def elevations(self) -> np.ndarray:
        """Return array of elevations for all units."""
        return np.array([u.elevation for u in self._units.values()])

    def to_geodataframe(self) -> gpd.GeoDataFrame:
        """Convert to a GeoDataFrame."""
        records = []
        for u in self._units.values():
            record = {
                "id": u.id,
                "x": u.x,
                "y": u.y,
                "elevation": u.elevation,
                "surface_type": u.surface_type,
                "area_m2": u.area_m2,
                "geometry": Point(u.x, u.y) if u.geometry is None else u.geometry,
            }
            record.update(u.attributes)
            records.append(record)

        gdf = gpd.GeoDataFrame(records)
        return gdf
