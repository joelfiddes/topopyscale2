"""Variable-resolution polygon (HRU) generation for TopoPyScale 2.0.

This module implements Hydrological Response Unit (HRU) generation by intersecting
elevation bands, aspect classes, and optionally catchment boundaries.
"""

from typing import Optional

import numpy as np
import xarray as xr
from scipy import ndimage
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

from topopyscale2.config.schema import PolygonConfig
from topopyscale2.spatial.units import SpatialUnit


class HRUPolygonGenerator:
    """Generate variable-resolution polygons (HRUs) from DEM data.

    HRUs are created by intersecting:
    - Elevation bands (e.g., every 200m)
    - Aspect classes (4 = N/E/S/W or 8 = cardinal + intermediate)
    - Optionally, catchment boundaries

    Each unique combination creates a polygon SpatialUnit.
    """

    def __init__(self, config: PolygonConfig):
        """Initialize the HRU polygon generator.

        Args:
            config: Polygon configuration with elevation_bands, aspect_classes,
                    min_area_m2, and respect_catchments settings.
        """
        self.config = config
        self._membership: Optional[xr.DataArray] = None

    def generate(
        self,
        dem_data: xr.Dataset,
        catchments: Optional[xr.DataArray] = None,
    ) -> list[SpatialUnit]:
        """Generate HRU polygons from DEM data.

        Args:
            dem_data: Dataset with elevation, slope, aspect, svf variables.
            catchments: Optional catchment ID raster (from DEMProcessor.delineate_catchments).
                       If provided and config.respect_catchments is True, polygons
                       will be split at catchment boundaries.

        Returns:
            List of SpatialUnit objects, one per HRU polygon.
        """
        elevation = dem_data["elevation"]
        aspect = dem_data["aspect"]

        # Get resolution for area calculations
        if "resolution_m" in elevation.attrs:
            res = float(elevation.attrs["resolution_m"])
        elif len(elevation.x) > 1:
            res = abs(float(elevation.x[1] - elevation.x[0]))
        else:
            # Single-pixel DEM - use a default resolution
            res = 30.0
        pixel_area = res * res

        # Create classification rasters
        elev_bands = self._bin_elevation(elevation, self.config.elevation_bands)
        aspect_classes = self._classify_aspect(aspect, self.config.aspect_classes)

        # Create combined label raster
        # Each unique (elevation_band, aspect_class, catchment_id) gets a unique label
        labels = self._create_combined_labels(elev_bands, aspect_classes, catchments)

        # Store membership for external use
        self._membership = xr.DataArray(
            labels,
            dims=["y", "x"],
            coords={"y": elevation.y, "x": elevation.x},
        )

        # Compute adjacency between polygons
        adjacency = self._compute_adjacency(labels)

        # Build spatial units
        units = []
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels >= 0]  # Exclude nodata (-1)

        for label in unique_labels:
            mask = labels == label
            area = float(np.sum(mask)) * pixel_area

            # Filter by minimum area
            if area < self.config.min_area_m2:
                continue

            # Compute statistics
            stats = self._polygon_stats(dem_data, mask)

            # Compute geometry
            geometry = self._mask_to_polygon(mask, elevation.x.values, elevation.y.values, res)

            # Compute centroid from pixel coordinates
            y_indices, x_indices = np.where(mask)
            cx = float(np.mean(elevation.x.values[x_indices]))
            cy = float(np.mean(elevation.y.values[y_indices]))
            cz = stats["elevation"]

            # Get neighbors
            neighbors = [f"hru_{n:04d}" for n in adjacency.get(label, [])]

            unit = SpatialUnit(
                id=f"hru_{int(label):04d}",
                centroid=(cx, cy, cz),
                attributes=stats,
                surface_type="open",  # Default, can be updated later
                area_m2=area,
                geometry=geometry,
                neighbors=neighbors if neighbors else None,
            )
            units.append(unit)

        return units

    def _bin_elevation(self, dem: xr.DataArray, band_size: float) -> np.ndarray:
        """Create elevation band raster.

        Args:
            dem: Elevation DataArray.
            band_size: Size of each elevation band in meters.

        Returns:
            Integer array where each value represents the elevation band index.
            Nodata (NaN) values are assigned -1.
        """
        elevation = dem.values.copy()
        nodata_mask = np.isnan(elevation)

        # Compute band index: floor((elevation - min_elevation) / band_size)
        valid_elev = elevation[~nodata_mask]
        if len(valid_elev) == 0:
            return np.full(elevation.shape, -1, dtype=np.int32)

        min_elev = np.floor(np.nanmin(valid_elev) / band_size) * band_size
        bands = np.floor((elevation - min_elev) / band_size).astype(np.int32)
        bands[nodata_mask] = -1

        return bands

    def _classify_aspect(self, aspect: xr.DataArray, n_classes: int) -> np.ndarray:
        """Create aspect class raster.

        Args:
            aspect: Aspect DataArray in degrees (0=N, 90=E, 180=S, 270=W).
            n_classes: Number of aspect classes (4 or 8).

        Returns:
            Integer array where each value represents the aspect class index.
            For n_classes=4: 0=N, 1=E, 2=S, 3=W
            For n_classes=8: 0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
            Nodata (NaN) values are assigned -1.
        """
        asp = aspect.values.copy()
        nodata_mask = np.isnan(asp)

        # Shift aspect so that class 0 (North) is centered at 0 degrees
        # Class width = 360 / n_classes
        class_width = 360.0 / n_classes
        half_width = class_width / 2.0

        # Shift so North (0 deg) is centered in class 0
        shifted = (asp + half_width) % 360.0
        classes = np.floor(shifted / class_width).astype(np.int32)
        classes[nodata_mask] = -1

        return classes

    def _create_combined_labels(
        self,
        elev_bands: np.ndarray,
        aspect_classes: np.ndarray,
        catchments: Optional[xr.DataArray],
    ) -> np.ndarray:
        """Create combined label raster from elevation bands, aspect classes, and catchments.

        Args:
            elev_bands: Elevation band indices.
            aspect_classes: Aspect class indices.
            catchments: Optional catchment ID raster.

        Returns:
            Integer array with unique labels for each (elev_band, aspect_class, catchment) combo.
        """
        ny, nx = elev_bands.shape

        # Handle catchments
        if catchments is not None and self.config.respect_catchments:
            catch_vals = catchments.values
        else:
            catch_vals = np.ones((ny, nx), dtype=np.int32)

        # Create unique label from combination
        # Use a hash-like approach: label = elev_band * 10000 + aspect_class * 1000 + catchment_id
        # But this could overflow, so use tuple-based unique mapping instead
        combined = np.stack([elev_bands, aspect_classes, catch_vals], axis=-1)
        combined_flat = combined.reshape(-1, 3)

        # Handle nodata
        nodata_mask = (elev_bands == -1) | (aspect_classes == -1)
        if catchments is not None and self.config.respect_catchments:
            nodata_mask |= catchments.values == 0  # Catchment nodata

        # Get unique combinations and assign labels
        unique_combos, inverse = np.unique(
            combined_flat, axis=0, return_inverse=True
        )

        labels = inverse.reshape(ny, nx).astype(np.int32)
        labels[nodata_mask] = -1

        # Relabel to consecutive integers starting from 0
        valid_labels = labels[~nodata_mask]
        if len(valid_labels) > 0:
            unique_valid = np.unique(valid_labels)
            label_map = {old: new for new, old in enumerate(unique_valid)}
            labels_new = np.full_like(labels, -1)
            for old, new in label_map.items():
                labels_new[labels == old] = new
            labels = labels_new

        return labels

    def _compute_adjacency(self, labels: np.ndarray) -> dict[int, set[int]]:
        """Find neighboring polygons from label raster.

        Two polygons are neighbors if they share at least one edge (4-connectivity).
        Uses vectorized numpy comparisons instead of nested Python loops.

        Args:
            labels: Integer label raster.

        Returns:
            Dictionary mapping each label to a set of neighboring labels.
        """
        adjacency: dict[int, set[int]] = {}

        # Horizontal neighbors: compare each pixel to its right neighbor
        h_left = labels[:, :-1]
        h_right = labels[:, 1:]
        h_mask = (h_left >= 0) & (h_right >= 0) & (h_left != h_right)
        for l1, l2 in zip(h_left[h_mask], h_right[h_mask]):
            adjacency.setdefault(int(l1), set()).add(int(l2))
            adjacency.setdefault(int(l2), set()).add(int(l1))

        # Vertical neighbors: compare each pixel to its bottom neighbor
        v_top = labels[:-1, :]
        v_bot = labels[1:, :]
        v_mask = (v_top >= 0) & (v_bot >= 0) & (v_top != v_bot)
        for l1, l2 in zip(v_top[v_mask], v_bot[v_mask]):
            adjacency.setdefault(int(l1), set()).add(int(l2))
            adjacency.setdefault(int(l2), set()).add(int(l1))

        return adjacency

    def _polygon_stats(self, dem_data: xr.Dataset, mask: np.ndarray) -> dict[str, float]:
        """Compute statistics for pixels within a polygon mask.

        Args:
            dem_data: Dataset with elevation, slope, aspect, svf variables.
            mask: Boolean mask selecting pixels in the polygon.

        Returns:
            Dictionary with mean elevation, slope, aspect (circular mean), svf.
        """
        stats = {}

        # Elevation: simple mean
        elev = dem_data["elevation"].values[mask]
        stats["elevation"] = float(np.nanmean(elev))

        # Slope: simple mean
        if "slope" in dem_data:
            slope = dem_data["slope"].values[mask]
            stats["slope"] = float(np.nanmean(slope))

        # Aspect: circular mean
        if "aspect" in dem_data:
            asp = dem_data["aspect"].values[mask]
            valid = ~np.isnan(asp)
            if np.any(valid):
                sin_a = np.mean(np.sin(np.radians(asp[valid])))
                cos_a = np.mean(np.cos(np.radians(asp[valid])))
                stats["aspect"] = float(np.degrees(np.arctan2(sin_a, cos_a)) % 360)
            else:
                stats["aspect"] = 0.0

        # SVF: simple mean
        if "svf" in dem_data:
            svf = dem_data["svf"].values[mask]
            stats["svf"] = float(np.nanmean(svf))

        return stats

    def _mask_to_polygon(
        self,
        mask: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        resolution: float,
    ) -> Optional[Polygon]:
        """Convert a boolean mask to a Shapely polygon.

        Creates polygon geometry from the outer boundary of connected pixels.

        Args:
            mask: Boolean mask of the polygon pixels.
            x_coords: X coordinate array.
            y_coords: Y coordinate array.
            resolution: Pixel resolution in meters.

        Returns:
            Shapely Polygon (or MultiPolygon if disconnected), or None if empty.
        """
        if not np.any(mask):
            return None

        # Use connected components to handle potential multi-part polygons
        labeled, n_components = ndimage.label(mask)

        polygons = []
        half_res = resolution / 2.0

        for component in range(1, n_components + 1):
            component_mask = labeled == component
            y_idx, x_idx = np.where(component_mask)

            if len(y_idx) == 0:
                continue

            # Create pixel rectangles and union them
            pixel_polys = []
            for yi, xi in zip(y_idx, x_idx):
                x_center = x_coords[xi]
                y_center = y_coords[yi]
                # Create pixel polygon
                pixel_poly = Polygon([
                    (x_center - half_res, y_center - half_res),
                    (x_center + half_res, y_center - half_res),
                    (x_center + half_res, y_center + half_res),
                    (x_center - half_res, y_center + half_res),
                ])
                pixel_polys.append(pixel_poly)

            if pixel_polys:
                merged = unary_union(pixel_polys)
                if merged.is_valid and not merged.is_empty:
                    polygons.append(merged)

        if not polygons:
            return None
        elif len(polygons) == 1:
            return polygons[0] if isinstance(polygons[0], Polygon) else None
        else:
            multi = unary_union(polygons)
            if isinstance(multi, (Polygon, MultiPolygon)):
                return multi
            return None

    def get_membership(self) -> Optional[xr.DataArray]:
        """Return polygon membership raster (pixel -> polygon label).

        Returns:
            DataArray with integer labels, or None if generate() hasn't been called.
        """
        return self._membership

    def get_aspect_class_name(self, class_idx: int) -> str:
        """Get human-readable name for an aspect class.

        Args:
            class_idx: Aspect class index.

        Returns:
            Name string (e.g., "N", "NE", "E", etc.)
        """
        if self.config.aspect_classes == 4:
            names = ["N", "E", "S", "W"]
        else:  # 8 classes
            names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

        if 0 <= class_idx < len(names):
            return names[class_idx]
        return f"class_{class_idx}"
