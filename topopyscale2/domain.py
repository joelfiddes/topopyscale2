"""Domain: high-level orchestrator tying spatial + input + downscaling together."""

import hashlib
import json
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import xarray as xr

from topopyscale2._optional import not_in_release
from topopyscale2.config.schema import TPS2Config
from topopyscale2.spatial.clusters import TopoSUBClustering
from topopyscale2.spatial.dem import DEMProcessor, resolve_dem_crs
from topopyscale2.spatial.polygons import HRUPolygonGenerator
from topopyscale2.spatial.surface_types import SurfaceTypeClassifier
from topopyscale2.spatial.units import SpatialUnit, SpatialUnitCollection

log = logging.getLogger(__name__)


def _lonlat_to_crs_transform(target_crs):
    """Build a ``(lon, lat) -> (x, y)`` converter into ``target_crs``.

    Returns None when no conversion is needed -- an unset CRS, or a geographic
    one where the DEM axes already are lon/lat.
    """
    if target_crs is None:
        return None
    from pyproj import CRS, Transformer

    crs_obj = CRS.from_user_input(target_crs)
    if crs_obj.is_geographic:
        return None
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), crs_obj, always_xy=True
    )
    return lambda lon, lat: transformer.transform(lon, lat)


class Domain:
    """Top-level domain object: setup, fetch, downscale."""

    def __init__(self, config: TPS2Config):
        self.config = config
        self.dem_data: Optional[xr.Dataset] = None
        self.units: Optional[SpatialUnitCollection] = None
        self._nwp_source = None
        self._ds_surface: Optional[xr.Dataset] = None
        self._ds_pressure: Optional[xr.Dataset] = None
        self._ds_forecast_surface: Optional[xr.Dataset] = None
        self._ds_forecast_pressure: Optional[xr.Dataset] = None
        self._membership: Optional[xr.DataArray] = None
        self._clustering: Optional[TopoSUBClustering] = None
        # Per-group ERA5 bboxes for sparse points mode
        self._sparse_era5_bboxes: Optional[list[tuple[float, float, float, float]]] = None

    @classmethod
    def from_config(cls, path: str | Path) -> "Domain":
        """Create Domain from a YAML config file."""
        config = TPS2Config.from_yaml(path)
        return cls(config)

    def load_cached_units(self) -> bool:
        """Load spatial units from the domain-state cache, writing nothing.

        Returns True if a matching cached state was found. Unlike :meth:`setup`
        this never acquires a DEM, never re-clusters and never writes — it is
        the entry point for read-only consumers (the Validation Lab) that must
        score against *the units the run actually used*.

        Re-deriving units instead would be worse than a missing cache: k-means
        on a freshly fetched DEM produces different `unit_id`s, so the caller
        would silently compare model output against units that do not
        correspond to it.
        """
        return self._try_load_cached_domain()

    def _domain_config_hash(self) -> str:
        """Hash domain-relevant config fields to detect changes."""
        # Include point coordinates in hash for sparse points cache invalidation
        points_data = None
        if self.config.domain.points and self.config.domain.points.coordinates:
            points_data = [
                {"name": p.name, "lon": p.lon, "lat": p.lat, "elevation": p.elevation}
                for p in self.config.domain.points.coordinates
            ]
        key = json.dumps({
            "bbox": self.config.domain.bbox,
            "dem_source": self.config.domain.dem_source,
            "grid_resolution": self.config.domain.grid_resolution,
            "crs": self.config.domain.crs,
            "spatial_mode": self.config.domain.spatial_mode,
            "svf_mode": getattr(self.config.domain, "svf_mode", "centroid"),
            "n_clusters": self.config.clustering.n_clusters,
            "features": self.config.clustering.features,
            "feature_weights": self.config.clustering.feature_weights,
            "horizon_n_directions": self.config.domain.horizon_n_directions,
            "horizon_max_distance_m": self.config.domain.horizon_max_distance_m,
            "points": points_data,
            "dem_buffer_m": getattr(self.config.domain, "dem_buffer_m", None),
        }, sort_keys=True)
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _try_load_cached_domain(self) -> bool:
        """Try to load domain state from cache. Returns True if successful."""
        dem_cache = self.config.storage.dem_cache
        if dem_cache is None:
            return False
        cache_path = dem_cache / "domain_state.pkl"
        if not cache_path.exists():
            return False
        try:
            with open(cache_path, "rb") as f:
                state = pickle.load(f)
            if state.get("config_hash") != self._domain_config_hash():
                log.info("Domain config changed, re-running setup")
                return False
            self.dem_data = state["dem_data"]
            self.units = state["units"]
            self._membership = state.get("membership")
            self._sparse_era5_bboxes = state.get("sparse_era5_bboxes")
            log.info("Loaded cached domain state (%d units)", len(self.units))
            return True
        except Exception as e:
            log.warning("Failed to load domain cache: %s", e)
            return False

    def _save_domain_state(self) -> None:
        """Save domain state to cache for fast reload.

        Two things this deliberately does not leave to chance:

        * The cache directory is created here. In the usual path it already
          exists because the DEM *download* made it, but a domain configured
          with a local ``domain.dem`` never downloads, so the write used to
          raise FileNotFoundError and the cache was never written at all.
        * The pickle is written to a temporary file in the same directory and
          then renamed. A run interrupted mid-dump can therefore never leave a
          half-written ``domain_state.pkl`` behind for the next run to load.

        A cache failure stays non-fatal — it must not kill a run — but it is
        logged at ERROR, not warning: the cost is silent and large (units get
        re-derived on every subsequent run, which for a large domain means a
        full k-means plus horizon angles, and the Validation Lab refuses to
        score a run whose units were never cached).
        """
        dem_cache = self.config.storage.dem_cache
        if dem_cache is None:
            return
        dem_cache = Path(dem_cache)
        cache_path = dem_cache / "domain_state.pkl"
        tmp_path: Optional[Path] = None
        try:
            dem_cache.mkdir(parents=True, exist_ok=True)
            state = {
                "config_hash": self._domain_config_hash(),
                "dem_data": self.dem_data,
                "units": self.units,
                "membership": self._membership,
                "sparse_era5_bboxes": self._sparse_era5_bboxes,
            }
            fd, tmp_name = tempfile.mkstemp(
                dir=str(dem_cache), prefix=".domain_state-", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            with os.fdopen(fd, "wb") as f:
                pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, cache_path)
            tmp_path = None
            log.info("Saved domain state to %s", cache_path)
        except Exception as e:
            log.error(
                "Failed to save domain cache %s: %s — units will be re-derived "
                "next run (re-clustering a large domain costs many minutes) and "
                "the Validation Lab cannot score this run without it",
                cache_path,
                e,
                exc_info=True,
            )
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:  # pragma: no cover - best-effort cleanup
                    pass

    def setup(self, progress_callback=None) -> None:
        """Run domain setup: DEM → terrain → surface types → spatial units.

        Parameters
        ----------
        progress_callback : callable, optional
            Called as progress_callback(step, total_steps, description) at each
            sub-step boundary for progress reporting.

        Depending on spatial_mode in config:
        - "clusters": Use TopoSUB k-means clustering (default)
        - "polygons": Use HRU polygon generation (elevation bands + aspect classes)
        - "grid": Use regular grid (not yet implemented)

        SVF computation is controlled by svf_mode in domain config:
        - "full": Compute horizon for all pixels (slow for large domains)
        - "centroid": Compute horizon only at cluster centroids (recommended)
        - "approximate": Use SVF ≈ cos(slope) approximation (fastest)
        """
        # Try loading cached domain state first
        if self._try_load_cached_domain():
            if progress_callback is not None:
                progress_callback(4, 4, "Loaded from cache")
            return

        spatial_mode = self.config.domain.spatial_mode
        svf_mode = getattr(self.config.domain, "svf_mode", "centroid")

        # Sparse points mode: no bbox, auto-derive per-point DEM patches
        if (
            spatial_mode == "points"
            and self.config.domain.bbox is None
            and self.config.domain.dem is None
        ):
            self._setup_sparse_points(progress_callback)
            return

        # 1. DEM processing
        # Pass wind_config and polygon_config to enable optional computations
        wind_config = self.config.downscaling.wind_config
        polygon_config = self.config.polygons if spatial_mode == "polygons" else None

        dem_proc = DEMProcessor(
            self.config.domain,
            wind_config=wind_config,
            polygon_config=polygon_config,
        )
        self._dem_processor = dem_proc  # Keep reference for later SVF computation

        # Skip horizon if using centroid or approximate mode
        skip_horizon = svf_mode in ("centroid", "approximate")
        self.dem_data = dem_proc.process(skip_horizon=skip_horizon)
        if progress_callback is not None:
            progress_callback(1, 4, "DEM processing")

        # 2. Surface types
        classifier = SurfaceTypeClassifier()
        raster_path = getattr(self.config.surface_types, "raster_path", None)
        # Resolve relative paths against config file directory
        if raster_path is not None and not raster_path.is_absolute():
            config_dir = getattr(self.config, "_config_dir", None)
            if config_dir:
                raster_path = config_dir / raster_path
        st_cfg = self.config.surface_types
        surface_types = classifier.classify(
            self.dem_data,
            raster_path=raster_path,
            glacier_source=getattr(st_cfg, "glacier_source", "none"),
            rgi_regions=getattr(st_cfg, "rgi_regions", None),
            rgi_cache=getattr(st_cfg, "rgi_cache", None),
        )
        if progress_callback is not None:
            progress_callback(2, 4, "Surface types")

        # 3. Spatial unit generation (depends on spatial_mode)
        if spatial_mode == "points":
            # Explicit point locations — create units directly from coordinates
            units = self._create_point_units()

        elif spatial_mode == "polygons":
            # HRU polygon generation
            generator = HRUPolygonGenerator(self.config.polygons)

            # Get catchments from DEM data if available and respect_catchments is True
            catchments = None
            if self.config.polygons.respect_catchments and "catchments" in self.dem_data:
                catchments = self.dem_data["catchments"]

            units = generator.generate(self.dem_data, catchments=catchments)
            self._membership = generator.get_membership()

        elif spatial_mode == "grid":
            # Regular grid mode - not yet implemented
            raise NotImplementedError(
                "Grid spatial mode is not yet implemented. "
                "Use 'clusters' or 'polygons' instead."
            )

        else:
            # Default: TopoSUB clustering
            self._clustering = TopoSUBClustering(
                self.config.clustering,
                self.config.surface_types,
            )
            domain_mask = self._load_domain_mask()
            units = self._clustering.fit(
                self.dem_data, surface_types, domain_mask=domain_mask
            )
            self._membership = self._clustering.get_membership()

        self.units = SpatialUnitCollection(units)
        if progress_callback is not None:
            progress_callback(3, 4, "Clustering")

        # 4. Compute SVF at cluster centroids if using centroid mode
        if svf_mode == "centroid":
            self._compute_centroid_svf()
        elif svf_mode == "approximate":
            self._compute_approximate_svf()

        # Save cluster/membership map for visualization
        self._save_membership_map()
        # Cache domain state for fast reload on re-run
        self._save_domain_state()
        if progress_callback is not None:
            progress_callback(4, 4, "SVF computation")

    def _load_domain_mask(self) -> Optional[np.ndarray]:
        """Rasterize an optional domain-mask shapefile onto the DEM grid.

        Returns a boolean array (ny, nx) where True marks pixels inside the
        polygon union (the modelling domain), or None if no mask_shapefile is
        configured. Mirrors the transform/CRS handling in
        ``CatchmentUnitGenerator.compute_area_weights`` so the mask aligns with
        the cluster-membership raster and the catchment aggregation.
        """
        mask_path = getattr(self.config.domain, "mask_shapefile", None)
        if mask_path is None:
            return None

        # Resolve relative paths against the config file directory
        mask_path = Path(mask_path)
        if not mask_path.is_absolute():
            config_dir = getattr(self.config, "_config_dir", None)
            if config_dir:
                mask_path = config_dir / mask_path
        if not mask_path.exists():
            raise FileNotFoundError(f"domain.mask_shapefile not found: {mask_path}")

        import geopandas as gpd
        from rasterio.features import geometry_mask
        from rasterio.transform import from_bounds

        gdf = gpd.read_file(mask_path)

        # Reproject to the DEM CRS if known
        dem_crs = self.dem_data.attrs.get("crs") if self.dem_data is not None else None
        if dem_crs is not None and gdf.crs is not None and str(gdf.crs) != str(dem_crs):
            gdf = gdf.to_crs(dem_crs)

        # Optional buffer (in DEM/projected units, typically metres)
        buffer_m = getattr(self.config.domain, "mask_buffer_m", None)
        if buffer_m:
            gdf = gdf.copy()
            gdf["geometry"] = gdf.geometry.buffer(buffer_m)

        x = self.dem_data.x.values
        y = self.dem_data.y.values
        ny, nx = self.dem_data["elevation"].shape
        transform = from_bounds(x.min(), y.min(), x.max(), y.max(), nx, ny)

        geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
        if not geoms:
            raise ValueError(f"domain.mask_shapefile has no valid geometries: {mask_path}")

        # invert=True → True inside the polygons (the domain we keep)
        inside = geometry_mask(geoms, out_shape=(ny, nx), transform=transform, invert=True)
        n_in = int(inside.sum())
        log.info(
            "Domain mask %s: %d/%d pixels inside (%.1f%%)",
            mask_path.name, n_in, inside.size, 100.0 * n_in / inside.size,
        )
        if n_in == 0:
            raise ValueError(
                f"domain.mask_shapefile {mask_path.name} masks out the entire DEM "
                "(0 pixels inside) — check CRS/extent alignment"
            )
        return inside

    def _create_point_units(self) -> list[SpatialUnit]:
        """Create SpatialUnits from explicit point coordinates.

        Looks up elevation, slope, aspect from DEM at each point location.

        Points are configured as lon/lat, but the DEM grid and every
        SpatialUnit centroid live in the domain CRS -- that is the invariant
        cluster mode establishes, and the downscaler depends on it: it
        reprojects centroids *from* ``dem_data.attrs['crs']`` when computing
        ERA5 bilinear weights. So lon/lat must be converted here. Without the
        conversion a projected domain CRS reads degrees as metres, which both
        samples terrain at the wrong place and collapses every point onto the
        projection origin, giving all units the same ERA5 column.
        """
        points_cfg = self.config.domain.points
        if points_cfg is None or not points_cfg.coordinates:
            raise ValueError(
                "spatial_mode='points' requires domain.points.coordinates to be set"
            )

        elev = self.dem_data["elevation"]
        slope_arr = self.dem_data["slope"] if "slope" in self.dem_data else None
        aspect_arr = self.dem_data["aspect"] if "aspect" in self.dem_data else None

        to_dem = _lonlat_to_crs_transform(self.dem_data.attrs.get("crs"))

        units = []
        for pt in points_cfg.coordinates:
            px, py = to_dem(pt.lon, pt.lat) if to_dem is not None else (pt.lon, pt.lat)

            # Look up elevation from DEM if not provided
            if pt.elevation is not None:
                z = pt.elevation
            else:
                z = float(elev.sel(x=px, y=py, method="nearest").values)

            attrs = {"elevation": z, "slope": 0.0, "aspect": 0.0, "svf": 1.0}
            if slope_arr is not None:
                attrs["slope"] = float(slope_arr.sel(x=px, y=py, method="nearest").values)
            if aspect_arr is not None:
                attrs["aspect"] = float(aspect_arr.sel(x=px, y=py, method="nearest").values)

            unit = SpatialUnit(
                id=pt.name,
                centroid=(px, py, z),
                attributes=attrs,
                surface_type="open",
                area_m2=1.0,  # Point has no area
            )
            units.append(unit)

        return units

    # ------------------------------------------------------------------
    # Sparse points mode: per-point/per-group DEM patches
    # ------------------------------------------------------------------

    def _setup_sparse_points(self, progress_callback=None) -> None:
        """Setup for sparse points: per-group DEM patches, merged units.

        When spatial_mode='points' and no bbox is given, each point (or group
        of nearby points) gets its own small DEM patch.  All terrain attributes
        are extracted from the local patch and baked into SpatialUnits.  A
        minimal ``self.dem_data`` Dataset is created with just CRS metadata so
        the downscaler can proceed.
        """
        from topopyscale2.inputs.dem_download import download_dem

        points_cfg = self.config.domain.points
        if points_cfg is None or not points_cfg.coordinates:
            raise ValueError(
                "spatial_mode='points' requires domain.points.coordinates"
            )

        coords = points_cfg.coordinates
        max_dist = self.config.domain.horizon_max_distance_m
        n_dirs = self.config.domain.horizon_n_directions
        buffer_m = self.config.domain.dem_buffer_m
        if buffer_m is None:
            buffer_m = max_dist + 1000.0

        # 1. Group nearby points
        groups = self._group_points(coords, threshold_m=2 * max_dist)
        log.info("Sparse points: %d points → %d groups", len(coords), len(groups))
        if progress_callback is not None:
            progress_callback(1, 4, "Grouped points")

        # 2. Per-group: download DEM patch → extract terrain → create units
        dem_cache = Path(self.config.storage.dem_cache)
        dem_cache.mkdir(parents=True, exist_ok=True)
        dem_source = self.config.domain.dem_source

        all_units: list[SpatialUnit] = []
        era5_bboxes: list[tuple[float, float, float, float]] = []

        for gi, group in enumerate(groups):
            patch_bbox = self._compute_patch_bbox(group, buffer_m)
            patch_path = dem_cache / f"dem_patch_{gi:03d}.tif"

            # Download DEM patch (uses cache if file exists)
            if not patch_path.exists():
                download_dem(
                    bbox=list(patch_bbox),
                    output_path=str(patch_path),
                    source=dem_source,
                )
                log.info("Downloaded DEM patch %d: %s", gi, patch_bbox)
            else:
                log.info("Using cached DEM patch %d: %s", gi, patch_path)

            # Load and process terrain
            dem_proc = DEMProcessor(self.config.domain)
            dem_da = dem_proc.load(patch_path)
            slope_da, aspect_da = dem_proc.compute_slope_aspect(dem_da)

            # Compute SVF at each point in this group
            pt_coords = [(p.lon, p.lat) for p in group]
            svf_values = dem_proc.compute_svf_at_points(
                dem_da, pt_coords, n_directions=n_dirs, max_distance_m=max_dist
            )

            # Create SpatialUnit per point
            for pi, pt in enumerate(group):
                if pt.elevation is not None:
                    z = pt.elevation
                else:
                    z = float(
                        dem_da.sel(x=pt.lon, y=pt.lat, method="nearest").values
                    )

                s = float(
                    slope_da.sel(x=pt.lon, y=pt.lat, method="nearest").values
                )
                a = float(
                    aspect_da.sel(x=pt.lon, y=pt.lat, method="nearest").values
                )

                unit = SpatialUnit(
                    id=pt.name,
                    centroid=(pt.lon, pt.lat, z),
                    attributes={
                        "elevation": z,
                        "slope": s,
                        "aspect": a,
                        "svf": float(svf_values[pi]),
                    },
                    surface_type="open",
                    area_m2=1.0,
                )
                all_units.append(unit)

            # ERA5 bbox for this group: point extent + 1° padding
            lons = [p.lon for p in group]
            lats = [p.lat for p in group]
            era5_bboxes.append((
                min(lons) - 1.0,
                min(lats) - 1.0,
                max(lons) + 1.0,
                max(lats) + 1.0,
            ))

        if progress_callback is not None:
            progress_callback(2, 4, "DEM patches processed")

        # 3. Assemble results
        self.units = SpatialUnitCollection(all_units)
        self._sparse_era5_bboxes = era5_bboxes

        # Minimal dem_data — just CRS for the downscaler
        self.dem_data = xr.Dataset(attrs={"crs": "EPSG:4326"})

        if progress_callback is not None:
            progress_callback(3, 4, "Units assembled")

        # 4. Cache
        self._save_domain_state()
        if progress_callback is not None:
            progress_callback(4, 4, "Sparse points setup complete")

        log.info(
            "Sparse points setup complete: %d units, %d ERA5 patches",
            len(all_units),
            len(era5_bboxes),
        )

    @staticmethod
    def _group_points(
        coordinates: list,
        threshold_m: float = 6000.0,
    ) -> list[list]:
        """Group points by proximity using single-linkage clustering.

        Parameters
        ----------
        coordinates : list[PointCoordinate]
            Point coordinates to group.
        threshold_m : float
            Distance threshold in meters. Points closer than this are grouped.

        Returns
        -------
        list[list[PointCoordinate]]
            Groups of point coordinates.
        """
        if len(coordinates) <= 1:
            return [list(coordinates)]

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        # Approximate geographic distance in meters
        lons = np.array([p.lon for p in coordinates])
        lats = np.array([p.lat for p in coordinates])

        # Convert to approximate meters using mean latitude
        mean_lat = np.mean(lats)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))

        pts_m = np.column_stack([
            lons * m_per_deg_lon,
            lats * m_per_deg_lat,
        ])

        dists = pdist(pts_m)
        Z = linkage(dists, method="single")
        labels = fcluster(Z, t=threshold_m, criterion="distance")

        groups: dict[int, list] = {}
        for coord, label in zip(coordinates, labels):
            groups.setdefault(label, []).append(coord)

        return list(groups.values())

    @staticmethod
    def _compute_patch_bbox(
        group: list,
        buffer_m: float,
    ) -> tuple[float, float, float, float]:
        """Compute DEM bbox for a group of points with buffer.

        Parameters
        ----------
        group : list[PointCoordinate]
            Points in the group.
        buffer_m : float
            Buffer radius in meters around the point extent.

        Returns
        -------
        tuple
            (west, south, east, north) in degrees.
        """
        lons = [p.lon for p in group]
        lats = [p.lat for p in group]

        mean_lat = np.mean(lats)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(mean_lat))

        buf_lat = buffer_m / m_per_deg_lat
        buf_lon = buffer_m / m_per_deg_lon

        return (
            min(lons) - buf_lon,
            min(lats) - buf_lat,
            max(lons) + buf_lon,
            max(lats) + buf_lat,
        )

    def _compute_centroid_svf(self) -> None:
        """Compute SVF at cluster centroids and assign to units.

        Much faster than full-raster computation for large domains.
        For 2000 clusters, computes ~2000 points instead of millions.
        """
        if self.units is None or self._dem_processor is None:
            return

        # Get centroid coordinates from units
        points = []
        for unit in self.units:
            lon, lat, _ = unit.centroid
            points.append((lon, lat))

        if not points:
            return

        # Compute SVF at centroids
        n_dirs = getattr(self.config.domain, "horizon_n_directions", 8)
        max_dist = getattr(self.config.domain, "horizon_max_distance_m", 3000.0)

        dem = self.dem_data["elevation"]
        svf_values = self._dem_processor.compute_svf_at_points(
            dem, points, n_directions=n_dirs, max_distance_m=max_dist
        )

        # Assign SVF to each unit
        for i, unit in enumerate(self.units):
            unit.attributes["svf"] = float(svf_values[i])

        print(f"  SVF computed for {len(points)} centroids (range: {svf_values.min():.2f} - {svf_values.max():.2f})")

    def _compute_approximate_svf(self) -> None:
        """Compute approximate SVF from slope.

        Uses the simple approximation: SVF ≈ (1 + cos(slope)) / 2
        Fast but less accurate for complex terrain with obstructions.
        """
        if self.units is None or self.dem_data is None:
            return

        print("  Computing approximate SVF from slope...")

        for unit in self.units:
            slope_rad = np.radians(unit.attributes.get("slope", 0.0))
            # Simple approximation: flat terrain = 1.0, vertical = 0.5
            svf = (1 + np.cos(slope_rad)) / 2
            unit.attributes["svf"] = float(svf)

        print("  Approximate SVF assigned to all units.")

    def _save_membership_map(self) -> None:
        """Save the membership (cluster assignment) map as a GeoTiff.

        Georeferencing is read from ``dem_cache/dem.tif`` when that file is
        present (the downloaded-DEM case — unchanged). A domain configured with
        a local ``domain.dem`` never puts a ``dem.tif`` in the cache, and this
        used to return early for those, so ``cluster_map.tif`` was silently
        never written. In that case the transform and CRS are derived from
        ``self.dem_data`` instead.
        """
        if self._membership is None or self.dem_data is None:
            return

        dem_cache = self.config.storage.dem_cache
        if dem_cache is None:
            return
        dem_cache = Path(dem_cache)

        dem_path = dem_cache / "dem.tif"
        cluster_map_path = dem_cache / "cluster_map.tif"

        try:
            data = np.asarray(self._membership).astype(np.int16)

            if dem_path.exists():
                # Read DEM metadata for georeferencing
                with rasterio.open(dem_path) as src:
                    profile = src.profile.copy()
            else:
                profile = self._membership_profile_from_dem_data(data.shape)

            # Update profile for int16 cluster IDs
            profile.update(dtype=rasterio.int16, nodata=-1, count=1)

            # Write cluster map
            dem_cache.mkdir(parents=True, exist_ok=True)
            with rasterio.open(cluster_map_path, "w", **profile) as dst:
                dst.write(data, 1)
            log.info("Saved cluster map to %s", cluster_map_path)
        except Exception as e:
            log.error(
                "Failed to write cluster map %s: %s — units cannot be mapped "
                "back to DEM pixels (raster plots and pixel-level validation "
                "will be unavailable for this run)",
                cluster_map_path,
                e,
                exc_info=True,
            )

    def _membership_profile_from_dem_data(self, shape: tuple[int, int]) -> dict:
        """Build a rasterio profile for the cluster map from ``self.dem_data``.

        Used when no ``dem.tif`` is cached (local-DEM domains). The DEM x/y
        coordinates are pixel *centres* (see ``DEMProcessor.load``), so the
        affine origin is shifted back by half a pixel.
        """
        from rasterio.crs import CRS as RioCRS
        from rasterio.transform import Affine

        ny, nx = shape
        x = np.asarray(self.dem_data["x"].values, dtype=float)
        y = np.asarray(self.dem_data["y"].values, dtype=float)
        if x.size != nx or y.size != ny:
            raise ValueError(
                f"membership shape {shape} does not match the DEM grid "
                f"({y.size}, {x.size})"
            )

        fallback_res = self._dem_resolution_fallback()
        res_x = float(x[1] - x[0]) if x.size > 1 else fallback_res
        res_y = float(y[1] - y[0]) if y.size > 1 else -fallback_res
        transform = Affine.translation(
            float(x[0]) - res_x / 2.0, float(y[0]) - res_y / 2.0
        ) * Affine.scale(res_x, res_y)

        crs_str = resolve_dem_crs(self.dem_data, override=self.config.domain.crs)
        if crs_str is None:
            log.warning(
                "No CRS recorded on the DEM — cluster_map.tif will be written "
                "without georeferencing metadata"
            )
            crs = None
        else:
            crs = RioCRS.from_user_input(crs_str)

        return {
            "driver": "GTiff",
            "height": ny,
            "width": nx,
            "count": 1,
            "dtype": rasterio.int16,
            "crs": crs,
            "transform": transform,
        }

    def _dem_resolution_fallback(self) -> float:
        """Pixel size to use when an axis is too short to infer one from."""
        res = self.dem_data.attrs.get("resolution_m")
        if res is None and "elevation" in getattr(self.dem_data, "data_vars", {}):
            res = self.dem_data["elevation"].attrs.get("resolution_m")
        try:
            return float(res) if res is not None else 1.0
        except (TypeError, ValueError):
            return 1.0

    def fetch_forcing(self, time_range: Optional[list[str]] = None) -> None:
        """Fetch NWP forcing data (ERA5 or IFS).

        Respects ``output.variables`` from config — when set, only the ERA5
        variables needed to produce the requested outputs are downloaded,
        potentially saving significant bandwidth and time.

        In sparse points mode, fetches per-group ERA5 patches and merges them
        to avoid downloading data for large empty regions between distant points.
        """
        primary = self.config.inputs.primary

        if primary == "ifs":
            try:
                from topopyscale2.inputs.hres import IFSSource
            except ModuleNotFoundError as e:
                raise not_in_release("inputs.primary 'ifs'", e, "topopyscale2.inputs.hres") from e
            self._nwp_source = IFSSource(self.config.inputs, self.config.storage)
        else:
            from topopyscale2.inputs.era5 import ERA5Source
            self._nwp_source = ERA5Source(self.config.inputs, self.config.storage)

        tr = time_range or self.config.inputs.time_range

        # Determine minimal ERA5 variable sets if output.variables is specified
        required_vars = None
        if self.config.output.variables:
            from topopyscale2.core.downscale import required_era5_variables
            required_vars = required_era5_variables(
                self.config.output.variables,
                self.config.downscaling.mode,
            )

        # Sparse points mode: fetch per-group ERA5 patches and merge
        if self._sparse_era5_bboxes and len(self._sparse_era5_bboxes) > 1:
            self._fetch_sparse_forcing(tr, required_vars)
        else:
            if self._sparse_era5_bboxes and len(self._sparse_era5_bboxes) == 1:
                bbox = self._sparse_era5_bboxes[0]
            else:
                bbox = self._get_bbox()
            self._ds_surface, self._ds_pressure = self._nwp_source.fetch(
                bbox, tr, required_vars=required_vars,
            )

    def _fetch_sparse_forcing(
        self,
        time_range: list[str],
        required_vars: dict | None,
    ) -> None:
        """Fetch ERA5 for each sparse-points group and merge.

        Downloads a small ERA5 patch per group of points, then merges
        the patches into single surface and pressure datasets.  Overlapping
        grid cells are combined automatically by xarray; non-overlapping
        regions are filled with NaN (which the downscaler never reads).
        """
        surf_patches: list[xr.Dataset] = []
        pres_patches: list[xr.Dataset] = []

        for i, bbox in enumerate(self._sparse_era5_bboxes):
            log.info("Fetching ERA5 patch %d/%d: %s", i + 1, len(self._sparse_era5_bboxes), bbox)
            ds_s, ds_p = self._nwp_source.fetch(
                bbox, time_range, required_vars=required_vars,
            )
            surf_patches.append(ds_s)
            if ds_p is not None:
                pres_patches.append(ds_p)

        # Merge patches — overlapping cells keep one value, gaps become NaN
        self._ds_surface = xr.merge(surf_patches, join="outer")
        self._ds_pressure = (
            xr.merge(pres_patches, join="outer") if pres_patches else None
        )
        log.info("Merged %d ERA5 patches", len(surf_patches))

    def fetch_forecast(self, backfill_days: int | None = None) -> None:
        """Fetch forecast data using ForecastSource.

        Parameters
        ----------
        backfill_days : int, optional
            Override ``forecast.backfill_days`` from config. Used by the
            forecast pipeline to ensure enough backfill for ERA5 overlap.

        Stores results in ``_ds_forecast_surface`` and
        ``_ds_forecast_pressure`` for later blending or direct use.
        """
        try:
            from topopyscale2.inputs.forecast_source import ForecastSource
        except ModuleNotFoundError as e:
            raise not_in_release("Forecast downscaling", e, "topopyscale2.inputs.forecast_source") from e

        fc_cfg = self.config.forecast
        source = ForecastSource(
            self.config.inputs, self.config.storage, fc_cfg
        )
        bbox = self._get_bbox()
        self._ds_forecast_surface, self._ds_forecast_pressure = source.fetch(
            bbox, backfill_days=backfill_days,
        )
        log.info("Forecast data fetched")

    def fetch_and_blend(self, time_range: Optional[list[str]] = None) -> None:
        """Fetch reanalysis + forecast, then blend into a seamless timeline.

        Calls :meth:`fetch_forcing` for reanalysis, :meth:`fetch_forecast`
        for the forecast, then applies the configured blending method.
        The blended result is stored in ``_ds_surface`` / ``_ds_pressure``
        so downstream pipeline steps work unchanged.
        """
        try:
            from topopyscale2.inputs.temporal_blend import blend_reanalysis_forecast
        except ModuleNotFoundError as e:
            raise not_in_release("Forecast blending", e, "topopyscale2.inputs.temporal_blend") from e

        # 1. Fetch reanalysis (ERA5 or IFS hindcast)
        self.fetch_forcing(time_range=time_range)

        # 2. Fetch forecast
        self.fetch_forecast()

        # 3. Blend
        fc_cfg = self.config.forecast
        blending = fc_cfg.blending

        if blending.enabled and self._ds_forecast_surface is not None:
            log.info(
                "Blending reanalysis and forecast (method=%s, crossfade=%dh)",
                blending.method,
                blending.crossfade_hours,
            )
            self._ds_surface, self._ds_pressure = blend_reanalysis_forecast(
                ds_reanalysis_surf=self._ds_surface,
                ds_reanalysis_plev=self._ds_pressure,
                ds_forecast_surf=self._ds_forecast_surface,
                ds_forecast_plev=self._ds_forecast_pressure,
                method=blending.method,
                crossfade_hours=blending.crossfade_hours,
            )
        else:
            log.info("Blending disabled; using forecast data directly")
            self._ds_surface = self._ds_forecast_surface
            self._ds_pressure = self._ds_forecast_pressure

    def _get_bbox(self) -> tuple[float, float, float, float]:
        """Get bounding box from config, padded by 1 degree for ERA5."""
        if self.config.domain.bbox:
            w, s, e, n = self.config.domain.bbox
        elif (
            self.config.domain.spatial_mode == "points"
            and self.config.domain.points
            and self.config.domain.points.coordinates
        ):
            # Derive bbox from point coordinates (sparse points mode)
            coords = self.config.domain.points.coordinates
            lons = [p.lon for p in coords]
            lats = [p.lat for p in coords]
            w, e = min(lons), max(lons)
            s, n = min(lats), max(lats)
        elif self.dem_data is not None and "elevation" in self.dem_data:
            w = float(self.dem_data.x.min())
            e = float(self.dem_data.x.max())
            s = float(self.dem_data.y.min())
            n = float(self.dem_data.y.max())
        else:
            raise ValueError("No bbox or DEM data available")

        # Pad by 1 degree for ERA5 interpolation
        return (w - 1.0, s - 1.0, e + 1.0, n + 1.0)

    def info(self) -> str:
        """Return summary information about the domain."""
        lines = ["Domain Summary", "=" * 40]

        if self.config.domain.bbox:
            lines.append(f"  BBox: {self.config.domain.bbox}")
        if self.config.domain.dem:
            lines.append(f"  DEM: {self.config.domain.dem}")

        if self.dem_data is not None and "elevation" in self.dem_data:
            elev = self.dem_data["elevation"]
            lines.append(f"  Elevation range: {float(elev.min()):.0f} – {float(elev.max()):.0f} m")
            lines.append(f"  Grid size: {elev.shape[0]} × {elev.shape[1]}")

        if self.units is not None:
            lines.append(f"  Spatial units: {len(self.units)}")

            # Surface type breakdown
            types = {}
            for u in self.units:
                types[u.surface_type] = types.get(u.surface_type, 0) + 1
            for t, c in sorted(types.items()):
                lines.append(f"    {t}: {c}")

            elevations = self.units.elevations()
            lines.append(f"  Unit elevation range: {elevations.min():.0f} – {elevations.max():.0f} m")

        lines.append(f"  Kernel backend: {self.config.execution.kernel_backend}")
        lines.append(f"  Spatial mode: {self.config.domain.spatial_mode}")

        return "\n".join(lines)

    def get_membership(self) -> Optional[xr.DataArray]:
        """Return cluster/polygon membership raster (pixel → unit id).

        Returns
        -------
        xr.DataArray or None
            2D raster with dimensions (y, x) containing integer unit IDs.
            -1 indicates unassigned/nodata pixels.
            None if setup() hasn't been called.
        """
        return self._membership

    def persist_domain_data(self) -> Path:
        """Save domain data including membership to zarr for later reuse.

        Persists DEM data and cluster membership to a zarr store in the
        dem_cache directory. This allows reloading domain state without
        re-running setup().

        Returns
        -------
        Path
            Path to the saved domain_data.zarr store.

        Raises
        ------
        ValueError
            If setup() hasn't been called yet.
        """
        if self.dem_data is None:
            raise ValueError("No domain data to persist. Run setup() first.")

        # Create a copy of dem_data to add membership
        domain_data = self.dem_data.copy()

        # Add membership if available
        if self._membership is not None:
            domain_data["cluster_membership"] = self._membership

        # Add unit metadata as attributes
        if self.units is not None:
            domain_data.attrs["n_units"] = len(self.units)
            domain_data.attrs["spatial_mode"] = self.config.domain.spatial_mode

        # Persist to zarr
        cache_path = Path(self.config.storage.dem_cache)
        cache_path.mkdir(parents=True, exist_ok=True)
        output_path = cache_path / "domain_data.zarr"

        domain_data.to_zarr(str(output_path), mode="w")

        return output_path

    def load_domain_data(self) -> bool:
        """Load persisted domain data if available.

        Attempts to load domain data from the dem_cache directory.
        If successful, sets self.dem_data, self._membership, and
        reconstructs self.units from the membership array.

        Returns
        -------
        bool
            True if domain data was loaded successfully, False otherwise.
        """
        cache_path = Path(self.config.storage.dem_cache) / "domain_data.zarr"

        if not cache_path.exists():
            return False

        try:
            domain_data = xr.open_zarr(str(cache_path))

            # Extract membership if present
            if "cluster_membership" in domain_data:
                self._membership = domain_data["cluster_membership"]
                # Remove from dem_data copy
                self.dem_data = domain_data.drop_vars("cluster_membership")
            else:
                self.dem_data = domain_data
                self._membership = None

            # Reconstruct units from membership and DEM data
            if self._membership is not None:
                self.units = self._reconstruct_units()

            return True
        except Exception:
            return False

    def _reconstruct_units(self) -> SpatialUnitCollection:
        """Reconstruct SpatialUnits from membership array and DEM data.

        Returns
        -------
        SpatialUnitCollection
            Collection of reconstructed spatial units.
        """
        if self._membership is None or self.dem_data is None:
            return SpatialUnitCollection([])

        membership = self._membership.values
        elev = self.dem_data["elevation"]

        # Get resolution for area calculations
        if "resolution_m" in elev.attrs:
            res = float(elev.attrs["resolution_m"])
        elif len(elev.x) > 1:
            res = abs(float(elev.x[1] - elev.x[0]))
        else:
            res = 30.0
        pixel_area = res * res

        # Get coordinate arrays
        x_coords = self.dem_data.x.values
        y_coords = self.dem_data.y.values
        xx, yy = np.meshgrid(x_coords, y_coords)

        # Get terrain attributes
        elev_vals = elev.values
        slope_vals = self.dem_data["slope"].values if "slope" in self.dem_data else None
        aspect_vals = self.dem_data["aspect"].values if "aspect" in self.dem_data else None
        svf_vals = self.dem_data["svf"].values if "svf" in self.dem_data else None

        # Find unique unit IDs
        unique_ids = np.unique(membership)
        unique_ids = unique_ids[unique_ids >= 0]  # Exclude -1 (nodata)

        units = []
        for uid in unique_ids:
            mask = membership == uid

            # Compute centroid
            cx = float(np.mean(xx[mask]))
            cy = float(np.mean(yy[mask]))
            cz = float(np.nanmean(elev_vals[mask]))

            # Compute attributes
            attrs = {"elevation": cz}
            if slope_vals is not None:
                slope_mean = np.nanmean(slope_vals[mask])
                attrs["slope"] = float(slope_mean) if not np.isnan(slope_mean) else 0.0
            if aspect_vals is not None:
                valid_asp = aspect_vals[mask]
                valid_asp = valid_asp[~np.isnan(valid_asp)]
                if len(valid_asp) > 0:
                    sin_a = np.mean(np.sin(np.radians(valid_asp)))
                    cos_a = np.mean(np.cos(np.radians(valid_asp)))
                    attrs["aspect"] = float(np.degrees(np.arctan2(sin_a, cos_a)) % 360)
                else:
                    attrs["aspect"] = 0.0
            if svf_vals is not None:
                svf_mean = np.nanmean(svf_vals[mask])
                attrs["svf"] = float(svf_mean) if not np.isnan(svf_mean) else 1.0

            # Determine unit ID format based on spatial mode
            spatial_mode = self.config.domain.spatial_mode
            if spatial_mode == "polygons":
                unit_id = f"hru_{int(uid):04d}"
            else:
                unit_id = f"unit_{int(uid):04d}"

            unit = SpatialUnit(
                id=unit_id,
                centroid=(cx, cy, cz),
                attributes=attrs,
                surface_type="open",
                area_m2=float(np.sum(mask)) * pixel_area,
            )
            units.append(unit)

        return SpatialUnitCollection(units)
