"""TopoSUB clustering with surface-type stratification."""

from typing import Optional

import numpy as np
import xarray as xr
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from topopyscale2.config.schema import ClusteringConfig, SurfaceTypeConfig
from topopyscale2.spatial.surface_types import SURFACE_TYPE_NAMES
from topopyscale2.spatial.units import SpatialUnit

# Threshold for switching to MiniBatchKMeans (number of samples)
MINIBATCH_THRESHOLD = 500_000  # 500K pixels


class TopoSUBClustering:
    """K-means clustering in topographic feature space."""

    def __init__(
        self,
        config: ClusteringConfig,
        surface_config: Optional[SurfaceTypeConfig] = None,
    ):
        self.config = config
        self.surface_config = surface_config
        self._membership: Optional[xr.DataArray] = None
        self._labels: Optional[np.ndarray] = None
        self._features_scaled: Optional[np.ndarray] = None
        self._inertia: Optional[float] = None

    def fit(
        self,
        dem_data: xr.Dataset,
        surface_types: Optional[xr.DataArray] = None,
        domain_mask: Optional[np.ndarray] = None,
    ) -> list[SpatialUnit]:
        """Cluster DEM features into spatial units.

        Parameters
        ----------
        dem_data : xr.Dataset
            Must contain variables matching config.features
            (elevation, slope, aspect, svf, etc.).
        surface_types : xr.DataArray, optional
            Raster of surface type labels. Required if stratification is enabled.
        domain_mask : np.ndarray, optional
            Boolean array (ny, nx); True = inside the modelling domain.
            Pixels outside the mask are excluded from clustering (membership
            stays -1). Used to restrict the domain to an arbitrary footprint
            (e.g. a basin shapefile) instead of the full DEM rectangle.

        Returns
        -------
        list[SpatialUnit]
        """
        features, weights = self._extract_features(dem_data)
        ny, nx = dem_data["elevation"].shape

        # Apply an external domain mask by marking out-of-domain pixels as NaN
        # in the feature array. The per-path finite filter then excludes them,
        # so this works uniformly for single/terrain-adaptive/stratified modes
        # without touching their masking logic.
        if domain_mask is not None:
            if domain_mask.shape != (ny, nx):
                raise ValueError(
                    f"domain_mask shape {domain_mask.shape} does not match "
                    f"DEM grid {(ny, nx)}"
                )
            outside = ~domain_mask.ravel()
            if outside.any():
                features[outside] = np.nan
                print(
                    f"  Domain mask: {int((~outside).sum()):,} / {outside.size:,} "
                    f"pixels inside domain"
                )

        # Guardrail: n_clusters must be at least the number of ERA5 0.25° cells
        # covering the domain, to avoid degrading ERA5 spatial resolution.
        n_era5 = self._estimate_era5_cells(dem_data)
        if n_era5 > 0 and self.config.n_clusters < n_era5:
            print(
                f"  ⚠ n_clusters={self.config.n_clusters} < ERA5 grid cells={n_era5} "
                f"— raising to {n_era5} to preserve ERA5 resolution"
            )
            self.config.n_clusters = n_era5

        stratify = (
            surface_types is not None
            and self.surface_config is not None
            and self.surface_config.stratify
        )
        # Surface type is the outer split and terrain the inner one, never
        # either-or. Ranking them the other way silently discarded the surface
        # types whenever terrain_adaptive was set -- which is the case for the
        # production Central Asia config, so glacier stratification would have
        # looked enabled and done nothing.
        if stratify:
            return self._fit_stratified(features, weights, dem_data, surface_types, ny, nx)
        elif self.config.terrain_adaptive:
            return self._fit_terrain_adaptive(features, weights, dem_data, ny, nx)
        else:
            return self._fit_single(features, weights, dem_data, ny, nx, surface_type="open")

    @staticmethod
    def _estimate_era5_cells(dem_data: xr.Dataset) -> int:
        """Estimate number of ERA5 0.25° grid cells covering the domain.

        Uses the DEM coordinate extent. If the DEM is in a projected CRS,
        falls back to a rough estimate from coordinate range.
        """
        x = dem_data.x.values
        y = dem_data.y.values

        # Check if coordinates are geographic (degrees) or projected (metres)
        x_range = float(x.max() - x.min())
        y_range = float(y.max() - y.min())

        if x_range < 360 and y_range < 180:
            # Geographic coordinates (degrees)
            lon_cells = max(1, int(np.ceil(x_range / 0.25)))
            lat_cells = max(1, int(np.ceil(y_range / 0.25)))
        else:
            # Projected coordinates (metres) — convert to approximate degrees
            # Use 111 km per degree as rough estimate
            lon_deg = x_range / 111_000
            lat_deg = y_range / 111_000
            lon_cells = max(1, int(np.ceil(lon_deg / 0.25)))
            lat_cells = max(1, int(np.ceil(lat_deg / 0.25)))

        return lon_cells * lat_cells

    def _extract_features(self, dem_data: xr.Dataset) -> tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract and stack feature arrays from DEM dataset.

        Memory-efficient: pre-allocates a single output array and fills columns
        one at a time, avoiding large intermediate 2D arrays.

        Returns
        -------
        features : np.ndarray
            Stacked feature array (n_pixels, n_features), float32 for large DEMs
        weights : np.ndarray or None
            Feature weights matching extracted features (skipped features removed)
        """
        ny, nx = dem_data["elevation"].shape
        n_pixels = ny * nx

        # First pass: count features (some may be skipped)
        weight_indices = []
        feat_names = []
        for i, feat in enumerate(self.config.features):
            if feat == "svf" and feat not in dem_data:
                print(f"  Note: '{feat}' not in DEM (will be computed at centroids)")
                continue
            feat_names.append(feat)
            weight_indices.append(i)

        n_features = len(feat_names)
        # Use float32 for large DEMs to halve memory
        dtype = np.float32 if n_pixels > 50_000_000 else np.float64
        features = np.empty((n_pixels, n_features), dtype=dtype)

        # Cache raveled slope/aspect radians if needed by multiple features
        _slope_rad_flat = None
        _aspect_rad_flat = None
        needs_slope_rad = {"sx", "sy", "northness"}
        needs_aspect_rad = {"sx", "sy", "northness", "cos_aspect", "sin_aspect"}
        feat_set = set(feat_names)

        if feat_set & needs_slope_rad:
            _slope_rad_flat = dem_data["slope"].values.ravel().astype(dtype)
            np.radians(_slope_rad_flat, out=_slope_rad_flat)
        if feat_set & needs_aspect_rad:
            _aspect_rad_flat = dem_data["aspect"].values.ravel().astype(dtype)
            np.radians(_aspect_rad_flat, out=_aspect_rad_flat)

        # Reusable buffer for trig intermediates (avoids temporary allocations)
        _buf = np.empty(n_pixels, dtype=dtype) if feat_set & needs_slope_rad else None

        for col, feat in enumerate(feat_names):
            if feat == "cos_aspect":
                np.cos(_aspect_rad_flat, out=features[:, col])
            elif feat == "sin_aspect":
                np.sin(_aspect_rad_flat, out=features[:, col])
            elif feat in ("sx", "northness"):
                np.sin(_slope_rad_flat, out=features[:, col])
                np.cos(_aspect_rad_flat, out=_buf)
                features[:, col] *= _buf
            elif feat == "sy":
                np.sin(_slope_rad_flat, out=features[:, col])
                np.sin(_aspect_rad_flat, out=_buf)
                features[:, col] *= _buf
            elif feat == "x":
                features[:, col] = np.tile(dem_data.x.values.astype(dtype), ny)
            elif feat == "y":
                features[:, col] = np.repeat(dem_data.y.values.astype(dtype), nx)
            elif feat in dem_data:
                features[:, col] = dem_data[feat].values.ravel().astype(dtype)
            else:
                raise ValueError(f"Feature '{feat}' not found in DEM dataset")

        # Free cached arrays
        del _slope_rad_flat, _aspect_rad_flat, _buf

        weights = None
        if self.config.feature_weights is not None:
            weights = np.array([self.config.feature_weights[i] for i in weight_indices])

        return features, weights

    def _fit_single(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray],
        dem_data: xr.Dataset,
        ny: int,
        nx: int,
        surface_type: str,
        n_clusters: Optional[int] = None,
        mask: Optional[np.ndarray] = None,
        id_offset: int = 0,
    ) -> list[SpatialUnit]:
        """Fit K-means on a single group of pixels."""
        n_clusters = n_clusters or self.config.n_clusters

        if mask is not None:
            valid = mask.ravel()
        else:
            valid = np.ones(features.shape[0], dtype=bool)

        # Filter out NaN values from features
        finite_mask = np.all(np.isfinite(features), axis=1)
        valid = valid & finite_mask
        del finite_mask

        n_valid = int(valid.sum())
        if n_valid == 0:
            return []

        n_clusters = min(n_clusters, n_valid)

        # For large datasets, standardize and weight in-place to avoid copies
        n_pixels = features.shape[0]
        large_dataset = n_pixels > 50_000_000

        if large_dataset:
            valid_idx = np.where(valid)[0]
            del valid  # Free bool array (~500 MB)

            # Subsample for stats and training (2M is plenty)
            import os
            rng = np.random.RandomState(42)
            max_sample = min(2_000_000, len(valid_idx))
            sample_idx = rng.choice(valid_idx, size=max_sample, replace=False)
            sample_feats = features[sample_idx]  # Small: 2M × 6 × 4 = ~48 MB

            if self.config.feature_standardization:
                means = np.mean(sample_feats, axis=0)
                stds = np.std(sample_feats, axis=0)
                stds[stds == 0] = 1.0
                # Standardize in-place on the original features array
                features -= means
                features /= stds
                # Re-extract sample after standardization
                sample_feats = features[sample_idx]

            if weights is not None:
                features *= weights
                sample_feats = features[sample_idx]

            n_cores = os.cpu_count() or 4
            batch_size = 256 * n_cores

            print(f"  Using MiniBatchKMeans ({n_valid:,} samples, {n_clusters} clusters, "
                  f"train={max_sample:,}, batch={batch_size})...", flush=True)
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=batch_size,
            )
            kmeans.fit(sample_feats)
            del sample_feats

            # Predict in chunks to avoid creating one huge array
            chunk_size = 1_000_000
            labels = np.empty(len(valid_idx), dtype=np.int32)
            for ci in range(0, len(valid_idx), chunk_size):
                chunk_idx = valid_idx[ci:ci + chunk_size]
                labels[ci:ci + chunk_size] = kmeans.predict(features[chunk_idx])

            self._labels = labels
            self._inertia = kmeans.inertia_
            self._features_scaled = None
            del features
        else:
            feats = features[valid]
            if self.config.feature_standardization:
                scaler = StandardScaler()
                feats_scaled = scaler.fit_transform(feats)
            else:
                feats_scaled = feats
            if weights is not None:
                feats_scaled = feats_scaled * weights

            # Use MiniBatchKMeans for large datasets (much faster)
            n_samples = len(feats_scaled)
            if n_samples > MINIBATCH_THRESHOLD:
                import os
                if self.config.minibatch_batch_size is not None:
                    batch_size = self.config.minibatch_batch_size
                else:
                    n_cores = os.cpu_count() or 4
                    batch_size = 256 * n_cores
                print(f"  Using MiniBatchKMeans ({n_samples:,} samples, {n_clusters} clusters, batch={batch_size})...", flush=True)
                kmeans = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=batch_size,
                )
            else:
                print(f"  Using KMeans ({n_samples:,} samples, {n_clusters} clusters)...", flush=True)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

            labels = kmeans.fit_predict(feats_scaled)
            self._labels = labels
            self._inertia = kmeans.inertia_
            self._features_scaled = feats_scaled

        print("  Clustering complete.", flush=True)

        # Build membership raster as 2D with coordinates (like HRUPolygonGenerator)
        if self._membership is None:
            self._membership = xr.DataArray(
                np.full((ny, nx), -1, dtype=np.int32),
                dims=["y", "x"],
                coords={"y": dem_data.y, "x": dem_data.x},
            )
        # Convert valid mask to 2D for assignment
        membership_values = self._membership.values.copy()
        if large_dataset:
            # Reconstruct valid bool from valid_idx to avoid keeping full bool array
            membership_flat = membership_values.ravel()
            membership_flat[valid_idx] = labels + id_offset
            membership_values = membership_flat.reshape(ny, nx)
        else:
            valid_2d = valid.reshape(ny, nx)
            membership_values[valid_2d] = labels + id_offset
        self._membership = xr.DataArray(
            membership_values,
            dims=["y", "x"],
            coords={"y": dem_data.y, "x": dem_data.x},
        )

        # Compute pixel area
        elev = dem_data["elevation"]
        res = float(elev.attrs.get("resolution_m", abs(float(elev.x[1] - elev.x[0]))))
        pixel_area = res * res

        # Build spatial units — sort-and-split for O(n log n) instead of O(n × k)
        units = []
        _valid_flat_idx = valid_idx if large_dataset else np.where(valid)[0]

        # Sort pixel indices by cluster label (one pass, not per-cluster scans)
        sort_order = np.argsort(labels, kind="stable")
        sorted_labels = labels[sort_order]
        sorted_pixel_idx = _valid_flat_idx[sort_order]
        # Find split points between clusters
        split_points = np.searchsorted(sorted_labels, np.arange(1, n_clusters))
        cluster_groups = np.split(sorted_pixel_idx, split_points)
        del sort_order, sorted_labels, sorted_pixel_idx, _valid_flat_idx

        elev_flat = dem_data["elevation"].values.ravel()
        x_coords = dem_data.x.values
        y_coords = dem_data.y.values
        slope_flat = dem_data["slope"].values.ravel() if "slope" in dem_data else None
        aspect_flat = dem_data["aspect"].values.ravel() if "aspect" in dem_data else None
        svf_flat = dem_data["svf"].values.ravel() if "svf" in dem_data else None

        for k in range(n_clusters):
            pixel_indices = cluster_groups[k]

            # Derive x/y from flat index (no meshgrid needed)
            rows = pixel_indices // nx
            cols = pixel_indices % nx
            cx = float(np.mean(x_coords[cols]))
            cy = float(np.mean(y_coords[rows]))
            cz = float(np.mean(elev_flat[pixel_indices]))

            attrs = {"elevation": cz}
            if slope_flat is not None:
                slope_mean = np.nanmean(slope_flat[pixel_indices])
                attrs["slope"] = float(slope_mean) if not np.isnan(slope_mean) else 0.0
            if aspect_flat is not None:
                valid_aspects = aspect_flat[pixel_indices]
                valid_aspects = valid_aspects[~np.isnan(valid_aspects)]
                if len(valid_aspects) > 0:
                    sin_a = np.mean(np.sin(np.radians(valid_aspects)))
                    cos_a = np.mean(np.cos(np.radians(valid_aspects)))
                    attrs["aspect"] = float(np.degrees(np.arctan2(sin_a, cos_a)) % 360)
                else:
                    attrs["aspect"] = 0.0
            if svf_flat is not None:
                svf_mean = np.nanmean(svf_flat[pixel_indices])
                attrs["svf"] = float(svf_mean) if not np.isnan(svf_mean) else 1.0

            unit = SpatialUnit(
                id=f"unit_{k + id_offset:04d}",
                centroid=(cx, cy, cz),
                attributes=attrs,
                surface_type=surface_type,
                area_m2=float(len(pixel_indices)) * pixel_area,
            )
            units.append(unit)

        return units

    def _fit_terrain_adaptive(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray],
        dem_data: xr.Dataset,
        ny: int,
        nx: int,
        mask: Optional[np.ndarray] = None,
        n_clusters: Optional[int] = None,
        surface_type: str = "open",
        id_offset: int = 0,
    ) -> list[SpatialUnit]:
        """Split domain into flat and mountain zones, cluster each independently.

        Flat zone (slope ≤ threshold): clustered on x, y only.
        Mountain zone (slope > threshold): clustered on configured features.
        Cluster count is split by terrain_mountain_fraction.

        ``mask``/``n_clusters``/``surface_type``/``id_offset`` let this run
        inside one surface-type stratum, so a domain can be split by surface
        type and by terrain at the same time.
        """
        slope_thresh = self.config.terrain_slope_threshold
        mtn_frac = self.config.terrain_mountain_fraction
        n_total = self.config.n_clusters if n_clusters is None else n_clusters
        in_stratum = mask if mask is not None else True

        # Build slope mask
        slope = dem_data["slope"].values  # (ny, nx) in degrees
        slope_flat = slope.ravel()
        finite = np.all(np.isfinite(features), axis=1)
        is_mountain = (slope_flat > slope_thresh) & finite & in_stratum
        is_flat = (slope_flat <= slope_thresh) & finite & in_stratum

        n_mtn_pix = is_mountain.sum()
        n_flat_pix = is_flat.sum()
        n_mtn_clusters = max(1, int(n_total * mtn_frac))
        n_flat_clusters = max(1, n_total - n_mtn_clusters)

        # Guardrail: flat clusters must be >= ERA5 grid cells to avoid
        # degrading ERA5 spatial resolution in flat terrain. It is a statement
        # about the whole domain, so it does not apply inside a stratum -- the
        # glacier stratum covers ~3% of Central Asia and raising its flat count
        # to the domain's ERA5 cell count would swamp the run with ice units.
        n_era5 = 0 if mask is not None else self._estimate_era5_cells(dem_data)
        if n_era5 > 0 and n_flat_clusters < n_era5:
            print(
                f"  ⚠ Flat clusters ({n_flat_clusters}) < ERA5 grid cells ({n_era5})"
                f" — raising flat count to {n_era5}"
            )
            n_flat_clusters = n_era5

        # Don't allocate more clusters than pixels
        n_mtn_clusters = min(n_mtn_clusters, n_mtn_pix)
        n_flat_clusters = min(n_flat_clusters, n_flat_pix)

        print(f"  Terrain-adaptive clustering [{surface_type}] "
              f"(slope threshold={slope_thresh}°):")
        print(f"    Flat:     {n_flat_pix:,} pixels → {n_flat_clusters} clusters")
        print(f"    Mountain: {n_mtn_pix:,} pixels → {n_mtn_clusters} clusters")
        print(f"    Total:    {n_flat_clusters + n_mtn_clusters} clusters")

        all_units = []

        # Both zones use the same features — the difference is cluster density.
        # Flat terrain naturally has low northness variance (slope~0), so
        # x, y, elevation dominate the clustering there regardless.

        # Cluster flat zone
        print("  Clustering flat terrain...", flush=True)
        flat_units = self._fit_single(
            features, weights, dem_data, ny, nx,
            surface_type=surface_type,
            n_clusters=n_flat_clusters,
            mask=is_flat,
            id_offset=id_offset,
        )
        all_units.extend(flat_units)

        # Cluster mountain zone
        print("  Clustering mountain terrain...", flush=True)
        mtn_units = self._fit_single(
            features, weights, dem_data, ny, nx,
            surface_type=surface_type,
            n_clusters=n_mtn_clusters,
            mask=is_mountain,
            id_offset=id_offset + len(flat_units),
        )
        all_units.extend(mtn_units)

        print(f"  Total: {len(all_units)} units "
              f"({len(flat_units)} flat + {len(mtn_units)} mountain)")

        return all_units

    # Default mapping from integer surface type codes to names
    #: Imported rather than redefined -- see surface_types.SURFACE_TYPE_CODES.
    SURFACE_TYPE_NAMES = SURFACE_TYPE_NAMES

    def _fit_stratified(
        self,
        features: np.ndarray,
        weights: Optional[np.ndarray],
        dem_data: xr.Dataset,
        surface_types: xr.DataArray,
        ny: int,
        nx: int,
    ) -> list[SpatialUnit]:
        """Cluster independently within each surface type.

        Each surface type gets its own set of clusters, ensuring glacier
        pixels are never mixed with non-glacier clusters (and vice versa).
        Per-type n_clusters can be set in config surface_types.types.
        """
        st_flat = surface_types.values.ravel()
        unique_types = np.unique(st_flat)
        unique_types = unique_types[~np.isnan(unique_types.astype(float))]

        # Count only pixels that will actually be clustered. fit() marks
        # out-of-domain pixels NaN, and _fit_single drops them, so counting the
        # raw raster would allocate clusters for terrain outside the domain --
        # on a basin-masked domain that skews every proportional share. Seen on
        # Central Asia: 7.4 M raster pixels against 1.9 M inside the basins.
        in_domain = np.isfinite(features).all(axis=1)

        all_units = []
        id_offset = 0

        # Build code→name map: use defaults (0=open, 1=glacier, etc.)
        # Config types dict specifies per-type settings, not code mapping
        code_to_name = dict(self.SURFACE_TYPE_NAMES)

        # Build name→n_clusters from config
        name_to_nclusters = {}
        if self.surface_config and self.surface_config.types:
            for name, entry in self.surface_config.types.items():
                if entry.n_clusters is not None:
                    name_to_nclusters[name] = entry.n_clusters

        total_pixels = int(in_domain.sum())
        for st_val in unique_types:
            st_code = int(st_val)
            st_name = code_to_name.get(st_code, f"type_{st_code}")
            mask = (st_flat == st_val) & in_domain
            n_pixels = int(mask.sum())

            # Skip types with too few pixels
            if n_pixels < 2:
                continue

            # Determine n_clusters for this type
            if st_name in name_to_nclusters:
                n_clusters = name_to_nclusters[st_name]
            else:
                # Proportional: allocate clusters by pixel fraction
                frac = n_pixels / total_pixels
                n_clusters = max(1, int(self.config.n_clusters * frac))

            # Don't exceed pixel count
            n_clusters = min(n_clusters, n_pixels)

            print(f"  Stratified clustering: {st_name} ({n_pixels} pixels, {n_clusters} clusters)")
            fit = (
                self._fit_terrain_adaptive
                if self.config.terrain_adaptive
                else self._fit_single
            )
            units = fit(
                features, weights, dem_data, ny, nx,
                surface_type=st_name,
                n_clusters=n_clusters,
                mask=mask.ravel(),
                id_offset=id_offset,
            )
            id_offset += len(units)
            all_units.extend(units)

        return all_units

    def evaluate(self, max_silhouette_samples: int = 50_000) -> dict:
        """Compute cluster quality metrics on the most recent fit.

        Parameters
        ----------
        max_silhouette_samples : int
            Maximum samples for silhouette score (subsampled for speed,
            since silhouette is O(n^2)).

        Returns
        -------
        dict with keys:
            n_clusters, n_samples, inertia,
            calinski_harabasz, davies_bouldin, silhouette
        """
        if self._labels is None or self._features_scaled is None:
            raise RuntimeError("Must call fit() before evaluate()")

        labels = self._labels
        feats = self._features_scaled
        n_samples = len(labels)
        n_clusters = len(np.unique(labels))

        metrics: dict = {
            "n_clusters": n_clusters,
            "n_samples": n_samples,
            "inertia": self._inertia,
        }

        # Calinski-Harabasz and Davies-Bouldin are fast (linear)
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(feats, labels))
        metrics["davies_bouldin"] = float(davies_bouldin_score(feats, labels))

        # Silhouette is O(n^2) — subsample for large datasets
        if n_samples > max_silhouette_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(n_samples, max_silhouette_samples, replace=False)
            metrics["silhouette"] = float(silhouette_score(feats[idx], labels[idx]))
            metrics["silhouette_sampled"] = True
        else:
            metrics["silhouette"] = float(silhouette_score(feats, labels))
            metrics["silhouette_sampled"] = False

        return metrics

    def elbow_analysis(
        self,
        dem_data: xr.Dataset,
        k_values: list[int],
        surface_types: Optional[xr.DataArray] = None,
        max_silhouette_samples: int = 50_000,
    ) -> list[dict]:
        """Run clustering at multiple k values and return metrics for each.

        Parameters
        ----------
        dem_data : xr.Dataset
            Same DEM data as fit().
        k_values : list[int]
            Cluster counts to evaluate.
        surface_types : xr.DataArray, optional
            Surface types for stratified clustering.
        max_silhouette_samples : int
            Max samples for silhouette computation.

        Returns
        -------
        list[dict]
            Metrics dict for each k value.
        """
        results = []
        original_k = self.config.n_clusters

        for k in k_values:
            print(f"\n--- Evaluating k={k} ---", flush=True)
            self.config.n_clusters = k
            # Reset state
            self._membership = None
            self._labels = None
            self._features_scaled = None
            self._inertia = None

            self.fit(dem_data, surface_types)
            metrics = self.evaluate(max_silhouette_samples)
            results.append(metrics)

        # Restore original config
        self.config.n_clusters = original_k
        return results

    def get_membership(self) -> Optional[xr.DataArray]:
        """Return cluster membership raster (pixel → cluster id)."""
        return self._membership
