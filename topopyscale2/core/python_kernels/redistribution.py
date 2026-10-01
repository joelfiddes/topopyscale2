"""Snow redistribution kernels — Python reference implementation.

SYNC WARNING: This file must stay in sync with:
  - src/redistribution.rs
  - topopyscale2/core/jax_kernels/redistribution.py
Any changes here MUST be applied to all three backends in the same commit.

Implements wind-driven snow transport and gravitational (avalanche) redistribution.
"""

import numpy as np


def wind_transport(
    swe: np.ndarray,
    sx: np.ndarray,
    wind_speed: np.ndarray,
    transport_coeff: float,
    area: np.ndarray,
    neighbor_indices: np.ndarray,
    neighbor_weights: np.ndarray,
) -> np.ndarray:
    """Compute wind-driven snow redistribution.

    Erodes snow from exposed terrain (high Sx) and deposits on sheltered
    terrain (low Sx). The transport is mass-conservative: sum(swe * area)
    is preserved.

    Algorithm:
    1. Compute erosion potential based on exposure (positive Sx) and wind
    2. Compute deposition potential based on shelter (negative Sx)
    3. Transport mass from eroding cells to neighboring sheltered cells
    4. Scale to maintain mass conservation

    SYNC: src/redistribution.rs::wind_transport
    SYNC: jax_kernels/redistribution.py::wind_transport

    Parameters
    ----------
    swe : array_like, shape (n_units,)
        Snow water equivalent for each spatial unit [mm or kg/m^2].
    sx : array_like, shape (n_units,)
        Winstral Sx terrain parameter [degrees]. Positive = exposed,
        negative = sheltered.
    wind_speed : array_like, shape (n_units,)
        Wind speed at each unit [m/s].
    transport_coeff : float
        Transport coefficient controlling erosion rate [1/m/s/degree].
        Typical range: 0.001 to 0.01.
    area : array_like, shape (n_units,)
        Area of each spatial unit [m^2].
    neighbor_indices : array_like, shape (n_units, max_neighbors)
        Indices of neighbors for each unit. Use -1 for missing neighbors.
    neighbor_weights : array_like, shape (n_units, max_neighbors)
        Weights for each neighbor (should sum to 1 for units with neighbors).
        Use 0.0 weight for missing neighbors (index=-1).

    Returns
    -------
    np.ndarray, shape (n_units,)
        Updated SWE after redistribution [mm or kg/m^2].
    """
    swe = np.asarray(swe, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)
    wind_speed = np.asarray(wind_speed, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    neighbor_weights = np.asarray(neighbor_weights, dtype=np.float64)

    n_units = len(swe)
    swe_new = swe.copy()

    # Erosion occurs where Sx > 0 (exposed)
    # Erosion rate is proportional to exposure, wind speed, and available snow
    # erosion_potential = transport_coeff * max(0, Sx) * wind_speed * swe
    erosion_potential = transport_coeff * np.maximum(0.0, sx) * wind_speed * swe

    # Cannot erode more than available
    erosion = np.minimum(erosion_potential, swe)

    # Remove eroded snow
    swe_new = swe_new - erosion

    # Compute total mass eroded
    total_eroded_mass = np.sum(erosion * area)

    if total_eroded_mass <= 0:
        return swe_new

    # Deposition occurs where Sx < 0 (sheltered)
    # Deposition affinity is proportional to shelter degree
    deposition_affinity = np.maximum(0.0, -sx)

    # Distribute eroded mass to neighbors based on their shelter affinity
    # Each eroding cell sends snow to its neighbors weighted by neighbor_weights
    # and neighbors' deposition affinity
    deposition = np.zeros_like(swe_new)

    for i in range(n_units):
        if erosion[i] <= 0:
            continue

        eroded_mass_i = erosion[i] * area[i]

        # Compute weighted deposition to neighbors
        weights_sum = 0.0
        for j_idx in range(neighbor_indices.shape[1]):
            j = neighbor_indices[i, j_idx]
            if j < 0:
                continue
            w = neighbor_weights[i, j_idx]
            affinity_j = deposition_affinity[j]
            weights_sum += w * affinity_j

        if weights_sum <= 0:
            # No sheltered neighbors - snow is lost (transported out of domain)
            # or distribute evenly to neighbors regardless of affinity
            for j_idx in range(neighbor_indices.shape[1]):
                j = neighbor_indices[i, j_idx]
                if j < 0:
                    continue
                w = neighbor_weights[i, j_idx]
                deposition[j] += (eroded_mass_i * w) / area[j]
        else:
            for j_idx in range(neighbor_indices.shape[1]):
                j = neighbor_indices[i, j_idx]
                if j < 0:
                    continue
                w = neighbor_weights[i, j_idx]
                affinity_j = deposition_affinity[j]
                fraction = (w * affinity_j) / weights_sum
                deposition[j] += (eroded_mass_i * fraction) / area[j]

    swe_new = swe_new + deposition

    return swe_new


def avalanche_redistribute(
    swe: np.ndarray,
    slope: np.ndarray,
    slope_threshold: float,
    flow_fractions: np.ndarray,
    neighbor_indices: np.ndarray,
    area: np.ndarray,
) -> np.ndarray:
    """Compute gravitational (avalanche) snow redistribution.

    Moves SWE from slopes exceeding threshold to downslope neighbors via
    D-infinity flow routing. Mass-conservative: sum(swe * area) preserved.

    Algorithm:
    1. Identify cells where slope > threshold
    2. Move snow from steep cells to downslope neighbors
    3. Use flow_fractions (D-inf style) for routing
    4. Process iteratively from steep to shallow

    SYNC: src/redistribution.rs::avalanche_redistribute
    SYNC: jax_kernels/redistribution.py::avalanche_redistribute

    Parameters
    ----------
    swe : array_like, shape (n_units,)
        Snow water equivalent for each spatial unit [mm or kg/m^2].
    slope : array_like, shape (n_units,)
        Slope angle for each unit [degrees].
    slope_threshold : float
        Minimum slope for avalanche triggering [degrees]. Typical: 30-45.
    flow_fractions : array_like, shape (n_units, max_neighbors)
        Fraction of flow going to each neighbor (D-inf style).
        Should sum to 1.0 for cells with outflow neighbors.
        Use 0.0 for non-neighbor slots.
    neighbor_indices : array_like, shape (n_units, max_neighbors)
        Indices of downslope neighbors for each unit.
        Use -1 for missing neighbors.
    area : array_like, shape (n_units,)
        Area of each spatial unit [m^2].

    Returns
    -------
    np.ndarray, shape (n_units,)
        Updated SWE after avalanche redistribution [mm or kg/m^2].
    """
    swe = np.asarray(swe, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    area = np.asarray(area, dtype=np.float64)
    flow_fractions = np.asarray(flow_fractions, dtype=np.float64)
    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)

    n_units = len(swe)
    swe_new = swe.copy()

    # Identify cells exceeding slope threshold
    steep_mask = slope > slope_threshold

    # Fraction of SWE to redistribute increases with slope excess
    # At exactly threshold: 0%, at 2x threshold: 100%
    # fraction = min(1, (slope - threshold) / threshold)
    slope_excess = np.maximum(0.0, slope - slope_threshold)
    redistribute_fraction = np.minimum(1.0, slope_excess / slope_threshold)

    # Process each steep cell
    for i in range(n_units):
        if not steep_mask[i]:
            continue

        # Amount to redistribute
        swe_to_move = swe_new[i] * redistribute_fraction[i]
        if swe_to_move <= 0:
            continue

        mass_to_move = swe_to_move * area[i]

        # Remove from source
        swe_new[i] -= swe_to_move

        # Distribute to downslope neighbors via flow fractions
        total_flow = 0.0
        for j_idx in range(neighbor_indices.shape[1]):
            j = neighbor_indices[i, j_idx]
            if j < 0:
                continue
            frac = flow_fractions[i, j_idx]
            if frac <= 0:
                continue
            total_flow += frac
            deposit_mass = mass_to_move * frac
            swe_new[j] += deposit_mass / area[j]

        # If no outflow neighbors, snow stays in place (shouldn't happen normally)
        if total_flow <= 0:
            swe_new[i] += swe_to_move

    return swe_new


def compute_transport_rate(
    wind_speed: np.ndarray,
    sx: np.ndarray,
    threshold_wind: float = 5.0,
    max_rate: float = 0.1,
) -> np.ndarray:
    """Compute potential transport rate based on wind and exposure.

    Helper function to compute mass transport rate from wind and terrain.
    Erosion occurs only when wind exceeds threshold on exposed terrain.

    SYNC: src/redistribution.rs::compute_transport_rate
    SYNC: jax_kernels/redistribution.py::compute_transport_rate

    Parameters
    ----------
    wind_speed : array_like
        Wind speed [m/s].
    sx : array_like
        Winstral Sx terrain parameter [degrees].
    threshold_wind : float, optional
        Minimum wind speed for transport [m/s]. Default 5.0.
    max_rate : float, optional
        Maximum transport rate fraction [0-1]. Default 0.1.

    Returns
    -------
    np.ndarray
        Transport rate (fraction of available snow that can be moved).
    """
    wind_speed = np.asarray(wind_speed, dtype=np.float64)
    sx = np.asarray(sx, dtype=np.float64)

    # Transport only occurs above wind threshold
    excess_wind = np.maximum(0.0, wind_speed - threshold_wind)

    # Scale by exposure (only exposed terrain contributes)
    exposure_factor = np.maximum(0.0, sx) / 15.0  # Normalized by typical Sx range

    # Rate increases with wind and exposure
    rate = 0.01 * excess_wind * exposure_factor

    # Cap at maximum rate
    return np.minimum(rate, max_rate)
