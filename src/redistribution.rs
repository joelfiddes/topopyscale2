// SYNC WARNING: This file must stay in sync with:
//   - topopyscale2/core/python_kernels/redistribution.py
//   - topopyscale2/core/jax_kernels/redistribution.py
// Any changes here MUST be applied to all three backends in the same commit.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Scalar pure function: compute_transport_rate
// ---------------------------------------------------------------------------

/// Compute potential transport rate based on wind and exposure (scalar).
///
/// Returns fraction of snow that can be transported.
#[inline]
fn compute_transport_rate_scalar(
    wind_speed: f64,
    sx: f64,
    threshold_wind: f64,
    max_rate: f64,
) -> f64 {
    // Transport only occurs above wind threshold
    let excess_wind = (wind_speed - threshold_wind).max(0.0);

    // Scale by exposure (only exposed terrain contributes)
    let exposure_factor = sx.max(0.0) / 15.0; // Normalized by typical Sx range

    // Rate increases with wind and exposure
    let rate = 0.01 * excess_wind * exposure_factor;

    // Cap at maximum rate
    rate.min(max_rate)
}

// ---------------------------------------------------------------------------
// Array wrapper: compute_transport_rate
// ---------------------------------------------------------------------------

/// Compute potential transport rate based on wind and exposure.
///
/// SYNC: python_kernels/redistribution.py::compute_transport_rate
/// SYNC: jax_kernels/redistribution.py::compute_transport_rate
#[pyfunction]
#[pyo3(signature = (wind_speed, sx, threshold_wind=5.0, max_rate=0.1))]
fn compute_transport_rate<'py>(
    py: Python<'py>,
    wind_speed: PyReadonlyArrayDyn<'py, f64>,
    sx: PyReadonlyArrayDyn<'py, f64>,
    threshold_wind: f64,
    max_rate: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let wind_speed = wind_speed.as_array();
    let sx = sx.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(wind_speed.shape());

    let ws_flat = wind_speed.as_slice().expect("contiguous array");
    let sx_flat = sx.as_slice().expect("contiguous array");
    let out_flat = out.as_slice_mut().expect("contiguous array");

    for i in 0..out_flat.len() {
        out_flat[i] = compute_transport_rate_scalar(ws_flat[i], sx_flat[i], threshold_wind, max_rate);
    }

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

// ---------------------------------------------------------------------------
// Array wrapper: wind_transport
// ---------------------------------------------------------------------------

/// Compute wind-driven snow redistribution.
///
/// Erodes snow from exposed terrain (high Sx) and deposits on sheltered
/// terrain (low Sx). Mass-conservative: sum(swe * area) is preserved.
///
/// SYNC: python_kernels/redistribution.py::wind_transport
/// SYNC: jax_kernels/redistribution.py::wind_transport
#[pyfunction]
fn wind_transport<'py>(
    py: Python<'py>,
    swe: PyReadonlyArrayDyn<'py, f64>,
    sx: PyReadonlyArrayDyn<'py, f64>,
    wind_speed: PyReadonlyArrayDyn<'py, f64>,
    transport_coeff: f64,
    area: PyReadonlyArrayDyn<'py, f64>,
    neighbor_indices: PyReadonlyArrayDyn<'py, i64>,
    neighbor_weights: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let swe = swe.as_array();
    let sx = sx.as_array();
    let wind_speed = wind_speed.as_array();
    let area = area.as_array();
    let neighbor_indices = neighbor_indices.as_array();
    let neighbor_weights = neighbor_weights.as_array();

    let swe_flat = swe.as_slice().expect("contiguous array");
    let sx_flat = sx.as_slice().expect("contiguous array");
    let ws_flat = wind_speed.as_slice().expect("contiguous array");
    let area_flat = area.as_slice().expect("contiguous array");

    let n_units = swe_flat.len();
    let max_neighbors = neighbor_indices.shape()[1];

    let mut swe_new = swe_flat.to_vec();

    // Compute erosion
    let mut erosion = vec![0.0; n_units];
    for i in 0..n_units {
        let erosion_potential = transport_coeff * sx_flat[i].max(0.0) * ws_flat[i] * swe_flat[i];
        erosion[i] = erosion_potential.min(swe_flat[i]);
        swe_new[i] -= erosion[i];
    }

    // Total eroded mass
    let total_eroded_mass: f64 = erosion.iter().zip(area_flat.iter()).map(|(e, a)| e * a).sum();

    if total_eroded_mass <= 0.0 {
        let out = ndarray::ArrayD::from_shape_vec(swe.shape(), swe_new).unwrap();
        return Ok(PyArrayDyn::from_owned_array_bound(py, out));
    }

    // Deposition affinity (shelter = negative Sx)
    let deposition_affinity: Vec<f64> = sx_flat.iter().map(|&s| (-s).max(0.0)).collect();

    // Distribute eroded mass to neighbors
    let mut deposition = vec![0.0; n_units];

    for i in 0..n_units {
        if erosion[i] <= 0.0 {
            continue;
        }

        let eroded_mass_i = erosion[i] * area_flat[i];

        // Compute weighted sum of neighbor affinities
        let mut weights_sum = 0.0;
        for j_idx in 0..max_neighbors {
            let j = neighbor_indices[[i, j_idx]];
            if j < 0 {
                continue;
            }
            let w = neighbor_weights[[i, j_idx]];
            let affinity_j = deposition_affinity[j as usize];
            weights_sum += w * affinity_j;
        }

        if weights_sum <= 0.0 {
            // No sheltered neighbors - distribute evenly by weights
            for j_idx in 0..max_neighbors {
                let j = neighbor_indices[[i, j_idx]];
                if j < 0 {
                    continue;
                }
                let w = neighbor_weights[[i, j_idx]];
                let deposit = (eroded_mass_i * w) / area_flat[j as usize];
                deposition[j as usize] += deposit;
            }
        } else {
            // Distribute by affinity-weighted neighbor weights
            for j_idx in 0..max_neighbors {
                let j = neighbor_indices[[i, j_idx]];
                if j < 0 {
                    continue;
                }
                let w = neighbor_weights[[i, j_idx]];
                let affinity_j = deposition_affinity[j as usize];
                let fraction = (w * affinity_j) / weights_sum;
                let deposit = (eroded_mass_i * fraction) / area_flat[j as usize];
                deposition[j as usize] += deposit;
            }
        }
    }

    // Add deposition
    for i in 0..n_units {
        swe_new[i] += deposition[i];
    }

    let out = ndarray::ArrayD::from_shape_vec(swe.shape(), swe_new).unwrap();
    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

// ---------------------------------------------------------------------------
// Array wrapper: avalanche_redistribute
// ---------------------------------------------------------------------------

/// Compute gravitational (avalanche) snow redistribution.
///
/// Moves SWE from slopes exceeding threshold to downslope neighbors via
/// D-infinity flow routing. Mass-conservative: sum(swe * area) preserved.
///
/// SYNC: python_kernels/redistribution.py::avalanche_redistribute
/// SYNC: jax_kernels/redistribution.py::avalanche_redistribute
#[pyfunction]
fn avalanche_redistribute<'py>(
    py: Python<'py>,
    swe: PyReadonlyArrayDyn<'py, f64>,
    slope: PyReadonlyArrayDyn<'py, f64>,
    slope_threshold: f64,
    flow_fractions: PyReadonlyArrayDyn<'py, f64>,
    neighbor_indices: PyReadonlyArrayDyn<'py, i64>,
    area: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let swe = swe.as_array();
    let slope = slope.as_array();
    let area = area.as_array();
    let flow_fractions = flow_fractions.as_array();
    let neighbor_indices = neighbor_indices.as_array();

    let swe_flat = swe.as_slice().expect("contiguous array");
    let slope_flat = slope.as_slice().expect("contiguous array");
    let area_flat = area.as_slice().expect("contiguous array");

    let n_units = swe_flat.len();
    let max_neighbors = neighbor_indices.shape()[1];

    let mut swe_new = swe_flat.to_vec();

    // Compute redistribute fraction based on slope excess
    // fraction = min(1, (slope - threshold) / threshold)
    for i in 0..n_units {
        if slope_flat[i] <= slope_threshold {
            continue;
        }

        let slope_excess = (slope_flat[i] - slope_threshold).max(0.0);
        let redistribute_fraction = (slope_excess / slope_threshold).min(1.0);

        let swe_to_move = swe_new[i] * redistribute_fraction;
        if swe_to_move <= 0.0 {
            continue;
        }

        let mass_to_move = swe_to_move * area_flat[i];

        // Remove from source
        swe_new[i] -= swe_to_move;

        // Distribute to downslope neighbors
        let mut total_flow = 0.0;
        for j_idx in 0..max_neighbors {
            let j = neighbor_indices[[i, j_idx]];
            if j < 0 {
                continue;
            }
            let frac = flow_fractions[[i, j_idx]];
            if frac <= 0.0 {
                continue;
            }
            total_flow += frac;
            let deposit_mass = mass_to_move * frac;
            swe_new[j as usize] += deposit_mass / area_flat[j as usize];
        }

        // If no outflow neighbors, snow stays in place
        if total_flow <= 0.0 {
            swe_new[i] += swe_to_move;
        }
    }

    let out = ndarray::ArrayD::from_shape_vec(swe.shape(), swe_new).unwrap();
    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_transport_rate, m)?)?;
    m.add_function(wrap_pyfunction!(wind_transport, m)?)?;
    m.add_function(wrap_pyfunction!(avalanche_redistribute, m)?)?;
    Ok(())
}
