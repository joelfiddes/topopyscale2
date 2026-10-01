//! Pressure-level interpolation kernels — Rust implementation.
//!
//! SYNC WARNING: This file must stay in sync with:
//!   - topopyscale2/core/python_kernels/interpolation.py
//!   - topopyscale2/core/jax_kernels/interpolation.py
//! Any changes here MUST be applied to all three backends in the same commit.

use ndarray::{Array1, ArrayView1, ArrayView2};
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// Interpolate a single timestep from pressure levels to target elevation.
///
/// # Arguments
/// * `values` - Values at pressure levels
/// * `z_levels` - Geopotential heights at pressure levels [m]
/// * `z_target` - Target elevation [m]
///
/// # Returns
/// Interpolated value at target elevation
fn interpolate_single(values: ArrayView1<f64>, z_levels: ArrayView1<f64>, z_target: f64) -> f64 {
    let n_levels = values.len();

    // Find levels above and below target
    let mut i_top: Option<usize> = None;
    let mut i_bot: Option<usize> = None;
    let mut min_dist_above = f64::INFINITY;
    let mut min_dist_below = f64::INFINITY;

    for i in 0..n_levels {
        let z = z_levels[i];
        if z > z_target {
            let dist = z - z_target;
            if dist < min_dist_above {
                min_dist_above = dist;
                i_top = Some(i);
            }
        } else if z < z_target {
            let dist = z_target - z;
            if dist < min_dist_below {
                min_dist_below = dist;
                i_bot = Some(i);
            }
        }
    }

    match (i_top, i_bot) {
        (Some(top), Some(bot)) => {
            // Bracketed: interpolate
            let z_top = z_levels[top];
            let z_bot = z_levels[bot];
            let v_top = values[top];
            let v_bot = values[bot];

            let d_top = z_top - z_target;
            let d_bot = z_target - z_bot;
            let total_dist = d_top + d_bot;

            if total_dist > 0.0 {
                let w_top = d_bot / total_dist;
                let w_bot = d_top / total_dist;
                w_bot * v_bot + w_top * v_top
            } else {
                v_top
            }
        }
        (Some(_), None) => {
            // Below all levels: use lowest
            let i_lowest = z_levels
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            values[i_lowest]
        }
        (None, Some(_)) => {
            // Above all levels: use highest
            let i_highest = z_levels
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            values[i_highest]
        }
        (None, None) => {
            // All levels equal target (unlikely)
            values[0]
        }
    }
}

/// Interpolate values from pressure levels to target elevation.
///
/// # Arguments
/// * `values` - Values at pressure levels, shape (n_time, n_levels)
/// * `z_levels` - Geopotential heights at pressure levels [m], shape (n_time, n_levels)
/// * `z_target` - Target elevation [m]
///
/// # Returns
/// Interpolated values at target elevation, shape (n_time,)
pub fn interpolate_pressure_levels(
    values: ArrayView2<f64>,
    z_levels: ArrayView2<f64>,
    z_target: f64,
) -> Array1<f64> {
    let n_time = values.nrows();
    let mut result = Array1::zeros(n_time);

    for t in 0..n_time {
        result[t] = interpolate_single(values.row(t), z_levels.row(t), z_target);
    }

    result
}

/// Interpolate u and v wind components from pressure levels.
pub fn interpolate_wind_components(
    u_levels: ArrayView2<f64>,
    v_levels: ArrayView2<f64>,
    z_levels: ArrayView2<f64>,
    z_target: f64,
) -> (Array1<f64>, Array1<f64>) {
    let u_target = interpolate_pressure_levels(u_levels, z_levels, z_target);
    let v_target = interpolate_pressure_levels(v_levels, z_levels, z_target);
    (u_target, v_target)
}

/// Compute wind speed and direction from u, v components.
pub fn compute_wind_speed_direction(u: ArrayView1<f64>, v: ArrayView1<f64>) -> (Array1<f64>, Array1<f64>) {
    let n = u.len();
    let mut speed = Array1::zeros(n);
    let mut direction = Array1::zeros(n);

    for i in 0..n {
        speed[i] = (u[i].powi(2) + v[i].powi(2)).sqrt();
        // Meteorological convention: direction FROM which wind blows
        let dir_rad = (-u[i]).atan2(-v[i]);
        direction[i] = dir_rad.to_degrees().rem_euclid(360.0);
    }

    (speed, direction)
}

/// Interpolate specific humidity and compute relative humidity at target.
pub fn interpolate_humidity(
    q_levels: ArrayView2<f64>,
    z_levels: ArrayView2<f64>,
    z_target: f64,
    t_target: ArrayView1<f64>,
    p_target: ArrayView1<f64>,
) -> (Array1<f64>, Array1<f64>) {
    let q_target = interpolate_pressure_levels(q_levels, z_levels, z_target);

    let n = t_target.len();
    let mut rh_target = Array1::zeros(n);

    for i in 0..n {
        let q = q_target[i];
        let t = t_target[i];
        let p = p_target[i];

        // Mixing ratio from specific humidity
        let mr = q / (1.0 - q);

        // Actual vapor pressure [Pa]
        let e = mr * p / (0.62197 + mr);

        // Bolton (1980) saturation vapor pressure [Pa]
        let e_sat = 611.2 * (17.67 * (t - 273.15) / (t - 29.65)).exp();

        // Relative humidity clamped to [0, 1]
        rh_target[i] = (e / e_sat).clamp(0.0, 1.0);
    }

    (q_target, rh_target)
}

// ============================================================================
// PyO3 Bindings
// ============================================================================

#[pyfunction]
#[pyo3(name = "interpolate_pressure_levels")]
pub fn py_interpolate_pressure_levels(
    py: Python<'_>,
    values: PyReadonlyArrayDyn<f64>,
    z_levels: PyReadonlyArrayDyn<f64>,
    z_target: f64,
) -> PyResult<Py<PyArrayDyn<f64>>> {
    let values_view = values.as_array();
    let z_levels_view = z_levels.as_array();

    // Get shape before consuming views
    let ndim = values_view.ndim();
    let shape_vec: Vec<usize> = values_view.shape().to_vec();

    let (values_2d, z_2d) = if ndim == 1 {
        let n = shape_vec[0];
        (
            values_view.into_shape_with_order((1, n)).unwrap(),
            z_levels_view.into_shape_with_order((1, n)).unwrap(),
        )
    } else {
        (
            values_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
            z_levels_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
        )
    };

    let result = interpolate_pressure_levels(values_2d, z_2d, z_target);
    let result_dyn = result.into_dyn();
    Ok(PyArrayDyn::from_owned_array_bound(py, result_dyn).into())
}

#[pyfunction]
#[pyo3(name = "interpolate_wind_components")]
pub fn py_interpolate_wind_components(
    py: Python<'_>,
    u_levels: PyReadonlyArrayDyn<f64>,
    v_levels: PyReadonlyArrayDyn<f64>,
    z_levels: PyReadonlyArrayDyn<f64>,
    z_target: f64,
) -> PyResult<(Py<PyArrayDyn<f64>>, Py<PyArrayDyn<f64>>)> {
    let u_view = u_levels.as_array();
    let v_view = v_levels.as_array();
    let z_view = z_levels.as_array();

    // Get shape before consuming views
    let ndim = u_view.ndim();
    let shape_vec: Vec<usize> = u_view.shape().to_vec();

    let (u_2d, v_2d, z_2d) = if ndim == 1 {
        let n = shape_vec[0];
        (
            u_view.into_shape_with_order((1, n)).unwrap(),
            v_view.into_shape_with_order((1, n)).unwrap(),
            z_view.into_shape_with_order((1, n)).unwrap(),
        )
    } else {
        (
            u_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
            v_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
            z_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
        )
    };

    let (u_target, v_target) = interpolate_wind_components(u_2d, v_2d, z_2d, z_target);

    Ok((
        PyArrayDyn::from_owned_array_bound(py, u_target.into_dyn()).into(),
        PyArrayDyn::from_owned_array_bound(py, v_target.into_dyn()).into(),
    ))
}

#[pyfunction]
#[pyo3(name = "compute_wind_speed_direction")]
pub fn py_compute_wind_speed_direction(
    py: Python<'_>,
    u: PyReadonlyArrayDyn<f64>,
    v: PyReadonlyArrayDyn<f64>,
) -> PyResult<(Py<PyArrayDyn<f64>>, Py<PyArrayDyn<f64>>)> {
    let u_view = u.as_array();
    let v_view = v.as_array();

    // Flatten to 1D
    let u_flat = u_view.iter().cloned().collect::<Vec<_>>();
    let v_flat = v_view.iter().cloned().collect::<Vec<_>>();

    let u_arr = Array1::from(u_flat);
    let v_arr = Array1::from(v_flat);

    let (speed, direction) = compute_wind_speed_direction(u_arr.view(), v_arr.view());

    Ok((
        PyArrayDyn::from_owned_array_bound(py, speed.into_dyn()).into(),
        PyArrayDyn::from_owned_array_bound(py, direction.into_dyn()).into(),
    ))
}

#[pyfunction]
#[pyo3(name = "interpolate_humidity")]
pub fn py_interpolate_humidity(
    py: Python<'_>,
    q_levels: PyReadonlyArrayDyn<f64>,
    z_levels: PyReadonlyArrayDyn<f64>,
    z_target: f64,
    t_target: PyReadonlyArrayDyn<f64>,
    p_target: PyReadonlyArrayDyn<f64>,
) -> PyResult<(Py<PyArrayDyn<f64>>, Py<PyArrayDyn<f64>>)> {
    let q_view = q_levels.as_array();
    let z_view = z_levels.as_array();
    let t_view = t_target.as_array();
    let p_view = p_target.as_array();

    // Get shape before consuming views
    let ndim = q_view.ndim();
    let shape_vec: Vec<usize> = q_view.shape().to_vec();

    let (q_2d, z_2d) = if ndim == 1 {
        let n = shape_vec[0];
        (
            q_view.into_shape_with_order((1, n)).unwrap(),
            z_view.into_shape_with_order((1, n)).unwrap(),
        )
    } else {
        (
            q_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
            z_view.into_shape_with_order((shape_vec[0], shape_vec[1])).unwrap(),
        )
    };

    // Flatten target arrays
    let t_flat: Vec<f64> = t_view.iter().cloned().collect();
    let p_flat: Vec<f64> = p_view.iter().cloned().collect();

    let t_arr = Array1::from(t_flat);
    let p_arr = Array1::from(p_flat);

    let (q_target, rh_target) = interpolate_humidity(q_2d, z_2d, z_target, t_arr.view(), p_arr.view());

    Ok((
        PyArrayDyn::from_owned_array_bound(py, q_target.into_dyn()).into(),
        PyArrayDyn::from_owned_array_bound(py, rh_target.into_dyn()).into(),
    ))
}

/// Register interpolation functions in the module.
pub fn register_interpolation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_interpolate_pressure_levels, m)?)?;
    m.add_function(wrap_pyfunction!(py_interpolate_wind_components, m)?)?;
    m.add_function(wrap_pyfunction!(py_compute_wind_speed_direction, m)?)?;
    m.add_function(wrap_pyfunction!(py_interpolate_humidity, m)?)?;
    Ok(())
}
