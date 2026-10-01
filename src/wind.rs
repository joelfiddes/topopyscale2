// SYNC WARNING: This file must stay in sync with:
//   - topopyscale2/core/python_kernels/wind.py
//   - topopyscale2/core/jax_kernels/wind.py
// Any changes here MUST be applied to all three backends in the same commit.

use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Scalar pure function
// ---------------------------------------------------------------------------

/// Apply log-profile correction to a single wind speed value.
///
/// Formula:
///     U_target = U_source * ln((z_target - d_target) / z0_target) / ln((z_source - d_source) / z0_source)
///
/// Returns 0.0 when effective height (z - d) is at or below z0.
#[inline]
fn log_profile_correction_scalar(
    u_source: f64,
    z_source: f64,
    z_target: f64,
    z0_source: f64,
    z0_target: f64,
    d_source: f64,
    d_target: f64,
) -> f64 {
    let h_source = z_source - d_source;
    let h_target = z_target - d_target;

    // Guard: return 0 when effective height <= roughness length
    if h_source <= z0_source || h_target <= z0_target {
        return 0.0;
    }

    let ln_source = (h_source / z0_source).ln();
    let ln_target = (h_target / z0_target).ln();

    u_source * ln_target / ln_source
}

/// Apply Winstral Sx-based wind exposure/sheltering correction (scalar).
///
/// Formula:
///     U_corrected = U_log * (1 + sx_scale * tanh(Sx / sx_ref))
///
/// Positive Sx = exposed (enhance wind), negative = sheltered (reduce wind).
#[inline]
fn winstral_wind_correction_scalar(u_log: f64, sx: f64, sx_scale: f64, sx_ref: f64) -> f64 {
    let correction_factor = 1.0 + sx_scale * (sx / sx_ref).tanh();
    u_log * correction_factor
}

// ---------------------------------------------------------------------------
// Array wrapper exposed to Python via PyO3
// ---------------------------------------------------------------------------

/// Apply logarithmic wind profile correction to wind speed arrays.
///
/// SYNC: python_kernels/wind.py::log_profile_correction
/// SYNC: jax_kernels/wind.py::log_profile_correction
///
/// Parameters
/// ----------
/// u_source : ndarray
///     Source wind speed [m/s].
/// z_source : ndarray
///     Source measurement height above ground [m].
/// z_target : ndarray
///     Target height above ground [m].
/// z0_source : ndarray
///     Roughness length at source [m].
/// z0_target : ndarray
///     Roughness length at target [m].
/// d_source : ndarray
///     Zero-plane displacement height at source [m].
/// d_target : ndarray
///     Zero-plane displacement height at target [m].
///
/// Returns
/// -------
/// ndarray
///     Corrected wind speed at target height [m/s].
#[pyfunction]
fn log_profile_correction<'py>(
    py: Python<'py>,
    u_source: PyReadonlyArrayDyn<'py, f64>,
    z_source: PyReadonlyArrayDyn<'py, f64>,
    z_target: PyReadonlyArrayDyn<'py, f64>,
    z0_source: PyReadonlyArrayDyn<'py, f64>,
    z0_target: PyReadonlyArrayDyn<'py, f64>,
    d_source: PyReadonlyArrayDyn<'py, f64>,
    d_target: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let u_source = u_source.as_array();
    let z_source = z_source.as_array();
    let z_target = z_target.as_array();
    let z0_source = z0_source.as_array();
    let z0_target = z0_target.as_array();
    let d_source = d_source.as_array();
    let d_target = d_target.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(u_source.shape());

    // ndarray Zip only supports up to 6 arrays. Since we have 7 inputs + 1 output,
    // we iterate over flat slices and use scalar function directly.
    let u_flat = u_source.as_slice().expect("contiguous array");
    let zs_flat = z_source.as_slice().expect("contiguous array");
    let zt_flat = z_target.as_slice().expect("contiguous array");
    let z0s_flat = z0_source.as_slice().expect("contiguous array");
    let z0t_flat = z0_target.as_slice().expect("contiguous array");
    let ds_flat = d_source.as_slice().expect("contiguous array");
    let dt_flat = d_target.as_slice().expect("contiguous array");
    let out_flat = out.as_slice_mut().expect("contiguous array");

    for i in 0..out_flat.len() {
        out_flat[i] = log_profile_correction_scalar(
            u_flat[i], zs_flat[i], zt_flat[i], z0s_flat[i], z0t_flat[i], ds_flat[i], dt_flat[i]
        );
    }

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

/// Apply Winstral Sx-based wind exposure/sheltering correction.
///
/// Modifies wind speed based on terrain exposure index (Sx). Positive Sx
/// indicates exposed terrain (wind enhancement), negative Sx indicates
/// sheltered terrain (wind reduction).
///
/// SYNC: python_kernels/wind.py::winstral_wind_correction
/// SYNC: jax_kernels/wind.py::winstral_wind_correction
///
/// Parameters
/// ----------
/// u_log : ndarray
///     Wind speed after log-profile correction [m/s].
/// sx : ndarray
///     Winstral Sx terrain parameter [degrees]. Positive = exposed,
///     negative = sheltered.
/// sx_scale : float, optional
///     Scaling factor for the Sx effect. Default is 0.5.
/// sx_ref : float, optional
///     Reference Sx value for normalization [degrees]. Default is 15.0.
///
/// Returns
/// -------
/// ndarray
///     Corrected wind speed [m/s].
#[pyfunction]
#[pyo3(signature = (u_log, sx, sx_scale=0.5, sx_ref=15.0))]
fn winstral_wind_correction<'py>(
    py: Python<'py>,
    u_log: PyReadonlyArrayDyn<'py, f64>,
    sx: PyReadonlyArrayDyn<'py, f64>,
    sx_scale: f64,
    sx_ref: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let u_log = u_log.as_array();
    let sx = sx.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(u_log.shape());

    let u_flat = u_log.as_slice().expect("contiguous array");
    let sx_flat = sx.as_slice().expect("contiguous array");
    let out_flat = out.as_slice_mut().expect("contiguous array");

    for i in 0..out_flat.len() {
        out_flat[i] = winstral_wind_correction_scalar(u_flat[i], sx_flat[i], sx_scale, sx_ref);
    }

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(log_profile_correction, m)?)?;
    m.add_function(wrap_pyfunction!(winstral_wind_correction, m)?)?;
    Ok(())
}
