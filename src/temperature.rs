// SYNC WARNING: This file must stay in sync with:
//   - topopyscale2/core/python_kernels/temperature.py
//   - topopyscale2/core/jax_kernels/temperature.py
// Any changes here MUST be applied to all three backends in the same commit.

use ndarray::Zip;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// Scalar pure functions
// ---------------------------------------------------------------------------

/// Apply simple lapse-rate correction from surface temperature.
///
/// T_target = T_surface - lapse_rate * (z_target - z_surface)
///
/// This is the "simple" downscaling mode that only requires ERA5-Land
/// surface data (t2m). Temperature decreases with elevation at a fixed rate.
#[inline]
fn simple_lapse_rate_scalar(t_surface: f64, z_surface: f64, z_target: f64, lapse_rate: f64) -> f64 {
    let dz = z_target - z_surface;
    t_surface - lapse_rate * dz
}

/// Apply lapse-rate correction to a single temperature value.
///
/// t_corrected = t_source + gamma * (z_unit - z_source)
#[inline]
fn lapse_rate_correction_scalar(t_source: f64, gamma: f64, z_unit: f64, z_source: f64) -> f64 {
    t_source + gamma * (z_unit - z_source)
}

// ---------------------------------------------------------------------------
// Array wrappers exposed to Python via PyO3
// ---------------------------------------------------------------------------

/// Apply simple lapse-rate correction from surface temperature.
///
/// SYNC: python_kernels/temperature.py::simple_lapse_rate
/// SYNC: jax_kernels/temperature.py::simple_lapse_rate
///
/// Parameters
/// ----------
/// t_surface : ndarray
///     Surface temperature [K], e.g., ERA5 t2m.
/// z_surface : ndarray
///     Surface elevation [m], e.g., ERA5 grid cell elevation.
/// z_target : f64
///     Target elevation [m] (scalar).
/// lapse_rate : f64
///     Environmental lapse rate [K/m], default 0.0065 (6.5 K/km).
///
/// Returns
/// -------
/// ndarray
///     Temperature at target elevation [K].
#[pyfunction]
#[pyo3(signature = (t_surface, z_surface, z_target, lapse_rate=0.0065))]
fn simple_lapse_rate<'py>(
    py: Python<'py>,
    t_surface: PyReadonlyArrayDyn<'py, f64>,
    z_surface: PyReadonlyArrayDyn<'py, f64>,
    z_target: f64,
    lapse_rate: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let t_surface = t_surface.as_array();
    let z_surface = z_surface.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(t_surface.shape());

    Zip::from(&mut out)
        .and(&t_surface)
        .and(&z_surface)
        .for_each(|o, &ts, &zs| {
            *o = simple_lapse_rate_scalar(ts, zs, z_target, lapse_rate);
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

/// Apply lapse-rate correction to temperature arrays.
///
/// SYNC: python_kernels/temperature.py::lapse_rate_correction
/// SYNC: jax_kernels/temperature.py::lapse_rate_correction
///
/// Parameters
/// ----------
/// t_source : ndarray
///     Source temperature [K].
/// gamma : ndarray
///     Lapse rate [K/m] (typically negative, e.g., -0.0065).
/// z_unit : ndarray
///     Target elevation [m].
/// z_source : ndarray
///     Source elevation [m].
///
/// Returns
/// -------
/// ndarray
///     Corrected temperature [K].
#[pyfunction]
fn lapse_rate_correction<'py>(
    py: Python<'py>,
    t_source: PyReadonlyArrayDyn<'py, f64>,
    gamma: PyReadonlyArrayDyn<'py, f64>,
    z_unit: PyReadonlyArrayDyn<'py, f64>,
    z_source: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let t_source = t_source.as_array();
    let gamma = gamma.as_array();
    let z_unit = z_unit.as_array();
    let z_source = z_source.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(t_source.shape());

    Zip::from(&mut out)
        .and(&t_source)
        .and(&gamma)
        .and(&z_unit)
        .and(&z_source)
        .for_each(|o, &ts, &g, &zu, &zs| {
            *o = lapse_rate_correction_scalar(ts, g, zu, zs);
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simple_lapse_rate, m)?)?;
    m.add_function(wrap_pyfunction!(lapse_rate_correction, m)?)?;
    Ok(())
}
