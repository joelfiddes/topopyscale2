// SYNC WARNING: This file must stay in sync with:
//   - topopyscale2/core/python_kernels/humidity.py
//   - topopyscale2/core/jax_kernels/humidity.py
// Any changes here MUST be applied to all three backends in the same commit.

use ndarray::Zip;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// ===========================================================================
// Scalar pure function
// ===========================================================================

/// Adjust humidity for downscaled temperature and pressure.
///
/// Specific humidity is conserved; relative humidity is recomputed from the
/// Clausius-Clapeyron-derived saturation vapour pressure (Tetens formula).
///
/// Returns (q_adjusted, rh).
#[inline]
fn adjust_humidity_scalar(q_source: f64, t_unit: f64, p_unit: f64) -> (f64, f64) {
    let q_adjusted = q_source;

    // Mixing ratio from specific humidity
    let mr = q_source / (1.0 - q_source);

    // Actual vapour pressure [Pa]
    let e = mr * p_unit / (0.62197 + mr);

    // Saturation vapour pressure [Pa] — Tetens / August-Roche-Magnus
    let e_sat = 611.2 * ((17.67 * (t_unit - 273.15)) / (t_unit - 29.65)).exp();

    // Relative humidity, clamped to [0, 1]
    let rh = (e / e_sat).clamp(0.0, 1.0);

    (q_adjusted, rh)
}

// ===========================================================================
// Array wrapper exposed to Python via PyO3
// ===========================================================================

/// Adjust humidity for downscaled conditions.
///
/// SYNC: python_kernels/humidity.py::adjust_humidity
/// SYNC: jax_kernels/humidity.py::adjust_humidity
///
/// Parameters
/// ----------
/// q_source : ndarray
///     Specific humidity [kg/kg].
/// t_unit : ndarray
///     Downscaled air temperature [K].
/// p_unit : ndarray
///     Surface pressure at target [Pa].
///
/// Returns
/// -------
/// (q_adjusted, rh) : tuple of ndarray
///     q_adjusted is the conserved specific humidity; rh is relative humidity [0-1].
#[pyfunction]
fn adjust_humidity<'py>(
    py: Python<'py>,
    q_source: PyReadonlyArrayDyn<'py, f64>,
    t_unit: PyReadonlyArrayDyn<'py, f64>,
    p_unit: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let q_source = q_source.as_array();
    let t_unit = t_unit.as_array();
    let p_unit = p_unit.as_array();

    let mut q_out = ndarray::ArrayD::<f64>::zeros(q_source.shape());
    let mut rh_out = ndarray::ArrayD::<f64>::zeros(q_source.shape());

    Zip::from(&mut q_out)
        .and(&mut rh_out)
        .and(&q_source)
        .and(&t_unit)
        .and(&p_unit)
        .for_each(|q, rh, &qs, &tu, &pu| {
            let (qa, r) = adjust_humidity_scalar(qs, tu, pu);
            *q = qa;
            *rh = r;
        });

    Ok((
        PyArrayDyn::from_owned_array_bound(py, q_out),
        PyArrayDyn::from_owned_array_bound(py, rh_out),
    ))
}

// ===========================================================================
// Module registration
// ===========================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(adjust_humidity, m)?)?;
    Ok(())
}
