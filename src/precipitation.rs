// SYNC WARNING: This file must stay in sync with:
//   - topopyscale2/core/python_kernels/precipitation.py
//   - topopyscale2/core/jax_kernels/precipitation.py
// Any changes here MUST be applied to all three backends in the same commit.

use ndarray::Zip;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

// Thermodynamic constants for the psychrometric wet-bulb solve.
const CP_AIR: f64 = 1004.0; // specific heat of air at constant pressure [J/kg/K]
const EPSILON: f64 = 0.622; // ratio of molar masses of water vapour to dry air
const LV: f64 = 2.501e6; // latent heat of vaporisation [J/kg]
const WET_BULB_ITERS: usize = 20; // fixed Newton iterations (must match across backends)

// ===========================================================================
// Scalar pure functions
// ===========================================================================

/// Elevation-gradient correction for precipitation.
///
/// p_corrected = p_source * (1 + gradient * dz), clamped >= 0.
#[inline]
fn elevation_gradient_scalar(p_source: f64, z_unit: f64, z_source: f64, gradient: f64) -> f64 {
    let dz = z_unit - z_source;
    let p_corrected = p_source * (1.0 + gradient * dz);
    p_corrected.max(0.0)
}

/// Psychrometric (pressure-aware) wet-bulb temperature [K].
///
/// Solves e = e_s(Tw) - gamma * (T - Tw) for Tw with fixed Newton iterations.
fn wet_bulb_temperature_scalar(t: f64, p: f64, q: f64) -> f64 {
    // Actual vapour pressure from specific humidity [Pa].
    let e = q * p / (EPSILON + (1.0 - EPSILON) * q);
    // Psychrometric "constant" gamma [Pa/K] (pressure-dependent).
    let gamma = (CP_AIR * p) / (EPSILON * LV);

    let mut tw = t; // initial guess: air temperature
    for _ in 0..WET_BULB_ITERS {
        let es = 611.2 * ((17.67 * (tw - 273.15)) / (tw - 29.65)).exp();
        let des_dtw = es * (17.67 * (273.15 - 29.65)) / ((tw - 29.65) * (tw - 29.65));
        let f = es - gamma * (t - tw) - e;
        let fprime = des_dtw + gamma;
        tw -= f / fprime;
    }

    tw.min(t)
}

/// Phase partition of total precipitation into rainfall and snowfall.
///
/// Returns (rainfall, snowfall).
#[inline]
fn phase_partition_scalar(
    p_total: f64,
    temperature: f64,
    t_rain: f64,
    t_snow: f64,
) -> (f64, f64) {
    let rain_fraction = if temperature >= t_rain {
        1.0
    } else if temperature <= t_snow {
        0.0
    } else {
        (temperature - t_snow) / (t_rain - t_snow)
    };

    let rainfall = p_total * rain_fraction;
    let snowfall = p_total * (1.0 - rain_fraction);
    (rainfall, snowfall)
}

// ===========================================================================
// Array wrappers exposed to Python via PyO3
// ===========================================================================

/// Apply elevation-gradient correction to precipitation.
///
/// SYNC: python_kernels/precipitation.py::elevation_gradient
/// SYNC: jax_kernels/precipitation.py::elevation_gradient
///
/// Parameters
/// ----------
/// p_source : ndarray
///     Source precipitation [kg m^-2 s^-1 or mm].
/// z_unit : ndarray
///     Target elevation [m].
/// z_source : ndarray
///     Source elevation [m].
/// gradient : ndarray
///     Precipitation gradient [1/m].
///
/// Returns
/// -------
/// ndarray
///     Corrected precipitation (>= 0).
#[pyfunction]
fn elevation_gradient<'py>(
    py: Python<'py>,
    p_source: PyReadonlyArrayDyn<'py, f64>,
    z_unit: PyReadonlyArrayDyn<'py, f64>,
    z_source: PyReadonlyArrayDyn<'py, f64>,
    gradient: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let p_source = p_source.as_array();
    let z_unit = z_unit.as_array();
    let z_source = z_source.as_array();
    let gradient = gradient.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(p_source.shape());

    Zip::from(&mut out)
        .and(&p_source)
        .and(&z_unit)
        .and(&z_source)
        .and(&gradient)
        .for_each(|o, &ps, &zu, &zs, &g| {
            *o = elevation_gradient_scalar(ps, zu, zs, g);
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

/// Psychrometric (pressure-aware) wet-bulb temperature.
///
/// SYNC: python_kernels/precipitation.py::wet_bulb_temperature
/// SYNC: jax_kernels/precipitation.py::wet_bulb_temperature
///
/// Parameters
/// ----------
/// temperature : ndarray
///     Air (dry-bulb) temperature [K].
/// pressure : ndarray
///     Air pressure [Pa].
/// specific_humidity : ndarray
///     Specific humidity [kg/kg].
///
/// Returns
/// -------
/// ndarray
///     Wet-bulb temperature [K], clamped <= air temperature.
#[pyfunction]
fn wet_bulb_temperature<'py>(
    py: Python<'py>,
    temperature: PyReadonlyArrayDyn<'py, f64>,
    pressure: PyReadonlyArrayDyn<'py, f64>,
    specific_humidity: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let temperature = temperature.as_array();
    let pressure = pressure.as_array();
    let specific_humidity = specific_humidity.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(temperature.shape());

    Zip::from(&mut out)
        .and(&temperature)
        .and(&pressure)
        .and(&specific_humidity)
        .for_each(|o, &t, &p, &q| {
            *o = wet_bulb_temperature_scalar(t, p, q);
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

/// Partition total precipitation into rainfall and snowfall.
///
/// SYNC: python_kernels/precipitation.py::phase_partition
/// SYNC: jax_kernels/precipitation.py::phase_partition
///
/// Parameters
/// ----------
/// p_total : ndarray
///     Total precipitation [kg m^-2 s^-1 or mm].
/// temperature : ndarray
///     Air temperature [K or degC — must be consistent with t_rain/t_snow].
/// t_rain : ndarray
///     Temperature above which all precipitation is rain.
/// t_snow : ndarray
///     Temperature below which all precipitation is snow.
///
/// Returns
/// -------
/// (rainfall, snowfall) : tuple of ndarray
#[pyfunction]
fn phase_partition<'py>(
    py: Python<'py>,
    p_total: PyReadonlyArrayDyn<'py, f64>,
    temperature: PyReadonlyArrayDyn<'py, f64>,
    t_rain: PyReadonlyArrayDyn<'py, f64>,
    t_snow: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let p_total = p_total.as_array();
    let temperature = temperature.as_array();
    let t_rain = t_rain.as_array();
    let t_snow = t_snow.as_array();

    let mut rainfall = ndarray::ArrayD::<f64>::zeros(p_total.shape());
    let mut snowfall = ndarray::ArrayD::<f64>::zeros(p_total.shape());

    Zip::from(&mut rainfall)
        .and(&mut snowfall)
        .and(&p_total)
        .and(&temperature)
        .and(&t_rain)
        .and(&t_snow)
        .for_each(|r, s, &pt, &temp, &tr, &ts| {
            let (rain, snow) = phase_partition_scalar(pt, temp, tr, ts);
            *r = rain;
            *s = snow;
        });

    Ok((
        PyArrayDyn::from_owned_array_bound(py, rainfall),
        PyArrayDyn::from_owned_array_bound(py, snowfall),
    ))
}

// ===========================================================================
// Module registration
// ===========================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(elevation_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(phase_partition, m)?)?;
    m.add_function(wrap_pyfunction!(wet_bulb_temperature, m)?)?;
    Ok(())
}
