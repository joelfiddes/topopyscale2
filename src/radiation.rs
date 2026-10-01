// SYNC WARNING: This file must stay in sync with:
//   - topopyscale2/core/python_kernels/radiation.py
//   - topopyscale2/core/jax_kernels/radiation.py
// Any changes here MUST be applied to all three backends in the same commit.

use ndarray::Zip;
use numpy::{PyArrayDyn, PyReadonlyArrayDyn};
use pyo3::prelude::*;

/// Stefan-Boltzmann constant [W m^-2 K^-4].
const SIGMA: f64 = 5.670374419e-8;

/// Degrees-to-radians conversion factor.
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Minimum sin(solar_elevation) for slope correction denominator.
/// sin(15°) ≈ 0.259; below this the cos_incidence/sin_elev ratio becomes
/// numerically unstable on steep slopes. See TopoPyScale PR #132 (Beria).
const MU0_MIN: f64 = 0.259;

/// Solar constant [W/m2] — absolute upper bound for direct beam on any surface.
const SOLAR_CONSTANT: f64 = 1361.0;

// ===========================================================================
// Scalar pure functions
// ===========================================================================

/// Erbs et al. (1982) diffuse-fraction model.
///
/// Returns (sw_direct, sw_diffuse).
#[inline]
fn partition_shortwave_scalar(
    sw_total: f64,
    solar_elevation: f64,
    clearness_index: f64,
) -> (f64, f64) {
    if solar_elevation <= 0.0 {
        return (0.0, sw_total);
    }

    let kt = clearness_index.clamp(0.0, 1.0);

    let kd = if kt <= 0.22 {
        1.0 - 0.09 * kt
    } else if kt <= 0.80 {
        let kt2 = kt * kt;
        let kt3 = kt2 * kt;
        let kt4 = kt3 * kt;
        0.9511 - 0.1604 * kt + 4.388 * kt2 - 16.638 * kt3 + 12.336 * kt4
    } else {
        0.165
    };

    let diffuse = kd * sw_total;
    let direct = sw_total - diffuse;
    (direct, diffuse)
}

/// Slope correction for direct shortwave radiation.
///
/// All angles in DEGREES; converted to radians internally.
#[inline]
fn slope_correction_scalar(
    sw_direct: f64,
    solar_elevation: f64,
    solar_azimuth: f64,
    slope: f64,
    aspect: f64,
) -> f64 {
    let elev_rad = solar_elevation * DEG2RAD;
    let slope_rad = slope * DEG2RAD;
    let azimuth_diff_rad = (solar_azimuth - aspect) * DEG2RAD;

    let sin_elev = elev_rad.sin();
    let cos_elev = elev_rad.cos();
    let cos_slope = slope_rad.cos();
    let sin_slope = slope_rad.sin();

    let cos_incidence =
        (sin_elev * cos_slope + cos_elev * sin_slope * azimuth_diff_rad.cos()).max(0.0);

    // Clamp sin(solar_elevation) to prevent numerical instability when
    // sun is near horizon. See MU0_MIN constant and TopoPyScale PR #132.
    let sin_stable = sin_elev.max(MU0_MIN);
    let correction = if sin_elev > 0.01 {
        cos_incidence / sin_stable
    } else {
        0.0
    };

    (sw_direct * correction).min(SOLAR_CONSTANT)
}

/// Sky-view-factor correction for diffuse shortwave radiation.
#[inline]
fn diffuse_correction_scalar(sw_diffuse: f64, svf: f64) -> f64 {
    sw_diffuse * svf
}

/// Longwave radiation correction using Brutsaert (1975) emissivity model.
///
/// Separates clear-sky and cloud emissivity contributions, then transfers
/// cloud emissivity to the target location.  Avoids the T^4 ratio
/// amplification for large elevation differences.
#[inline]
fn longwave_correction_scalar(
    lw_source: f64,
    t_source: f64,
    t_unit: f64,
    vp_source: f64,
    vp_unit: f64,
    svf: f64,
) -> f64 {
    let inv_x2: f64 = 1.0 / 5.7;

    // Brutsaert (1975) clear-sky emissivity: 0.23 + 0.43*(e/T)^(1/5.7)
    let cse_source = 0.23 + 0.43 * (vp_source / t_source).max(0.0).powf(inv_x2);
    let cse_target = 0.23 + 0.43 * (vp_unit / t_unit).max(0.0).powf(inv_x2);

    // Total emissivity at source (from ERA5 LW measurement)
    let t4_source = t_source * t_source * t_source * t_source;
    let eps_total = lw_source / (SIGMA * t4_source);

    // Cloud emissivity (non-negative)
    let cle = (eps_total - cse_source).max(0.0);

    // All-sky emissivity at target (capped at 1.0)
    let aef = (cse_target + cle).min(1.0);

    let t4_unit = t_unit * t_unit * t_unit * t_unit;
    let lw_sky = aef * SIGMA * t4_unit;
    let lw_terrain = SIGMA * t4_unit;

    svf * lw_sky + (1.0 - svf) * lw_terrain
}

// ===========================================================================
// Array wrappers exposed to Python via PyO3
// ===========================================================================

/// Partition total shortwave into direct and diffuse components (Erbs et al. 1982).
///
/// SYNC: python_kernels/radiation.py::partition_shortwave
/// SYNC: jax_kernels/radiation.py::partition_shortwave
///
/// Returns
/// -------
/// (sw_direct, sw_diffuse) : tuple of ndarray
#[pyfunction]
fn partition_shortwave<'py>(
    py: Python<'py>,
    sw_total: PyReadonlyArrayDyn<'py, f64>,
    solar_elevation: PyReadonlyArrayDyn<'py, f64>,
    clearness_index: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
    let sw_total = sw_total.as_array();
    let solar_elevation = solar_elevation.as_array();
    let clearness_index = clearness_index.as_array();

    let mut direct = ndarray::ArrayD::<f64>::zeros(sw_total.shape());
    let mut diffuse = ndarray::ArrayD::<f64>::zeros(sw_total.shape());

    Zip::from(&mut direct)
        .and(&mut diffuse)
        .and(&sw_total)
        .and(&solar_elevation)
        .and(&clearness_index)
        .for_each(|d, df, &sw, &elev, &ci| {
            let (dir, dif) = partition_shortwave_scalar(sw, elev, ci);
            *d = dir;
            *df = dif;
        });

    Ok((
        PyArrayDyn::from_owned_array_bound(py, direct),
        PyArrayDyn::from_owned_array_bound(py, diffuse),
    ))
}

/// Apply slope correction to direct shortwave radiation.
///
/// SYNC: python_kernels/radiation.py::slope_correction
/// SYNC: jax_kernels/radiation.py::slope_correction
///
/// All angles in degrees.
#[pyfunction]
fn slope_correction<'py>(
    py: Python<'py>,
    sw_direct: PyReadonlyArrayDyn<'py, f64>,
    solar_elevation: PyReadonlyArrayDyn<'py, f64>,
    solar_azimuth: PyReadonlyArrayDyn<'py, f64>,
    slope: PyReadonlyArrayDyn<'py, f64>,
    aspect: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let sw_direct = sw_direct.as_array();
    let solar_elevation = solar_elevation.as_array();
    let solar_azimuth = solar_azimuth.as_array();
    let slope = slope.as_array();
    let aspect = aspect.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(sw_direct.shape());

    Zip::from(&mut out)
        .and(&sw_direct)
        .and(&solar_elevation)
        .and(&solar_azimuth)
        .and(&slope)
        .and(&aspect)
        .for_each(|o, &swd, &elev, &az, &sl, &asp| {
            *o = slope_correction_scalar(swd, elev, az, sl, asp);
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

/// Apply sky-view-factor correction to diffuse shortwave radiation.
///
/// SYNC: python_kernels/radiation.py::diffuse_correction
/// SYNC: jax_kernels/radiation.py::diffuse_correction
#[pyfunction]
fn diffuse_correction<'py>(
    py: Python<'py>,
    sw_diffuse: PyReadonlyArrayDyn<'py, f64>,
    svf: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let sw_diffuse = sw_diffuse.as_array();
    let svf = svf.as_array();

    let mut out = ndarray::ArrayD::<f64>::zeros(sw_diffuse.shape());

    Zip::from(&mut out)
        .and(&sw_diffuse)
        .and(&svf)
        .for_each(|o, &df, &s| {
            *o = diffuse_correction_scalar(df, s);
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

/// Apply longwave radiation correction with Brutsaert emissivity model.
///
/// SYNC: python_kernels/radiation.py::longwave_correction
/// SYNC: jax_kernels/radiation.py::longwave_correction
#[pyfunction]
fn longwave_correction<'py>(
    py: Python<'py>,
    lw_source: PyReadonlyArrayDyn<'py, f64>,
    t_source: PyReadonlyArrayDyn<'py, f64>,
    t_unit: PyReadonlyArrayDyn<'py, f64>,
    vp_source: PyReadonlyArrayDyn<'py, f64>,
    vp_unit: PyReadonlyArrayDyn<'py, f64>,
    svf: PyReadonlyArrayDyn<'py, f64>,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let lw_source = lw_source.as_array();
    let t_source = t_source.as_array();
    let t_unit = t_unit.as_array();
    let vp_source = vp_source.as_array();
    let vp_unit = vp_unit.as_array();
    let svf = svf.as_array();

    let inv_x2: f64 = 1.0 / 5.7;

    // Pass 1: clear-sky emissivity at source + total emissivity
    // (ndarray Zip supports max 6 producers, so we split into two passes)
    let mut cse_source = ndarray::ArrayD::<f64>::zeros(lw_source.shape());
    let mut eps_total = ndarray::ArrayD::<f64>::zeros(lw_source.shape());

    Zip::from(&mut cse_source)
        .and(&mut eps_total)
        .and(&lw_source)
        .and(&t_source)
        .and(&vp_source)
        .for_each(|cs, et, &lw, &ts, &vps| {
            *cs = 0.23 + 0.43 * (vps / ts).max(0.0).powf(inv_x2);
            let t4 = ts * ts * ts * ts;
            *et = lw / (SIGMA * t4);
        });

    // Pass 2: target emissivity, cloud transfer, final longwave
    let mut out = ndarray::ArrayD::<f64>::zeros(lw_source.shape());

    Zip::from(&mut out)
        .and(&cse_source)
        .and(&eps_total)
        .and(&t_unit)
        .and(&vp_unit)
        .and(&svf)
        .for_each(|o, &cs_src, &et, &tu, &vpu, &s| {
            let cse_tgt = 0.23 + 0.43 * (vpu / tu).max(0.0).powf(inv_x2);
            let cle = (et - cs_src).max(0.0);
            let aef = (cse_tgt + cle).min(1.0);
            let t4 = tu * tu * tu * tu;
            let lw_sky = aef * SIGMA * t4;
            let lw_terrain = SIGMA * t4;
            *o = s * lw_sky + (1.0 - s) * lw_terrain;
        });

    Ok(PyArrayDyn::from_owned_array_bound(py, out))
}

// ===========================================================================
// Module registration
// ===========================================================================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(partition_shortwave, m)?)?;
    m.add_function(wrap_pyfunction!(slope_correction, m)?)?;
    m.add_function(wrap_pyfunction!(diffuse_correction, m)?)?;
    m.add_function(wrap_pyfunction!(longwave_correction, m)?)?;
    Ok(())
}
