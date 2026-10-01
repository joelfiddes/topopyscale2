use pyo3::prelude::*;

mod temperature;
mod radiation;
mod precipitation;
mod humidity;
mod wind;
mod redistribution;
mod interpolation;
// Snow-model kernels (FSM1, gFSM) are an application-layer feature, not part of the
// downscaling core: built only with the "snow" cargo feature (on by default here;
// the public core-only release builds without it).
#[cfg(feature = "snow")]
mod fsm1;
#[cfg(feature = "snow")]
mod gfsm;

#[pymodule]
fn _rust_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let temperature_mod = PyModule::new_bound(m.py(), "temperature")?;
    temperature::register(&temperature_mod)?;
    m.add_submodule(&temperature_mod)?;

    let radiation_mod = PyModule::new_bound(m.py(), "radiation")?;
    radiation::register(&radiation_mod)?;
    m.add_submodule(&radiation_mod)?;

    let precipitation_mod = PyModule::new_bound(m.py(), "precipitation")?;
    precipitation::register(&precipitation_mod)?;
    m.add_submodule(&precipitation_mod)?;

    let humidity_mod = PyModule::new_bound(m.py(), "humidity")?;
    humidity::register(&humidity_mod)?;
    m.add_submodule(&humidity_mod)?;

    let wind_mod = PyModule::new_bound(m.py(), "wind")?;
    wind::register(&wind_mod)?;
    m.add_submodule(&wind_mod)?;

    let redistribution_mod = PyModule::new_bound(m.py(), "redistribution")?;
    redistribution::register(&redistribution_mod)?;
    m.add_submodule(&redistribution_mod)?;

    let interpolation_mod = PyModule::new_bound(m.py(), "interpolation")?;
    interpolation::register_interpolation(&interpolation_mod)?;
    m.add_submodule(&interpolation_mod)?;

    #[cfg(feature = "snow")]
    {
        let fsm1_mod = PyModule::new_bound(m.py(), "fsm1")?;
        fsm1::register(&fsm1_mod)?;
        m.add_submodule(&fsm1_mod)?;

        let gfsm_mod = PyModule::new_bound(m.py(), "gfsm")?;
        gfsm::register(&gfsm_mod)?;
        m.add_submodule(&gfsm_mod)?;
    }

    Ok(())
}
