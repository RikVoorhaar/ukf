#![allow(non_snake_case)]

pub mod dynamic_functions;
// pub mod examples;
pub mod linalg_utils;
pub mod ring_buffer;
pub mod sigma_points;
pub mod unscented_kalman_filter;

pub mod types;
pub use types::Float;

use dynamic_functions::{MeasurementFunction, PythonMeasurementFunction};
// use examples::constant_speed_ukf;
use pyo3::prelude::*;
use sigma_points::SigmaPoints;
use unscented_kalman_filter::UnscentedKalmanFilter;



#[pymodule]
fn ukf(_py: Python, m: &PyModule) -> PyResult<()> {
    std::env::set_var("RUST_BACKTRACE", "full");

    pyo3_log::init();

    m.add_class::<SigmaPoints>()?;
    m.add_class::<UnscentedKalmanFilter>()?;
    m.add_class::<PythonMeasurementFunction>()?;
    // m.add_function(wrap_pyfunction!(constant_speed_ukf, m)?)?;

    Ok(())
}
