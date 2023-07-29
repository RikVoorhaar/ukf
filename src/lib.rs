#![allow(non_snake_case)]

pub mod dynamic_functions;
// pub mod examples;
pub mod examples;
pub mod linalg_utils;
pub mod ring_buffer;
pub mod sigma_points;
pub mod unscented_kalman_filter;

pub mod types;
pub use types::Float;

use dynamic_functions::{PythonMeasurementFunction, PythonTransitionFunction};
use pyo3::prelude::*;
use sigma_points::SigmaPoints;
use unscented_kalman_filter::UnscentedKalmanFilter;

#[pymodule]
fn ukf_pyrs(_py: Python, m: &PyModule) -> PyResult<()> {
    std::env::set_var("RUST_BACKTRACE", "full");

    pyo3_log::init();

    m.add_class::<SigmaPoints>()?;
    m.add_class::<UnscentedKalmanFilter>()?;
    m.add_class::<PythonMeasurementFunction>()?;
    m.add_class::<PythonTransitionFunction>()?;
    m.add_class::<examples::pinhole_camera::PinholeCamera>()?;
    m.add_class::<examples::pinhole_camera::CameraProjector>()?;
    // m.add_function(wrap_pyfunction!(constant_speed_ukf, m)?)?;

    Ok(())
}
