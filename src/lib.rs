#![allow(non_snake_case)]

pub mod dynamic_functions;
pub mod functions;
pub mod linalg_utils;
pub mod ring_buffer;
pub mod sigma_points;
pub mod unscented_kalman_filter;
pub mod parallel;

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
    m.add_class::<functions::pinhole_camera::PinholeCamera>()?;
    m.add_class::<functions::pinhole_camera::CameraProjector>()?;
    m.add_class::<functions::simple_example::CoordinateProjectionFunction>()?;
    m.add_class::<functions::simple_example::FirstOrderTransitionFunction>()?;
    m.add_class::<parallel::UKFParallel>()?;

    Ok(())
}
