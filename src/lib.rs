#![allow(non_snake_case)]

pub mod ring_buffer;
pub mod sigma_points;
pub mod unscented_kalman_filter;
pub mod linalg_utils;

pub mod types;
pub use types::Float;

use pyo3::prelude::*;
use sigma_points::SigmaPoints;
use unscented_kalman_filter::UnscentedKalmanFilter;

#[pymodule]
fn ukf(_py: Python, m: &PyModule) -> PyResult<()> {
    std::env::set_var("RUST_BACKTRACE", "full");
    pyo3_log::init();

    m.add_class::<SigmaPoints>()?;
    m.add_class::<UnscentedKalmanFilter>()?;
    Ok(())
}
