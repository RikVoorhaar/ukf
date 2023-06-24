#![allow(non_snake_case)]

pub mod ring_buffer;
pub mod sigma_points;
pub mod unscented_kalman_filter;

pub mod types;
pub use types::Float;

use pyo3::prelude::*;
use sigma_points::SigmaPoints;

#[pymodule]
fn ukf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SigmaPoints>()?;
    Ok(())
}