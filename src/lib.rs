#![allow(non_snake_case)]

pub mod ring_buffer;
pub mod sigma_points;
// pub mod py;

use pyo3::prelude::*;
use sigma_points::MerweSigmaPoints;

#[pymodule]
fn ukf(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MerweSigmaPoints>()?;
    Ok(())
}