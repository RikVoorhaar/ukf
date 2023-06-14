use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{
    exceptions::PyRuntimeError, pyclass, pymodule, types::PyModule, PyResult, Python,
};
use ndarray::{Array1}

#[pyclass]
struct MerweSigmaPoints {
    size: usize,
    lambda: f32,
    Wm: Array1<f32>,
    Wc: Array1<f32>,
}