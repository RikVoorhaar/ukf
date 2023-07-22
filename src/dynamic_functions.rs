// Defines and implements traits for dynamic transition and measurement functions

use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

use crate::Float;

pub trait MeasurementFunction: Send {
    fn call(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>>;

    fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
    ) -> PyResult<Py<PyArray1<Float>>> {
        let x = x.as_array();
        let result = self.call(x)?;
        Ok(result.into_pyarray(py).to_owned())
    }
}

#[pyclass]
pub struct ContextContainer {
    pub context: PyObject,
}

impl ContextContainer {
    pub fn new(context: PyObject) -> Self {
        Self { context }
    }
}

impl Clone for ContextContainer {
    fn clone(&self) -> Self {
        Python::with_gil(|py| -> PyResult<Self> {
            let context = self.context.clone_ref(py);
            Ok(Self { context })
        })
        .unwrap()
    }
}

#[pyclass(name = "MeasurementFunction")]
pub struct PythonMeasurementFunction {
    pub h: PyObject,
    pub context: Py<ContextContainer>,
}

impl Clone for PythonMeasurementFunction {
    fn clone(&self) -> Self {
        Python::with_gil(|py| -> PyResult<Self> {
            let h = self.h.clone_ref(py);
            let context = self.context.clone_ref(py);
            Ok(Self { h, context })
        })
        .unwrap()
    }
}

impl MeasurementFunction for PythonMeasurementFunction {
    fn call(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>> {
        Python::with_gil(|py| -> PyResult<Array1<Float>> {
            let x_py = x.to_owned().into_pyarray(py);
            let context_container = self.context.borrow(py);
            let result_py = self.h.call1(py, (x_py, &context_container.context))?;
            let result_py: &PyArray1<Float> = result_py.downcast(py).map_err(|_| {
                PyValueError::new_err(
                    "Could not downcast result to PyArray1. \
                    Make sure return type is 1-dimensional numpy array of the right \
                    dtype.",
                )
            })?;

            let result = result_py.to_owned_array();
            Ok(result)
        })
    }
}

#[pymethods]
impl PythonMeasurementFunction {
    #[new]
    pub fn new(h: PyObject, py: Python<'_>, context: PyObject) -> Self {
        let context_container = Py::new(py, ContextContainer::new(context)).unwrap();
        Self {
            h,
            context: context_container,
        }
    }

    #[pyo3(name = "__call__")]
    pub fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
    ) -> PyResult<Py<PyArray1<Float>>> {
        self.call(x.as_array())
            .map(|result| result.into_pyarray(py).to_owned())
    }

    #[getter]
    #[pyo3(name = "context")]
    pub fn get_context(&self, py: Python<'_>) -> PyResult<PyObject> {
        let context_container = self.context.borrow(py);
        Ok(context_container.context.clone())
    }

    #[setter]
    #[pyo3(name = "context")]
    pub fn set_context(&mut self, py: Python<'_>, context: PyObject) -> PyResult<()> {
        let mut context_container = self.context.borrow_mut(py);
        context_container.context = context;
        Ok(())
    }
}
