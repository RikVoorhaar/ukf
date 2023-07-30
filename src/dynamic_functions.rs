// Defines and implements traits for dynamic transition and measurement functions

use ndarray::{Array1, ArrayView1};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

use crate::Float;

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
// -------------------------------------------------------------------------------------
// Measurement function
// -------------------------------------------------------------------------------------
pub trait MeasurementFunction: Send {
    fn call_h(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>>;

    fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
    ) -> PyResult<Py<PyArray1<Float>>> {
        let x = x.as_array();
        let result = self.call_h(x)?;
        Ok(result.into_pyarray(py).to_owned())
    }

    fn to_measurement_box(&self) -> MeasurementFunctionBox;
}

#[pyclass(name = "MeasurementFunctionBox")]
pub struct MeasurementFunctionBox {
    pub h: Box<dyn MeasurementFunction>,
}

impl Clone for MeasurementFunctionBox {
    fn clone(&self) -> Self {
        self.h.to_measurement_box()
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
    fn call_h(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>> {
        Python::with_gil(|py| -> PyResult<Array1<Float>> {
            let x_py = x.to_owned().into_pyarray(py);
            let context_container = self.context.borrow(py);
            let context = &context_container.context;

            let result_py = if context.is_none(py) {
                self.h.call1(py, (x_py,))
            } else {
                self.h.call1(py, (x_py, context))
            }?;
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

    fn to_measurement_box(&self) -> MeasurementFunctionBox {
        let h = self.clone();
        MeasurementFunctionBox { h: Box::new(h) }
    }
}

#[pymethods]
impl PythonMeasurementFunction {
    #[new]
    pub fn new(h: PyObject, py: Python<'_>, context: Option<PyObject>) -> Self {
        let context_container = match context {
            Some(context) => Py::new(py, ContextContainer::new(context)).unwrap(),
            None => Py::new(py, ContextContainer::new(py.None())).unwrap(),
        };
        Self {
            h,
            context: context_container,
        }
    }

    #[pyo3(name = "to_measurement_box")]
    pub fn py_to_measurement_box(&self) -> PyResult<MeasurementFunctionBox> {
        let h = self.clone();
        Ok(MeasurementFunctionBox { h: Box::new(h) })
    }

    #[pyo3(name = "__call__")]
    pub fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
    ) -> PyResult<Py<PyArray1<Float>>> {
        self.call_h(x.as_array())
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

// -------------------------------------------------------------------------------------
// Transition function
// -------------------------------------------------------------------------------------
pub trait TransitionFunction: Send {
    fn call_f(&self, x: ArrayView1<Float>, dt: Float) -> PyResult<Array1<Float>>;

    fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        dt: Float,
    ) -> PyResult<Py<PyArray1<Float>>> {
        let x = x.as_array();
        let result = self.call_f(x, dt)?;
        Ok(result.into_pyarray(py).to_owned())
    }

    fn to_transition_box(&self) -> TransitionFunctionBox;
}
#[pyclass]
pub struct TransitionFunctionBox {
    pub f: Box<dyn TransitionFunction>,
}

impl Clone for TransitionFunctionBox {
    fn clone(&self) -> Self {
        self.f.to_transition_box()
    }
}

#[pyclass(name = "TransitionFunction")]
pub struct PythonTransitionFunction {
    pub f: PyObject,
    pub context: Py<ContextContainer>,
}

impl Clone for PythonTransitionFunction {
    fn clone(&self) -> Self {
        Python::with_gil(|py| -> PyResult<Self> {
            let f = self.f.clone_ref(py);
            let context = self.context.clone_ref(py);
            Ok(Self { f, context })
        })
        .unwrap()
    }
}

impl TransitionFunction for PythonTransitionFunction {
    fn call_f(&self, x: ArrayView1<Float>, dt: Float) -> PyResult<Array1<Float>> {
        Python::with_gil(|py| -> PyResult<Array1<Float>> {
            let x_py = x.to_owned().into_pyarray(py);
            let context_container = self.context.borrow(py);
            let context = &context_container.context;

            let result_py = if context.is_none(py) {
                self.f.call1(py, (x_py, dt))
            } else {
                self.f.call1(py, (x_py, dt, context))
            }?;
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

    fn to_transition_box(&self) -> TransitionFunctionBox {
        let f = self.clone();
        TransitionFunctionBox { f: Box::new(f) }
    }
}

#[pymethods]
impl PythonTransitionFunction {
    #[new]
    pub fn new(f: PyObject, py: Python<'_>, context: Option<PyObject>) -> Self {
        let context_container = match context {
            Some(context) => Py::new(py, ContextContainer::new(context)).unwrap(),
            None => Py::new(py, ContextContainer::new(py.None())).unwrap(),
        };
        Self {
            f,
            context: context_container,
        }
    }

    #[pyo3(name = "__call__")]
    pub fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        dt: Float,
    ) -> PyResult<Py<PyArray1<Float>>> {
        self.call_f(x.as_array(), dt)
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

    #[pyo3(name = "to_transition_box")]
    pub fn py_to_transition_box(&self) -> PyResult<TransitionFunctionBox> {
        let f = self.clone();
        Ok(TransitionFunctionBox { f: Box::new(f) })
    }
}
