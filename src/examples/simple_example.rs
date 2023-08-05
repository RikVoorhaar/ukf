use ndarray::{s, Array1, ArrayView1, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::dynamic_functions::{
    MeasurementFunction, MeasurementFunctionBox, TransitionFunction,
    TransitionFunctionBox,
};
use crate::Float;

#[pyclass]
#[derive(Clone)]
pub struct CoordinateProjectionFunction {
    dim_small: usize,
}

impl TransitionFunction for CoordinateProjectionFunction {
    fn call_f(&self, x: ArrayView1<Float>, _dt: Float) -> PyResult<Array1<Float>> {
        Ok(x.slice(s![..self.dim_small]).to_owned())
    }

    fn to_transition_box(&self) -> TransitionFunctionBox {
        TransitionFunctionBox {
            f: Box::new(self.clone()),
        }
    }

    fn update_py_context(&mut self, _: Python<'_>, _: PyObject) -> PyResult<()> {
        Ok(())
    }
}

impl MeasurementFunction for CoordinateProjectionFunction {
    fn call_h(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>> {
        Ok(x.slice(s![..self.dim_small]).to_owned())
    }

    fn to_measurement_box(&self) -> MeasurementFunctionBox {
        MeasurementFunctionBox {
            h: Box::new(self.clone()),
        }
    }
    fn update_py_context(&mut self, _: Python<'_>, _: PyObject) -> PyResult<()> {
        Ok(())
    }
}

#[pymethods]
impl CoordinateProjectionFunction {
    #[new]
    fn new(dim_small: usize) -> Self {
        Self { dim_small }
    }

    #[pyo3(name = "__call__")]
    pub fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        dt: Option<Float>,
    ) -> PyResult<Py<PyArray1<Float>>> {
        let x = x.as_array();
        match dt {
            Some(dt) => {
                let y = self.call_f(x.view(), dt)?;
                Ok(y.into_pyarray(py).to_owned())
            }
            None => {
                let y = self.call_h(x.view())?;
                Ok(y.into_pyarray(py).to_owned())
            }
        }
    }

    #[pyo3(name = "to_transition_box")]
    pub fn py_to_transition_box(&self) -> TransitionFunctionBox {
        self.to_transition_box()
    }

    #[pyo3(name = "to_measurement_box")]
    pub fn py_to_measurement_box(&self) -> MeasurementFunctionBox {
        self.to_measurement_box()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct FirstOrderTransitionFunction {
    dim: usize,
}

impl TransitionFunction for FirstOrderTransitionFunction {
    fn call_f(&self, x: ArrayView1<Float>, dt: Float) -> PyResult<Array1<Float>> {
        let mut x = x.to_owned();
        let (mut pos, vel) = x.view_mut().split_at(Axis(0), self.dim);

        pos.iter_mut()
            .zip(vel.iter())
            .for_each(|(x, v)| *x += v * dt);

        Ok(x)
    }

    fn to_transition_box(&self) -> TransitionFunctionBox {
        TransitionFunctionBox {
            f: Box::new(self.clone()),
        }
    }

    fn update_py_context(&mut self, _: Python<'_>, _: PyObject) -> PyResult<()> {
        Ok(())
    }
}

#[pymethods]
impl FirstOrderTransitionFunction {
    #[new]
    fn new(dim: usize) -> Self {
        Self { dim }
    }

    #[pyo3(name = "__call__")]
    fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        dt: Float,
    ) -> PyResult<Py<PyArray1<Float>>> {
        let x = x.as_array();
        let y = self.call_f(x.view(), dt)?;
        Ok(y.into_pyarray(py).to_owned())
    }

    #[pyo3(name = "to_transition_box")]
    pub fn py_to_transition_box(&self) -> TransitionFunctionBox {
        self.to_transition_box()
    }
}
