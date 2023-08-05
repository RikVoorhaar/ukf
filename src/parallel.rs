use ndarray::ArrayView1;
use numpy::{PyArray1, PyReadonlyArray1};
use rayon::prelude::*;

use anyhow::{anyhow, Error};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::sync::Arc;

use crate::unscented_kalman_filter::UnscentedKalmanFilter;
use crate::Float;

#[pyclass]
pub struct UKFParallel {
    ukfs: Vec<Arc<UnscentedKalmanFilter>>,
}

impl UKFParallel {
    pub fn new(ukfs: Vec<Arc<UnscentedKalmanFilter>>) -> Self {
        Self { ukfs }
    }

    pub fn predict(&mut self, dt_vec: Vec<Float>) -> Result<(), Error> {
        if self.ukfs.len() != dt_vec.len() {
            return Err(anyhow!(
                "Length of UKF vector ({}) does not match length of dt vector ({})",
                self.ukfs.len(),
                dt_vec.len()
            ));
        }

        self.ukfs
            .par_iter_mut()
            .zip(dt_vec.par_iter())
            .try_for_each(|(ukf, dt)| {
                Arc::get_mut(ukf).map_or_else(
                    || Err(anyhow!("Failed to get mutable reference to UKF")),
                    |ukf| ukf.predict(*dt),
                )
            })?;
        Ok(())
    }

    pub fn update(&mut self, z_vec: Vec<ArrayView1<Float>>) -> Result<(), Error> {
        if self.ukfs.len() != z_vec.len() {
            return Err(anyhow!(
                "Length of UKF vector ({}) does not match length of z vector ({})",
                self.ukfs.len(),
                z_vec.len()
            ));
        }

        self.ukfs
            .par_iter_mut()
            .zip(z_vec.par_iter())
            .try_for_each(|(ukf, z)| {
                Arc::get_mut(ukf).map_or_else(
                    || Err(anyhow!("Failed to get mutable reference to UKF")),
                    |ukf| ukf.update(*z),
                )
            })?;
        Ok(())
    }

    fn py_set_each<T, F>(&mut self, values: Vec<T>, mut func: F) -> PyResult<()>
    where
        F: FnMut(&mut UnscentedKalmanFilter, T) -> PyResult<()>,
    {
        for (ukf, value) in self.ukfs.iter_mut().zip(values.into_iter()) {
            if let Some(ukf) = Arc::get_mut(ukf) {
                func(ukf, value)?;
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Failed to get mutable reference to UKF",
                ));
            }
        }
        Ok(())
    }

    fn py_get_each<T, F>(&self, func: F) -> PyResult<Vec<T>>
    where
        F: Fn(&UnscentedKalmanFilter) -> PyResult<T>,
    {
        let mut values = Vec::new();
        for ukf in self.ukfs.iter() {
            values.push(func(ukf)?);
        }
        Ok(values)
    }
}

#[pymethods]
impl UKFParallel {
    #[new]
    pub fn py_new(py: Python, ukfs: Vec<PyObject>) -> PyResult<Self> {
        let mut ukf_vec = Vec::new();

        ukfs.iter().try_for_each(|ukf| -> PyResult<()> {
            let ukf = ukf.extract::<UnscentedKalmanFilter>(py).map_err(|_| {
                pyo3::exceptions::PyTypeError::new_err(
                    "Failed to extract UKF from list",
                )
            })?;
            ukf_vec.push(Arc::new(ukf));
            Ok(())
        })?;

        Ok(Self::new(ukf_vec))
    }

    #[pyo3(name = "predict")]
    pub fn py_predict(&mut self, dt_vec: Vec<Float>) -> PyResult<()> {
        self.predict(dt_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[pyo3(name = "update")]
    pub fn py_update(&mut self, z_vec: Vec<PyReadonlyArray1<Float>>) -> PyResult<()> {
        let z_vec = z_vec
            .iter()
            .map(|z| z.as_array())
            .collect::<Vec<ArrayView1<Float>>>();

        self.update(z_vec)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    #[setter]
    #[pyo3(name = "x")]
    fn py_set_x(&mut self, x_vec: Vec<PyReadonlyArray1<Float>>) -> PyResult<()> {
        self.py_set_each(x_vec, UnscentedKalmanFilter::py_set_x)
    }

    #[getter]
    #[pyo3(name = "x")]
    fn py_get_x(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_x(py))
    }
}
