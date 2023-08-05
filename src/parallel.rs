use ndarray::ArrayView1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use rayon::prelude::*;

use anyhow::{anyhow, Error};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use rayon::iter::ParallelIterator;


use crate::sigma_points::SigmaPoints;
use crate::unscented_kalman_filter::UnscentedKalmanFilter;
use crate::Float;

#[pyclass]
pub struct UKFParallel {
    ukfs: Vec<UnscentedKalmanFilter>,
}

impl UKFParallel {
    pub fn new(ukfs: Vec<UnscentedKalmanFilter>) -> Self {
        Self { ukfs }
    }

    pub fn predict(&mut self, dt: Float) -> Result<(), Error> {
        self.ukfs
            .par_iter_mut()
            .try_for_each(|ukf| ukf.predict(dt))?;
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
            .iter_mut()
            .zip(z_vec.iter())
            .try_for_each(|(ukf, z)| ukf.update(*z))?;
        Ok(())
    }

    fn py_set_each<T, F>(&mut self, values: Vec<T>, mut func: F) -> PyResult<()>
    where
        F: FnMut(&mut UnscentedKalmanFilter, T) -> PyResult<()>,
    {
        for (ukf, value) in self.ukfs.iter_mut().zip(values.into_iter()) {
            // if let Some(ukf) = Arc::get_mut(ukf) {
            func(ukf, value)?;
            // } else {
            //     return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            //         "Failed to get mutable reference to UKF",
            //     ));
            // }
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
            ukf_vec.push(ukf);
            Ok(())
        })?;

        Ok(Self::new(ukf_vec))
    }

    #[pyo3(name = "predict")]
    pub fn py_predict(&mut self, py: Python<'_>, dt: Float) -> PyResult<()> {
        py.allow_threads(|| {
            self.predict(dt)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[pyo3(name = "update")]
    pub fn py_update(
        &mut self,
        py: Python<'_>,
        z_vec: Vec<PyReadonlyArray1<Float>>,
    ) -> PyResult<()> {
        let z_vec = z_vec
            .iter()
            .map(|z| z.as_array())
            .collect::<Vec<ArrayView1<Float>>>();

        py.allow_threads(|| {
            self.update(z_vec)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    }

    #[setter]
    #[pyo3(name = "x")]
    fn py_set_x(&mut self, value_vec: Vec<PyReadonlyArray1<Float>>) -> PyResult<()> {
        self.py_set_each(value_vec, UnscentedKalmanFilter::py_set_x)
    }

    #[getter]
    #[pyo3(name = "x")]
    fn py_get_x(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_x(py))
    }

    #[setter]
    #[pyo3(name = "P")]
    fn py_set_P(&mut self, value_vec: Vec<PyReadonlyArray2<Float>>) -> PyResult<()> {
        self.py_set_each(value_vec, UnscentedKalmanFilter::py_set_P)
    }

    #[getter]
    #[pyo3(name = "P")]
    fn py_get_P(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_P(py))
    }

    #[getter]
    #[pyo3(name = "x_prior")]
    pub fn py_get_x_prior(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_x_prior(py))
    }

    #[getter]
    #[pyo3(name = "P_prior")]
    pub fn py_get_P_prior(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_P_prior(py))
    }

    #[getter]
    #[pyo3(name = "x_post")]
    pub fn py_get_x_post(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_x_post(py))
    }

    #[getter]
    #[pyo3(name = "P_post")]
    pub fn py_get_P_post(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_P_post(py))
    }

    #[getter]
    #[pyo3(name = "Q")]
    pub fn py_get_Q(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_Q(py))
    }

    #[setter]
    #[pyo3(name = "Q")]
    pub fn py_set_Q(
        &mut self,
        value_vec: Vec<PyReadonlyArray2<Float>>,
    ) -> PyResult<()> {
        self.py_set_each(value_vec, UnscentedKalmanFilter::py_set_Q)
    }

    #[getter]
    #[pyo3(name = "R")]
    pub fn py_get_R(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_R(py))
    }

    #[setter]
    #[pyo3(name = "R")]
    pub fn py_set_R(
        &mut self,
        value_vec: Vec<PyReadonlyArray2<Float>>,
    ) -> PyResult<()> {
        self.py_set_each(value_vec, UnscentedKalmanFilter::py_set_R)
    }

    #[getter]
    #[pyo3(name = "sigma_points")]
    pub fn py_get_sigma_points(&self) -> PyResult<Vec<SigmaPoints>> {
        self.py_get_each(|ukf| ukf.py_get_sigma_points())
    }

    #[getter]
    #[pyo3(name = "dim_x")]
    pub fn py_get_dim_x(&self) -> PyResult<Vec<usize>> {
        self.py_get_each(|ukf| ukf.py_get_dim_x())
    }

    #[getter]
    #[pyo3(name = "dim_z")]
    pub fn py_get_dim_z(&self) -> PyResult<Vec<usize>> {
        self.py_get_each(|ukf| ukf.py_get_dim_z())
    }

    #[getter]
    #[pyo3(name = "K")]
    pub fn py_get_K(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_K(py))
    }

    #[getter]
    #[pyo3(name = "y")]
    pub fn py_get_y(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_y(py))
    }

    #[getter]
    #[pyo3(name = "z")]
    pub fn py_get_z(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray1<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_z(py))
    }

    #[getter]
    #[pyo3(name = "S")]
    pub fn py_get_S(&self, py: Python<'_>) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_S(py))
    }

    #[getter]
    #[pyo3(name = "sigmas_f")]
    pub fn py_get_sigmas_f(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_sigmas_f(py))
    }

    #[getter]
    #[pyo3(name = "sigmas_h")]
    pub fn py_get_sigmas_h(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<PyArray2<Float>>>> {
        self.py_get_each(|ukf| ukf.py_get_sigmas_h(py))
    }
}
