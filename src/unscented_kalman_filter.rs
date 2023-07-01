use log::debug;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::{FactorizeHInto, SolveH};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::{error::Error, sync::Arc};

use crate::sigma_points::SigmaPoints;
use crate::Float;

pub type RustTranstionFunction =
    Arc<dyn Fn(ArrayView1<Float>, Float) -> PyResult<Array1<Float>> + Send + Sync>;

pub enum HybridTransitionFunction {
    Rust(RustTranstionFunction),
    Python(PyObject),
}

impl HybridTransitionFunction {
    pub fn call(&self, x: ArrayView1<Float>, dt: Float) -> PyResult<Array1<Float>> {
        match self {
            HybridTransitionFunction::Rust(f) => f(x, dt),
            HybridTransitionFunction::Python(f) => {
                Python::with_gil(|py| -> PyResult<Array1<Float>> {
                    let x_py = x.to_owned().into_pyarray(py);
                    let result_py = f.call1(py, (x_py, dt))?;
                    let result_py: &PyArray1<Float> =
                        result_py.downcast(py).map_err(|_| {
                            PyValueError::new_err("Could not downcast result to PyArray1. \
                                Make sure return type is 1-dimensional numpy array of the right dtype.")
                        })?;

                    let result = result_py.to_owned_array();
                    Ok(result)
                })
            }
        }
    }
}

pub type RustMeasurementFunction =
    Arc<dyn Fn(ArrayView1<Float>) -> PyResult<Array1<Float>> + Send + Sync>;

pub enum HybridMeasurementFunction {
    Rust(RustMeasurementFunction),
    Python(PyObject),
}

impl HybridMeasurementFunction {
    pub fn call(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>> {
        match self {
            HybridMeasurementFunction::Rust(f) => f(x),
            HybridMeasurementFunction::Python(f) => {
                Python::with_gil(|py| -> PyResult<Array1<Float>> {
                    let x_py = x.to_owned().into_pyarray(py);
                    let result_py = f.call1(py, (x_py,))?;
                    let result_py: &PyArray1<Float> =
                        result_py.downcast(py).map_err(|_| {
                            PyValueError::new_err("Could not downcast result to PyArray1. \
                                Make sure return type is 1-dimensional numpy array of the right dtype.")
                        })?;
                    let result = result_py.to_owned_array();
                    Ok(result)
                })
            }
        }
    }
}

#[pyclass]
pub struct UnscentedKalmanFilter {
    pub x: Array1<Float>,
    pub P: Array2<Float>,
    pub x_prior: Array1<Float>,
    pub P_prior: Array2<Float>,
    pub x_post: Array1<Float>,
    pub P_post: Array2<Float>,
    pub Q: Array2<Float>,
    pub R: Array2<Float>,
    sigma_points: SigmaPoints,
    pub dim_x: usize,
    pub dim_z: usize,
    hx: HybridMeasurementFunction,
    fx: HybridTransitionFunction,
    sigmas_f: Array2<Float>,
    sigmas_h: Array2<Float>,
    pub K: Array2<Float>,
    y: Array1<Float>,
    pub z: Array1<Float>,
    pub S: Array2<Float>,
}

impl UnscentedKalmanFilter {
    pub fn new(
        dim_x: usize,
        dim_z: usize,
        hx: HybridMeasurementFunction,
        fx: HybridTransitionFunction,
        sigma_points: SigmaPoints,
    ) -> Self {
        let x: Array1<Float> = Array1::zeros(dim_x);
        let P: Array2<Float> = Array2::eye(dim_x);
        let x_prior: Array1<Float> = Array1::zeros(dim_x);
        let P_prior: Array2<Float> = Array2::eye(dim_x);
        let x_post: Array1<Float> = Array1::zeros(dim_x);
        let P_post: Array2<Float> = Array2::eye(dim_x);
        let Q: Array2<Float> = Array2::eye(dim_x);
        let R: Array2<Float> = Array2::eye(dim_z);
        let sigmas_f: Array2<Float> = Array2::zeros((dim_x, 2 * dim_x + 1));
        let sigmas_h: Array2<Float> = Array2::zeros((dim_z, 2 * dim_x + 1));
        let K: Array2<Float> = Array2::zeros((dim_x, dim_z));
        let y: Array1<Float> = Array1::zeros(dim_z);
        let z: Array1<Float> = Array1::zeros(dim_z);
        let S: Array2<Float> = Array2::zeros((dim_z, dim_z));

        UnscentedKalmanFilter {
            x,
            P,
            x_prior,
            P_prior,
            x_post,
            P_post,
            Q,
            R,
            sigma_points,
            dim_x,
            dim_z,
            hx,
            fx,
            sigmas_f,
            sigmas_h,
            K,
            y,
            z,
            S,
        }
    }

    fn unscented_transform(
        &self,
        sigmas: &Array2<Float>,
        noise_cov: Option<&Array2<Float>>,
    ) -> (Array1<Float>, Array2<Float>) {
        let x: Array1<Float> = self.sigma_points.get_Wm().dot(&sigmas.t());

        let y = sigmas - x.clone().insert_axis(Axis(1));

        let wc_diag = Array2::from_diag(&self.sigma_points.get_Wc().to_owned());

        let mut P = y.dot(&wc_diag).dot(&y.t());
        // FIXME: the diagonal matrix here is inefficient.

        if let Some(nc) = noise_cov {
            P += nc;
        }
        (x, P)
    }

    fn compute_process_sigmas(&mut self, dt: Float) -> Result<(), Box<dyn Error>> {
        let sigmas = self.sigma_points.call(self.x.view(), self.P.view())?;

        for (col_sigmas, mut col_sigmas_f) in
            sigmas.outer_iter().zip(self.sigmas_f.columns_mut())
        {
            let result = self.fx.call(col_sigmas, dt)?;

            col_sigmas_f.assign(&result);
        }

        Ok(())
    }

    pub fn predict(&mut self, dt: Float) -> Result<(), Box<dyn Error>> {
        self.compute_process_sigmas(dt)?;

        let (x, P) = self.unscented_transform(&self.sigmas_f, Some(&self.Q));

        self.x_prior = x.clone();
        self.P_prior = P.clone();
        self.x = x;
        self.P = P;

        Ok(())
    }

    fn cross_variance(
        &self,
        x: ArrayView1<Float>,
        z: ArrayView1<Float>,
    ) -> Array2<Float> {
        let Wc = self.sigma_points.get_Wc();

        let mut L = &self.sigmas_f - x.insert_axis(Axis(1)).to_owned();

        for k in 0..self.sigma_points.n_points() {
            L.row_mut(k).mapv_inplace(|v| v * Wc[k]);
        }
        let R = &self.sigmas_h - z.insert_axis(Axis(1)).to_owned();

        L.dot(&R.t())
    }

    pub fn update(&mut self, z: ArrayView1<Float>) -> Result<(), Box<dyn Error>> {
        for i in 0..self.sigma_points.n_points() {
            let new_column = self.hx.call(self.sigmas_f.column(i))?;
            self.sigmas_h.column_mut(i).assign(&new_column);
        }

        let (z_pred, S) = self.unscented_transform(&self.sigmas_h, Some(&self.R));
        let mut Pxz = self.cross_variance(self.x.view(), z_pred.view());
        self.S = S.clone();

        // Right Hermitian solve consuming S and Pxz into K
        let S_factor = S.factorizeh_into().unwrap();
        Pxz.outer_iter_mut()
            .zip(self.K.rows_mut())
            .for_each(|(row_Pxz, mut row_K)| {
                row_K.assign(&S_factor.solveh_into(row_Pxz).unwrap())
            });

        self.z = z.to_owned();
        self.y = z.to_owned() - z_pred;

        self.x += &self.K.dot(&self.y);
        self.P -= &self.K.dot(&self.S.dot(&self.K.t()));

        self.x_post = self.x.clone();
        self.P_post = self.P.clone();

        Ok(())
    }
}

#[pymethods]
impl UnscentedKalmanFilter {
    #[new]
    fn py_new(
        dim_x: usize,
        dim_z: usize,
        hx: PyObject,
        fx: PyObject,
        sigma_points: &PyAny,
    ) -> PyResult<Self> {
        let sigma_points_rust = sigma_points.extract::<SigmaPoints>()?;

        Ok(Self::new(
            dim_x,
            dim_z,
            HybridMeasurementFunction::Python(hx),
            HybridTransitionFunction::Python(fx),
            sigma_points_rust,
        ))
    }

    #[pyo3(name = "predict")]
    fn py_predict(&mut self, dt: Float) -> PyResult<()> {
        self.predict(dt).map_err(|err| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error in predict: {}",
                err
            ))
        })
    }

    #[pyo3(name = "update")]
    fn py_update(&mut self, z: PyReadonlyArray1<Float>) -> PyResult<()> {
        let z = z.as_array();
        self.update(z).map_err(|err| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
                "Error in update: {}",
                err
            ))
        })
    }

    #[getter]
    #[pyo3(name = "x")]
    fn py_get_x(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "P")]
    fn py_get_P(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "Q")]
    fn py_get_Q(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.Q.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "R")]
    fn py_get_R(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.R.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "sigma_points")]
    fn py_get_sigma_points(&self) -> PyResult<SigmaPoints> {
        Ok(self.sigma_points.clone())
    }

    #[getter]
    #[pyo3(name = "dim_x")]
    fn py_get_dim_x(&self) -> PyResult<usize> {
        Ok(self.dim_x)
    }

    #[getter]
    #[pyo3(name = "dim_z")]
    fn py_get_dim_z(&self) -> PyResult<usize> {
        Ok(self.dim_z)
    }

    #[getter]
    #[pyo3(name = "K")]
    fn py_get_K(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.K.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "y")]
    fn py_get_y(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.y.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "z")]
    fn py_get_z(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.z.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "S")]
    fn py_get_S(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.S.clone().into_pyarray(py).to_owned();
        Ok(array)
    }
}
