use ndarray::{Array1, Array2, ArrayView1, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use anyhow::{anyhow, Error};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

use crate::dynamic_functions::{MeasurementFunctionBox, TransitionFunctionBox};
use crate::linalg_utils::right_solve_h;
use crate::sigma_points::SigmaPoints;
use crate::Float;

#[pyclass(name = "UKF")]
#[derive(Clone)]
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
    hx: MeasurementFunctionBox,
    fx: TransitionFunctionBox,
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
        hx: MeasurementFunctionBox,
        fx: TransitionFunctionBox,
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
        let sigmas_f: Array2<Float> = Array2::zeros((2 * dim_x + 1, dim_x));
        let sigmas_h: Array2<Float> = Array2::zeros((2 * dim_x + 1, dim_z));
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
        let x: Array1<Float> = self.sigma_points.get_Wm().dot(sigmas);

        let y = &sigmas.t() - x.clone().insert_axis(Axis(1));

        let wc_diag = Array2::from_diag(&self.sigma_points.get_Wc().to_owned());

        let mut P = y.dot(&wc_diag).dot(&y.t());
        // FIXME: the diagonal matrix here is inefficient.

        if let Some(nc) = noise_cov {
            P += nc;
        }
        (x, P)
    }

    // TODO: Make this use an immutable self and a return value instead
    fn compute_process_sigmas(&mut self, dt: Float) -> Result<(), Error> {
        let sigmas = self.sigma_points.call(self.x.view(), self.P.view())?;

        // for (row_sigmas, mut row_sigmas_f) in
        //     sigmas.axis_iter(Axis(0)).zip(self.sigmas_f.rows_mut())
        // {
        //     let result =
        //         self.fx.f.call_f(row_sigmas, dt).map_err(|e| {
        //             anyhow!(format!("Error in transition function: {}", e))
        //         })?;
        //     row_sigmas_f.assign(&result);
        // }
        self.fx
            .f
            .call_f_batch_mut(sigmas.view(), dt, &mut self.sigmas_f).map_err(|e| {
                anyhow!(format!("Error in transition function: {}", e))
            })?;

        Ok(())
    }

    pub fn predict(&mut self, dt: Float) -> Result<(), Error> {
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
        let mut L = &self.sigmas_f - x.insert_axis(Axis(0)).to_owned();
        for k in 0..self.sigma_points.n_points() {
            let mut row = L.row_mut(k);
            row *= Wc[k];
        }
        let R = &self.sigmas_h - z.insert_axis(Axis(0)).to_owned();

        L.t().dot(&R)
    }

    pub fn update(&mut self, z: ArrayView1<Float>) -> Result<(), Error> {
        self.sigmas_h = self.hx.h.call_h_batch(self.sigmas_f.view())?;

        let (z_pred, S) = self.unscented_transform(&self.sigmas_h, Some(&self.R));
        let Pxz = self.cross_variance(self.x.view(), z_pred.view());
        self.K = right_solve_h(&Pxz, S.clone());
        self.S = S;

        self.z = z.to_owned();
        self.y = z.to_owned() - z_pred; // TODO: Remove this from the struct

        self.x += &self.K.dot(&self.y);
        self.P -= &self.K.dot(&self.S.dot(&self.K.t()));

        self.x_post = self.x.clone();
        self.P_post = self.P.clone();

        Ok(())
    }
}

type UnscentedTransformResult = (Py<PyArray1<Float>>, Py<PyArray2<Float>>);
#[pymethods]
impl UnscentedKalmanFilter {
    #[new]
    fn py_new(
        py: Python,
        dim_x: usize,
        dim_z: usize,
        hx: PyObject,
        fx: PyObject,
        sigma_points: &PyAny,
    ) -> PyResult<Self> {
        let sigma_points_rust = sigma_points.extract::<SigmaPoints>()?;
        let h: MeasurementFunctionBox =
            hx.call_method0(py, "to_measurement_box")?.extract(py)?;
        let f: TransitionFunctionBox =
            fx.call_method0(py, "to_transition_box")?.extract(py)?;

        Ok(Self::new(dim_x, dim_z, h, f, sigma_points_rust))
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

    #[pyo3(name = "cross_variance")]
    fn py_cross_variance(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        z: PyReadonlyArray1<Float>,
    ) -> PyResult<Py<PyArray2<Float>>> {
        let x = x.as_array();
        let z = z.as_array();
        let result = self.cross_variance(x, z);
        let result = result.into_pyarray(py).to_owned();
        Ok(result)
    }

    #[pyo3(name = "unscented_transform")]
    fn py_unscented_transform(
        &self,
        py: Python<'_>,
        sigmas: &PyAny,
        additive: Option<PyReadonlyArray2<Float>>,
    ) -> PyResult<UnscentedTransformResult> {
        let sigmas = sigmas.extract::<PyReadonlyArray2<Float>>()?;
        let additive = additive.map(|a| a.as_array().to_owned());
        let (x, P) =
            self.unscented_transform(&sigmas.as_array().to_owned(), additive.as_ref());
        let x = x.into_pyarray(py).to_owned();
        let P = P.into_pyarray(py).to_owned();
        Ok((x, P))
    }

    #[getter]
    #[pyo3(name = "x")]
    pub fn py_get_x(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "x")]
    pub fn py_set_x(&mut self, x: PyReadonlyArray1<Float>) -> PyResult<()> {
        self.x = x.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "P")]
    pub fn py_get_P(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "P")]
    pub fn py_set_P(&mut self, P: PyReadonlyArray2<Float>) -> PyResult<()> {
        self.P = P.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "x_prior")]
    pub fn py_get_x_prior(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x_prior.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "P_prior")]
    pub fn py_get_P_prior(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P_prior.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "x_post")]
    pub fn py_get_x_post(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x_post.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "P_post")]
    pub fn py_get_P_post(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P_post.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "Q")]
    pub fn py_get_Q(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.Q.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "Q")]
    pub fn py_set_Q(&mut self, Q: PyReadonlyArray2<Float>) -> PyResult<()> {
        self.Q = Q.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "R")]
    pub fn py_get_R(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.R.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "R")]
    pub fn py_set_R(&mut self, R: PyReadonlyArray2<Float>) -> PyResult<()> {
        self.R = R.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "sigma_points")]
    pub fn py_get_sigma_points(&self) -> PyResult<SigmaPoints> {
        Ok(self.sigma_points.clone())
    }

    #[getter]
    #[pyo3(name = "dim_x")]
    pub fn py_get_dim_x(&self) -> PyResult<usize> {
        Ok(self.dim_x)
    }

    #[getter]
    #[pyo3(name = "dim_z")]
    pub fn py_get_dim_z(&self) -> PyResult<usize> {
        Ok(self.dim_z)
    }

    #[getter]
    #[pyo3(name = "K")]
    pub fn py_get_K(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.K.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "y")]
    pub fn py_get_y(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.y.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "z")]
    pub fn py_get_z(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.z.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "S")]
    pub fn py_get_S(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.S.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "sigmas_f")]
    pub fn py_get_sigmas_f(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.sigmas_f.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "sigmas_h")]
    pub fn py_get_sigmas_h(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.sigmas_h.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    pub fn update_measurement_context(
        &mut self,
        py: Python<'_>,
        context: PyObject,
    ) -> PyResult<()> {
        self.hx.h.update_py_context(py, context)
    }

    pub fn update_transition_context(
        &mut self,
        py: Python<'_>,
        context: PyObject,
    ) -> PyResult<()> {
        self.fx.f.update_py_context(py, context)
    }
}
