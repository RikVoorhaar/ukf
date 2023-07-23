use log::debug;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::error::Error;

use crate::dynamic_functions::{
    MeasurementFunction, PythonMeasurementFunction, PythonTransitionFunction,
    TransitionFunction,
};
use crate::linalg_utils::{right_solve_h, smallest_eigenvalue};
use crate::sigma_points::SigmaPoints;
use crate::Float;

#[pyclass(name = "UKF")]
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
    hx: Box<dyn MeasurementFunction>,
    fx: Box<dyn TransitionFunction>,
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
        hx: Box<dyn MeasurementFunction>,
        fx: Box<dyn TransitionFunction>,
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
        debug!("ut");
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

    // TODO: Make this use an immutable self and a return value instead
    fn compute_process_sigmas(&mut self, dt: Float) -> Result<(), Box<dyn Error>> {
        debug!("process sigmas");
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
        debug!("predict");
        self.compute_process_sigmas(dt)?;

        let (x, P) = self.unscented_transform(&self.sigmas_f, Some(&self.Q));
        debug!("self.P smallest eigval: {}", smallest_eigenvalue(&self.P));
        debug!("P smallest eigval: {}", smallest_eigenvalue(&P));

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
        debug!("cross variance");
        let Wc = self.sigma_points.get_Wc();

        let mut L = &self.sigmas_f - x.insert_axis(Axis(1)).to_owned();

        debug!("Shape of L: {:?}", L.shape());
        debug!("L before\n{:?}", L);
        debug!("Shape of Wc: {:?}", Wc.shape());
        // for k in 0..self.sigma_points.n_points() {
        //     L.column_mut(k).mapv_inplace(|v| v * Wc[k]);
        // }
        debug!("n_points= {:?}", self.sigma_points.n_points());
        for k in 0..self.sigma_points.n_points() {
            let mut col = L.column_mut(k);
            col *= Wc[k];
        }
        debug!("L after\n{:?}", L);
        let R = &self.sigmas_h - z.insert_axis(Axis(1)).to_owned();
        debug!("Shape of R: {:?}", R.shape());
        debug!("R after\n{:?}", R);

        L.dot(&R.t())
    }

    pub fn update(&mut self, z: ArrayView1<Float>) -> Result<(), Box<dyn Error>> {
        debug!("update");
        for i in 0..self.sigma_points.n_points() {
            let new_column = self.hx.call(self.sigmas_f.column(i))?;
            self.sigmas_h.column_mut(i).assign(&new_column);
        }

        let (z_pred, S) = self.unscented_transform(&self.sigmas_h, Some(&self.R));
        let Pxz = self.cross_variance(self.x.view(), z_pred.view());
        debug!("self.K before:\n{}", &self.K);
        self.K = right_solve_h(&Pxz, S.clone());
        self.S = S;
        debug!("self.S after:\n{}", &self.S);
        debug!("self.K after:\n{}", &self.K);
        debug!("Pxz:\n{}", Pxz);
        debug!("self.K . dot( self.S ):\n{}", self.K.dot(&self.S));

        self.z = z.to_owned();
        self.y = z.to_owned() - z_pred; // TODO: Remove this from the struct

        self.x += &self.K.dot(&self.y);
        debug!(
            "self.P smallest eigenval before: {}",
            smallest_eigenvalue(&self.P)
        );
        self.P -= &self.K.dot(&self.S.dot(&self.K.t()));
        debug!(
            "self.P smallest eigenval after: {}",
            smallest_eigenvalue(&self.P)
        );

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
        dim_x: usize,
        dim_z: usize,
        hx: PythonMeasurementFunction,
        fx: PythonTransitionFunction,
        sigma_points: &PyAny,
    ) -> PyResult<Self> {
        let sigma_points_rust = sigma_points.extract::<SigmaPoints>()?;

        Ok(Self::new(
            dim_x,
            dim_z,
            Box::new(hx),
            Box::new(fx),
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
    fn py_get_x(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "x")]
    fn py_set_x(&mut self, x: PyReadonlyArray1<Float>) -> PyResult<()> {
        self.x = x.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "P")]
    fn py_get_P(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "P")]
    fn py_set_P(&mut self, P: PyReadonlyArray2<Float>) -> PyResult<()> {
        self.P = P.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "x_prior")]
    fn py_get_x_prior(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x_prior.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "P_prior")]
    fn py_get_P_prior(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P_prior.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "x_post")]
    fn py_get_x_post(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.x_post.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "P_post")]
    fn py_get_P_post(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.P_post.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "Q")]
    fn py_get_Q(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.Q.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "Q")]
    fn py_set_Q(&mut self, Q: PyReadonlyArray2<Float>) -> PyResult<()> {
        self.Q = Q.as_array().to_owned();
        Ok(())
    }

    #[getter]
    #[pyo3(name = "R")]
    fn py_get_R(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.R.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[setter]
    #[pyo3(name = "R")]
    fn py_set_R(&mut self, R: PyReadonlyArray2<Float>) -> PyResult<()> {
        self.R = R.as_array().to_owned();
        Ok(())
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

    #[getter]
    #[pyo3(name = "sigmas_f")]
    fn py_get_sigmas_f(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.sigmas_f.clone().into_pyarray(py).to_owned();
        Ok(array)
    }

    #[getter]
    #[pyo3(name = "sigmas_h")]
    fn py_get_sigmas_h(&self, py: Python<'_>) -> PyResult<Py<PyArray2<Float>>> {
        let array = self.sigmas_h.clone().into_pyarray(py).to_owned();
        Ok(array)
    }
}
