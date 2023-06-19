use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_linalg::{FactorizeHInto, SolveH};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::error::Error;

use crate::sigma_points::{MerweSigmaPoints, SigmaPoints};
use crate::Float;

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
    sigma_points: Box<dyn SigmaPoints + Send>,
    pub dim_x: usize,
    pub dim_z: usize,
    hx: Box<dyn Fn(ArrayView1<Float>) -> PyResult<Array1<Float>> + Send>,
    fx: Box<dyn Fn(ArrayView1<Float>, Float) -> PyResult<Array1<Float>> + Send>,
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
        hx: Box<dyn Fn(ArrayView1<Float>) -> PyResult<Array1<Float>> + Send>,
        fx: Box<dyn Fn(ArrayView1<Float>, Float) -> PyResult<Array1<Float>> + Send>,
        sigma_points: Box<dyn SigmaPoints + Send>,
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
        let x: Array1<Float> = self.sigma_points.get_Wm().dot(sigmas);
        let y = sigmas - x.clone().insert_axis(Axis(0));
        let wc_diag = Array2::from_diag(self.sigma_points.get_Wc());
        let mut P = y.t().dot(&wc_diag).dot(&y);
        // FIXME: the diagonal matrix here is inefficient.

        if let Some(nc) = noise_cov {
            P += nc;
        }
        (x, P)
    }

    fn compute_process_sigmas(&mut self, dt: Float) -> Result<(), Box<dyn Error>> {
        let sigmas = self.sigma_points.sigma_points(&self.x, &self.P)?;

        for (row_sigmas, mut row_sigmas_f) in
            sigmas.outer_iter().zip(self.sigmas_f.rows_mut())
        {
            let result = (self.fx)(row_sigmas, dt)?;
            row_sigmas_f.assign(&result);
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
        let mut L = &self.sigmas_f - x.insert_axis(Axis(0)).to_owned();
        for k in 0..self.sigma_points.n_points() {
            L.row_mut(k).mapv_inplace(|v| v * Wc[k]);
        }
        let R = &self.sigmas_h - z.insert_axis(Axis(0)).to_owned();

        L.t().dot(&R)
    }

    pub fn update(&mut self, z: Array1<Float>) -> Result<(), Box<dyn Error>> {
        for i in 0..self.sigma_points.n_points() {
            let new_row = (self.hx)(self.sigmas_f.row(i))?;
            self.sigmas_h.row_mut(i).assign(&new_row);
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

        self.z = z.clone();
        self.y = z - z_pred;

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
        py: Python,
    ) -> PyResult<Self> {
        let hx_rust = |x: ArrayView1<Float>| -> PyResult<Array1<Float>> {
            let x_py = x.to_owned().into_pyarray(py);
            let result_py: &PyArray1<Float> = hx.call1(py, (x_py,))?.downcast(py)?;
            let result = result_py.to_owned_array();
            Ok(result)
        };

        let fx_rust = |x: ArrayView1<Float>, dt: Float| -> PyResult<Array1<Float>> {
            let x_py = x.to_owned().into_pyarray(py);
            let result_py: &PyArray1<Float> = fx.call1(py, (x_py, dt))?.downcast(py)?;
            let result = result_py.to_owned_array();
            Ok(result)
        };

        let sigma_points_rust = sigma_points.extract::<MerweSigmaPoints>()?;

        Ok(Self::new(dim_x, dim_z, hx_rust, fx_rust, sigma_points))
    }
}

// TODO: Expose to Python and write unit tests.
// Passing fx and hx from Python to Rust can be slow, so we should think how to do that
// efficiently. But more than that, we should also allow for factories for useful
// functions like f, h that are just matrix multiplication...
//
// What would be really cool is to take some kind of framework that can compile into
// LLVM, and then pass that over to rust, so it can be called with 0 overhead?

// So we need to rethink this a little. How are we actually going to use this? We will
// probably be in a situation where the functions hx, fx are constant. Maybe it's OK
// to have some kind of factory that outputs rust functions or closures? Let's think
// what kind of things we would need anyway. Maybe there are some good examples of
// these functions in filterpy. Let's write them down.

// No that's to complicated. I think that it's simply more realistic to define the
// problem in rust and export it. It would be cool to have a 'slow' mode with pure
// Python, but I'm not sure that's worth it. I think a realistic use case is to assume
// that measurements are coming in in Python, and we just want to process them real
// fast. Also we will have a lot of parallel observations (like with a skeleton
// tracker), so we can pass a fairly big array to rust and process it in parallel.

// What we need to do is to make a Python wrapper for UnscentedKalmanFilter. This is going
// to be a struct with a single object. We can then implement ::new methods in specific use cases.

// On the other hand, we should allow for closures over function pointers anyway in our
// design, and returing PyResult should be a zero-cost abstraction. So maybe we should
// just do that.

// NEW TODO:
// Refactor so that the ::new() command needs a copy of Wm and Wc; perhaps we can make
// a 'SigmaPointsContainer' struct that just defines that.
// The only place that we actually need sigma_points.sigma_points is in 'compute_process_sigmas', in the predict method.
// We should then go for a function approach. MerweSigmaPoints is actually stateless,
// So we can pass in a closure instead. This would have signature
// Fn(SigmaPointsContainer, ArrayView1<Float>, ArrayView2<Float>) -> Result<Array2<Float>, Box<dyn Error>>
