use crate::Float;
use anyhow::{anyhow, Error};
use log::debug;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::cholesky::{Cholesky, UPLO};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::ops::{Add, AddAssign, Sub, SubAssign};
use std::sync::Arc;

#[pyclass]
#[derive(Clone)]
pub struct SigmaPointsContainer {
    n_points: usize,
    size: usize,
    Wm: Array1<Float>,
    Wc: Array1<Float>,
}

pub type SigmaPointsGeneratorMethod = Arc<
    dyn Fn(
            &SigmaPointsContainer,
            ArrayView1<Float>,
            ArrayView2<Float>,
        ) -> Result<Array2<Float>, Error>
        + Send
        + Sync,
>;

#[pyclass]
#[derive(Clone)]
pub struct SigmaPoints {
    f: SigmaPointsGeneratorMethod,
    container: SigmaPointsContainer,
}

impl SigmaPoints {
    pub fn call(
        &self,
        x: ArrayView1<Float>,
        P: ArrayView2<Float>,
    ) -> Result<Array2<Float>, Error> {
        (self.f)(&self.container, x, P)
    }

    pub fn get_Wm(&self) -> ArrayView1<Float> {
        self.container.Wm.view()
    }

    pub fn get_Wc(&self) -> ArrayView1<Float> {
        self.container.Wc.view()
    }

    pub fn n_points(&self) -> usize {
        self.container.n_points
    }

    pub fn size(&self) -> usize {
        self.container.size
    }
}

impl SigmaPoints {
    pub fn merwe(size: usize, alpha: Float, beta: Float, kappa: Float) -> Self {
        let n: Float = size as Float;
        let lambda: Float = alpha * alpha * (n + kappa) - n;
        let c = 0.5 / (n + lambda);

        let n_points = size * 2 + 1;
        let mut Wc: Array1<Float> = Array1::from_elem(n_points, c);
        Wc[0] = lambda / (n + lambda) + (1.0 - alpha * alpha + beta);
        let mut Wm: Array1<Float> = Array1::from_elem(n_points, c);
        Wm[0] = lambda / (n + lambda);

        let f_inner = move |container: &SigmaPointsContainer,
                            x: ArrayView1<Float>,
                            P: ArrayView2<Float>|
              -> Result<Array2<Float>, Error> {
            if x.len() != container.size {
                return Err(anyhow!("Input x of unexpected size"));
            }
            if P.shape() != [container.size, container.size] {
                return Err(anyhow!("Input P of unexpected size"));
            }

            let n: Float = container.size as Float;

            let U: Array2<Float> = (lambda + n) * P.to_owned();
            let U: Array2<Float> = (U)
                .cholesky(UPLO::Upper)
                .map_err(|_| anyhow!("Cholesky failed"))?;

            let x_owned: Array1<Float> = x.to_owned();
            let repeated_x = x_owned
                .broadcast((container.n_points, container.size))
                .unwrap();
            let mut sigmas = repeated_x.to_owned();

            sigmas
                .slice_mut(s![1..container.size + 1, ..])
                .add_assign(&U);

            sigmas
                .slice_mut(s![container.size + 1..container.n_points, ..])
                .sub_assign(&U);

            Ok(sigmas)
        };
        let f: SigmaPointsGeneratorMethod = Arc::new(f_inner);

        SigmaPoints {
            f,
            container: SigmaPointsContainer {
                n_points,
                size,
                Wm,
                Wc,
            },
        }
    }
}

#[pymethods]
impl SigmaPoints {
    #[pyo3(name = "__call__")]
    pub fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        P: PyReadonlyArray2<Float>,
    ) -> PyResult<Py<PyArray2<Float>>> {
        let x = x.as_array();
        let P = P.as_array();
        let sigmas = self.call(x, P).map_err(|err| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
        })?;
        let py_array = sigmas.into_pyarray(py).to_owned();
        Ok(py_array)
    }

    #[getter]
    #[pyo3(name = "wm")]
    pub fn py_get_wm(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.container.Wm.clone();
        let py_array = array.into_pyarray(py).to_owned();
        Ok(py_array)
    }

    #[getter]
    #[pyo3(name = "wc")]
    pub fn py_get_wc(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.container.Wc.clone();
        let py_array = array.into_pyarray(py).to_owned();
        Ok(py_array)
    }

    #[staticmethod]
    #[pyo3(name = "merwe")]
    pub fn py_merwe(
        n: usize,
        alpha: Float,
        beta: Float,
        kappa: Float,
    ) -> PyResult<Self> {
        let sigma_point_gen = SigmaPoints::merwe(n, alpha, beta, kappa);
        Ok(sigma_point_gen)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f32::consts::SQRT_2;
    #[test]
    fn test_merwe_sigma_points() {
        let n = 2;

        let sigma_point_gen = SigmaPoints::merwe(n, 1.0, 2.0, 0.0);

        let x = Array1::from_iter(0..n).map(|x| *x as Float);
        let P = Array2::from_diag(&Array1::from_iter((1..=n).map(|x| x as Float)));

        let sigma = sigma_point_gen.call(x.view(), P.view()).unwrap();

        let expected_sigma = array![
            [0., 1.],
            [SQRT_2 as Float, 1.],
            [0., 3.],
            [(-SQRT_2) as Float, 1.],
            [0., -1.]
        ];

        assert!(sigma.abs_diff_eq(&expected_sigma, 1e-4));

        let expected_Wc = array![2., 0.25, 0.25, 0.25, 0.25];
        assert!(sigma_point_gen.get_Wc().abs_diff_eq(&expected_Wc, 1e-4));

        let expected_Wm = array![0., 0.25, 0.25, 0.25, 0.25];
        assert!(sigma_point_gen.get_Wm().abs_diff_eq(&expected_Wm, 1e-4));
    }
}
