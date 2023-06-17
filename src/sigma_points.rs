use crate::Float;
use ndarray::{Array1, Array2};
use ndarray_linalg::cholesky::{Cholesky, UPLO};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::error::Error;

pub trait SigmaPoints {
    fn get_Wm(&self) -> &Array1<Float>;
    fn get_Wc(&self) -> &Array1<Float>;
    fn sigma_points(
        &self,
        x: &Array1<Float>,
        P: &Array2<Float>,
    ) -> Result<Array2<Float>, Box<dyn Error>>;

    fn n_points(&self) -> usize;
}

#[pyclass]
pub struct MerweSigmaPoints {
    size: usize,
    lambda: Float,
    Wm: Array1<Float>,
    Wc: Array1<Float>,
}

impl MerweSigmaPoints {
    pub fn new(size: usize, alpha: Float, beta: Float, kappa: Float) -> Self {
        let n: Float = size as Float;
        let lambda: Float = alpha * alpha * (n + kappa) - n;
        let c = 0.5 / (n + lambda);

        let mut Wc: Array1<Float> = Array1::from_elem(size * 2 + 1, c);
        Wc[0] = lambda / (n + lambda) + (1.0 - alpha * alpha + beta);
        let mut Wm: Array1<Float> = Array1::from_elem(size * 2 + 1, c);
        Wm[0] = lambda / (n + lambda);

        MerweSigmaPoints {
            size,
            lambda,
            Wm,
            Wc,
        }
    }
}

impl SigmaPoints for MerweSigmaPoints {
    fn get_Wm(&self) -> &Array1<Float> {
        &self.Wm
    }

    fn get_Wc(&self) -> &Array1<Float> {
        &self.Wc
    }

    fn sigma_points(
        &self,
        x: &Array1<Float>,
        P: &Array2<Float>,
    ) -> Result<Array2<Float>, Box<dyn Error>> {
        if x.len() != self.size {
            return Err("Input x of unexpected size".into());
        }
        if P.shape() != [self.size, self.size] {
            return Err("Input P of unexpected size".into());
        }

        let n: Float = self.size as Float;

        let U: Array2<Float> = ((self.lambda + n) * P).cholesky(UPLO::Upper)?;
        let mut sigmas: Array2<Float> = Array2::zeros((2 * self.size + 1, self.size));
        sigmas.row_mut(0).assign(x);

        for k in 0..self.size {
            sigmas.row_mut(k + 1).assign(&(x + &U.row(k)));
            sigmas.row_mut(self.size + k + 1).assign(&(x - &U.row(k)));
        }
        Ok(sigmas)
    }

    fn n_points(&self) -> usize {
        self.size * 2 + 1
    }
}
#[pymethods]
impl MerweSigmaPoints {
    #[new(text_signature = "(size, alpha, beta, kappa)")]
    #[doc = "Initializes a new MerweSigmaPoints object with given size, alpha, beta, and kappa values."]
    #[pyo3(signature = (size, alpha, beta, kappa))]
    fn py_new(size: usize, alpha: Float, beta: Float, kappa: Float) -> Self {
        MerweSigmaPoints::new(size, alpha, beta, kappa)
    }

    #[getter]
    #[pyo3(name = "Wm")]
    fn py_get_Wm(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.get_Wm().clone();
        let py_array = array.into_pyarray(py).to_owned();
        Ok(py_array)
    }

    #[getter]
    #[pyo3(name = "Wc")]
    fn py_get_Wc(&self, py: Python<'_>) -> PyResult<Py<PyArray1<Float>>> {
        let array = self.get_Wc().clone();
        let py_array = array.into_pyarray(py).to_owned();
        Ok(py_array)
    }

    #[pyo3(name = "sigma_points")]
    fn py_sigma_points(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
        P: PyReadonlyArray2<Float>,
    ) -> PyResult<Py<PyArray2<Float>>> {
        let x = &x.as_array().to_owned();
        let P = &P.as_array().to_owned();
        let sigmas = self.sigma_points(x, P).map_err(|err| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!("{}", err))
        })?;
        let py_array = sigmas.into_pyarray(py).to_owned();
        Ok(py_array)
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

        let sigma_point_gen = MerweSigmaPoints::new(n, 1.0, 2.0, 0.0);

        let x = Array1::from_iter(0..n).map(|x| *x as Float);
        let P = Array2::from_diag(&Array1::from_iter((1..=n).map(|x| x as Float)));

        let sigma = sigma_point_gen.sigma_points(&x, &P).unwrap();

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
