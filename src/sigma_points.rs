use crate::Float;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use ndarray_linalg::cholesky::{Cholesky, UPLO};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};
use std::{error::Error, sync::Arc};

#[pyclass]
#[derive(Clone)]
pub struct SigmaPointsContainer {
    size: usize,
    Wm: Array1<Float>,
    Wc: Array1<Float>,
}

pub type SigmaPointsGeneratorMethod = Arc<
    dyn Fn(
            &SigmaPointsContainer,
            ArrayView1<Float>,
            ArrayView2<Float>,
        ) -> Result<Array2<Float>, Box<dyn Error>>
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
    ) -> Result<Array2<Float>, Box<dyn Error>> {
        (self.f)(&self.container, x, P)
    }

    pub fn get_Wm(&self) -> ArrayView1<Float> {
        self.container.Wm.view()
    }

    pub fn get_Wc(&self) -> ArrayView1<Float> {
        self.container.Wc.view()
    }

    pub fn n_points(&self) -> usize {
        self.container.size
    }
}

impl SigmaPoints {
    pub fn merwe(
        size: usize,
        lambda: Float,
        alpha: Float,
        beta: Float,
        kappa: Float,
    ) -> Self {
        let n: Float = size as Float;
        let l: Float = alpha * alpha * (n + kappa) - n;
        let c = 0.5 / (n + l);

        let mut Wc: Array1<Float> = Array1::from_elem(size * 2 + 1, c);
        Wc[0] = l / (n + l) + (1.0 - alpha * alpha + beta);
        let mut Wm: Array1<Float> = Array1::from_elem(size * 2 + 1, c);
        Wm[0] = l / (n + l);

        let f_inner = move |container: &SigmaPointsContainer,
                            x: ArrayView1<Float>,
                            P: ArrayView2<Float>|
              -> Result<Array2<Float>, Box<dyn Error>> {
            if x.len() != container.size {
                return Err("Input x of unexpected size".into());
            }
            if P.shape() != [container.size, container.size] {
                return Err("Input P of unexpected size".into());
            }

            let n: Float = container.size as Float;

            let U: Array2<Float> =
                ((lambda + n) * P.to_owned()).cholesky(UPLO::Upper)?;

            let mut sigmas: Array2<Float> =
                Array2::zeros((2 * container.size + 1, container.size));
            sigmas.row_mut(0).assign(&x.to_owned());

            for k in 0..container.size {
                sigmas.row_mut(k + 1).assign(&(x.to_owned() + U.row(k)));
                sigmas
                    .row_mut(container.size + k + 1)
                    .assign(&(x.to_owned() - U.row(k)));
            }
            Ok(sigmas)
        };
        let f: SigmaPointsGeneratorMethod = Arc::new(f_inner);

        SigmaPoints {
            f,
            container: SigmaPointsContainer { size, Wm, Wc },
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::f32::consts::SQRT_2;
    #[test]
    fn test_merwe_sigma_points() {
        let n = 2;

        let sigma_point_gen = SigmaPoints::merwe(n, 1e-4, 1.0, 2.0, 0.0);

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
