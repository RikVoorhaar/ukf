use ndarray::{Array1, Array2};
use ndarray_linalg::cholesky::{Cholesky, UPLO};
use std::error::Error;

pub trait SigmaPoints {
    fn get_Wm(&self) -> &Array1<f32>;
    fn get_Wc(&self) -> &Array1<f32>;
    fn sigma_points(
        &self,
        x: Array1<f32>,
        P: Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn Error>>;
}

pub struct MerweSigmaPoints {
    size: usize,
    lambda: f32,
    Wm: Array1<f32>,
    Wc: Array1<f32>,
}

impl MerweSigmaPoints {
    pub fn new(size: usize, alpha: f32, beta: f32, kappa: f32) -> Self {
        let n: f32 = size as f32;
        let lambda: f32 = alpha * alpha * (n + kappa) - n;
        let c = 0.5 / (n + lambda);

        let mut Wc: Array1<f32> = Array1::from_elem(size * 2 + 1, c);
        Wc[0] = lambda / (n + lambda) + (1.0 - alpha * alpha + beta);
        let mut Wm: Array1<f32> = Array1::from_elem(size * 2 + 1, c);
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
    fn get_Wm(&self) -> &Array1<f32> {
        &self.Wm
    }

    fn get_Wc(&self) -> &Array1<f32> {
        &self.Wc
    }

    fn sigma_points(
        &self,
        x: Array1<f32>,
        P: Array2<f32>,
    ) -> Result<Array2<f32>, Box<dyn Error>> {
        if x.len() != self.size {
            return Err("Input x of unexpected size".into());
        }
        if P.shape() != [self.size, self.size] {
            return Err("Input P of unexpected size".into());
        }

        let n: f32 = self.size as f32;

        let U: Array2<f32> = ((self.lambda + n) * P).cholesky(UPLO::Upper)?;
        let mut sigmas: Array2<f32> = Array2::zeros((2 * self.size + 1, self.size));
        sigmas.row_mut(0).assign(&x);

        for k in 0..self.size {
            sigmas.row_mut(k + 1).assign(&(&x + &U.row(k)));
            sigmas.row_mut(self.size + k + 1).assign(&(&x - &U.row(k)));
        }
        Ok(sigmas)
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

        let x = Array1::from_iter(0..n).map(|x| *x as f32);
        let P = Array2::from_diag(&Array1::from_iter((1..=n).map(|x| x as f32)));

        let sigma = sigma_point_gen.sigma_points(x, P).unwrap();

        let expected_sigma =
            array![[0., 1.], [SQRT_2, 1.], [0., 3.], [SQRT_2, 1.], [0., -1.]];
        assert!(sigma.abs_diff_eq(&expected_sigma, 1e-4));

        let expected_Wc = array![2., 0.25, 0.25, 0.25, 0.25];
        assert!(sigma_point_gen.get_Wc().abs_diff_eq(&expected_Wc, 1e-4));

        let expected_Wm = array![0., 0.25, 0.25, 0.25, 0.25];
        assert!(sigma_point_gen.get_Wm().abs_diff_eq(&expected_Wm, 1e-4));
    }
}
