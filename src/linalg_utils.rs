use crate::Float;
use ndarray::Array2;
use ndarray_linalg::{EigValsh, FactorizeHInto, Inverse, Norm, SolveH};

pub fn right_solve_h(A: &Array2<Float>, B: Array2<Float>) -> Array2<Float> {
    let mut out = Array2::zeros((A.shape()[0], B.shape()[0]));
    let B_factor = B.factorizeh_into().unwrap();
    A.outer_iter()
        .zip(out.rows_mut())
        .for_each(|(row_A, mut row_out)| {
            row_out.assign(&B_factor.solveh(&row_A.to_owned()).unwrap())
        });
    out
}

pub fn right_solve_inverse(A: &Array2<Float>, B: Array2<Float>) -> Array2<Float> {
    let B_inv = B.inv().unwrap();
    A.dot(&B_inv)
}

pub fn smallest_eigenvalue(matrix: &Array2<Float>) -> Float {
    let eigenvalues = matrix.eigvalsh(ndarray_linalg::UPLO::Lower).unwrap();
    eigenvalues[0]
}
pub fn relative_l2_error(a: &Array2<Float>, b: &Array2<Float>) -> Float {
    let norm_a = a.norm_l2();
    let norm_diff = (a - b).norm_l2();
    norm_diff / norm_a
}

pub fn check_symmetric(matrix: &Array2<Float>, tol: Float) -> bool {
    relative_l2_error(matrix, &matrix.t().to_owned()) < tol
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_right_solve_h() {
        let a: Array2<Float> = Array::random((5, 3), rand::distributions::Standard);
        let mut b: Array2<Float> = Array::random((3, 3), rand::distributions::Standard);
        b = b.clone() + b.t();

        let x = right_solve_h(&a, b.clone());

        let relative_error = relative_l2_error(&x.dot(&b), &a);
        println!("{}", relative_error);
        assert!(relative_error < 1e-6);
    }
}
