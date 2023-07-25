use ndarray::{s, Array1, ArrayView1};
use pyo3::prelude::*;
use std::sync::Arc;

use crate::unscented_kalman_filter::{
    HybridMeasurementFunction, HybridTransitionFunction, RustMeasurementFunction,
    RustTranstionFunction, UnscentedKalmanFilter,
};
use crate::Float;
use crate::SigmaPoints;
#[pyfunction]
pub fn constant_speed_ukf(
    dim_z: usize,
    sigma_points: SigmaPoints,
) -> PyResult<UnscentedKalmanFilter> {
    let dim_x = dim_z * 2;
    let fx: RustTranstionFunction = Arc::new(
        move |x: ArrayView1<Float>, dt: Float| -> PyResult<Array1<Float>> {
            let mut new_x = x.to_owned();
            for i in 0..dim_z {
                new_x[i] += dt * x[i + dim_z];
            }
            
            Ok(new_x)
        },
    );
    let fx: HybridTransitionFunction = HybridTransitionFunction::Rust(fx);

    let hx: RustMeasurementFunction =
        Arc::new(move |x: ArrayView1<Float>| -> PyResult<Array1<Float>> {
            Ok(x.slice(s![..dim_z]).to_owned())
        });
    let hx: HybridMeasurementFunction = HybridMeasurementFunction::Rust(hx);

    Ok(UnscentedKalmanFilter::new(
        dim_x,
        dim_z,
        hx,
        fx,
        sigma_points,
    ))
}
