#![allow(clippy::excessive_precision)]

extern crate ukf_pyrs;

use ndarray::{arr1, arr2, Array, Array1, Array2, ArrayView1, Axis};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use ukf_pyrs::dynamic_functions::{MeasurementFunction, TransitionFunction};
use ukf_pyrs::functions::pinhole_camera::{CameraProjector, PinholeCamera};
use ukf_pyrs::functions::simple_example::FirstOrderTransitionFunction;
use ukf_pyrs::parallel::UKFParallel;
use ukf_pyrs::sigma_points::SigmaPoints;
use ukf_pyrs::unscented_kalman_filter::UnscentedKalmanFilter;
use ukf_pyrs::Float;

fn create_cam1() -> PinholeCamera {
    let extrinsic_matrix = arr2(&[
        [0., 0., -1., 0.],
        [-0.89442718, 0.44721359, 0., 0.],
        [0.44721359, 0.89442718, 0., 111.80339754],
        [0., 0., 0., 1.],
    ]);

    let intrinsic_matrix = arr2(&[[320., 0., 320.], [0., 240., 240.], [0., 0., 1.]]);

    PinholeCamera::new(extrinsic_matrix, intrinsic_matrix)
}

fn create_cam2() -> PinholeCamera {
    let extrinsic_matrix = arr2(&[
        [0., 0., 1., 0.],
        [0.84799826, 0.52999896, 0., 0.],
        [-0.52999896, 0.84799826, 0., 94.33980882],
        [0., 0., 0., 1.],
    ]);

    let intrinsic_matrix = arr2(&[[320., 0., 320.], [0., 240., 240.], [0., 0., 1.]]);

    PinholeCamera::new(extrinsic_matrix, intrinsic_matrix)
}

fn points_rand(n_points: usize) -> Array2<Float> {
    let point_a = Array::from_vec(vec![-50.0, 0.0, 0.0]);
    let point_b = Array::from_vec(vec![0.0, 120.0, 130.0]);
    let point_c = &Array::from_vec(vec![10.0, -10.0, 10.0]) * 2.0;

    let t: Array1<Float> = Array::linspace(0.0, 1.0, n_points);

    let mut points: Array2<Float> = Array::zeros((n_points, 3));
    points
        .axis_iter_mut(Axis(0))
        .zip(t)
        .for_each(|(mut point, t_val)| {
            point += &(t_val * &(point_b.clone() - &point_a));
            point += &point_a;
            let s_val = (t_val * std::f32::consts::PI).cos().powi(2);
            point += &(s_val * &point_c.clone())
        });

    let mut rng = SmallRng::seed_from_u64(0); // Seed with a fixed value for reproducibility
    let rand_scale = 1.0;
    points.mapv(|x| x + (rng.gen::<f32>() * 2.0 - 1.0) * rand_scale)
}

fn main() {
    let n_points = 10000;
    let dt: Float = 1.0 / (n_points as Float);
    let points = points_rand(n_points);
    let cam1 = create_cam1();
    let cam2 = create_cam2();

    let mut measurement_function = CameraProjector::new(vec![cam1, cam2]);
    let points_proj1: Array2<Float> =
        measurement_function.call_h_batch(points.view()).unwrap();
    measurement_function.index = 1;
    // let points_proj2: Array2<Float> =
    //     measurement_function.call_h_batch(points.view()).unwrap();

    let transition_function = FirstOrderTransitionFunction::new(3);

    let n_filters = 10;
    let mut kalman_filters: Vec<UnscentedKalmanFilter> = (0..n_filters)
        .map(|_| {
            UnscentedKalmanFilter::new(
                6,
                2,
                measurement_function.to_measurement_box(),
                transition_function.to_transition_box(),
                SigmaPoints::merwe(6, 0.5, 2.0, -2.0),
            )
        })
        .collect();

    for ukf in &mut kalman_filters {
        ukf.Q = Array::from_diag(&arr1(&[0.01, 0.01, 0.01, 30.0, 30.0, 30.0]));
    }

    let mut ukf_parallel = UKFParallel::new(kalman_filters);
    for point in points_proj1.axis_iter(Axis(0)) {
        ukf_parallel.predict(dt).unwrap();

        let points_parallel: Vec<ArrayView1<Float>> =
            (0..n_filters).map(|_| point.view()).collect();

        ukf_parallel.update(points_parallel).unwrap();
    }
}
