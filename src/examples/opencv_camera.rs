use crate::Float;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use opencv::calib3d::rodrigues;
use opencv::core::no_array;
use opencv::prelude::*;
use opencv::core::{Vector, Mat};

pub struct PerspectiveCamera {
    rvec: Vector<Float>,
    tvec: Vector<Float>,
    camera_matrix: Mat,
}

impl PerspectiveCamera {
    pub fn new(
        rotation_rodrigures: Array1<Float>,
        position: Array1<Float>,
        camera_matrix: Array2<Float>,
    ) -> Self {
        let mut rvec: Vector<Float> = Vector::default();
        rvec.push(rotation_rodrigures[0]);
        rvec.push(rotation_rodrigures[1]);
        rvec.push(rotation_rodrigures[2]);


        let mut rmat: Mat;
        rodrigues(&rvec, &mut rmat, &mut no_array());
        let tvec = rmat * position; // How to multiply vecotr and mat? I think we first have to convert position to a vector or a mat
        PerspectiveCamera {
            rvec,
            tvec,
            camera_matrix,
        }
    }
}
