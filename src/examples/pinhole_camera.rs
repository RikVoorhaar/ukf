use crate::Float;
use ndarray::{arr1, s, Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

use crate::dynamic_functions::{MeasurementFunction, MeasurementFunctionBox};
use pyo3::prelude::*;
use pyo3::{pyclass, pymethods};

#[pyclass]
#[derive(Clone)]
pub struct PinholeCamera {
    pub extrinsic: Array2<Float>,
    pub intrinsic: Array2<Float>,
}

impl PinholeCamera {
    pub fn new(extrinsic: Array2<Float>, intrinsic: Array2<Float>) -> Self {
        Self {
            extrinsic,
            intrinsic,
        }
    }
}

impl PinholeCamera {
    pub fn rotation_matrix(&self) -> ArrayView2<Float> {
        self.extrinsic.slice(s![..3, ..3])
    }

    pub fn translation_vector(&self) -> ArrayView1<Float> {
        self.extrinsic.slice(s![..3, 3])
    }

    pub fn inverse_extrinsic(&self) -> Array2<Float> {
        let mut inverse_extrinsic = Array2::zeros((4, 4));
        inverse_extrinsic
            .slice_mut(s![..3, ..3])
            .assign(&self.rotation_matrix().t());
        inverse_extrinsic
            .column_mut(3)
            .assign(&(-self.rotation_matrix().t().dot(&self.translation_vector())));
        inverse_extrinsic[(3, 3)] = 1.0;
        inverse_extrinsic
    }

    pub fn fundamental_matrix(&self) -> Array2<Float> {
        self.intrinsic.dot(&self.rotation_matrix()).t().to_owned()
    }

    pub fn fundamental_translation_vector(&self) -> Array1<Float> {
        self.intrinsic.dot(&self.translation_vector()).to_owned()
    }
}

#[pymethods]
impl PinholeCamera {
    #[new]
    fn py_new(
        extrinsic: PyReadonlyArray2<Float>,
        intrinsic: PyReadonlyArray2<Float>,
    ) -> PyResult<Self> {
        Ok(Self {
            extrinsic: extrinsic.to_owned_array(),
            intrinsic: intrinsic.to_owned_array(),
        })
    }
}

#[pyclass]
#[derive(Clone)]
pub struct CameraProjector {
    pub index: usize,
    pub fundamental_matrices: Array3<Float>,
    pub translation_vectors: Array2<Float>,
}

impl CameraProjector {
    pub fn new(cameras: Vec<PinholeCamera>) -> Self {
        let num_cameras = cameras.len();
        let mut fundamental_matrices: Array3<Float> =
            Array3::zeros((num_cameras, 3, 3));
        fundamental_matrices
            .axis_iter_mut(Axis(0))
            .zip(cameras.iter())
            .for_each(|(mut f, camera)| {
                f.assign(&camera.fundamental_matrix());
            });

        let mut translation_vectors: Array2<Float> = Array2::zeros((num_cameras, 3));
        translation_vectors
            .axis_iter_mut(Axis(0))
            .zip(cameras.iter())
            .for_each(|(mut t, camera)| {
                t.assign(&camera.fundamental_translation_vector());
            });

        CameraProjector {
            index: 0,
            fundamental_matrices,
            translation_vectors,
        }
    }

    pub fn size(&self) -> usize {
        self.fundamental_matrices.len_of(Axis(0))
    }
}

impl MeasurementFunction for CameraProjector {
    fn call_h(&self, x: ArrayView1<Float>) -> PyResult<Array1<Float>> {
        let x = x.slice(s![..3]).to_owned();
        let x = x.dot(&self.fundamental_matrices.slice(s![self.index, .., ..]));
        let x = x + self.translation_vectors.slice(s![self.index, ..]);
        Ok(arr1(&[x[0] / x[2], x[1] / x[2]]))
    }

    fn to_measurement_box(&self) -> MeasurementFunctionBox {
        let h = self.clone();
        MeasurementFunctionBox { h: Box::new(h) }
    }

    fn update_py_context(&mut self, py: Python<'_>, context: PyObject) -> PyResult<()> {
        let index: usize = context.extract(py)?;
        if index >= self.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range",
                index
            )));
        }
        self.index = index;
        Ok(())
    }
}

#[pymethods]
impl CameraProjector {
    #[new]
    pub fn py_new(cam_list: Vec<&PyAny>) -> PyResult<Self> {
        let cam_vec: Vec<PinholeCamera> = cam_list
            .iter()
            .map(|cam| cam.extract::<PinholeCamera>())
            .collect::<PyResult<Vec<_>>>()?;

        Ok(Self::new(cam_vec))
    }

    pub fn select_camera(&mut self, index: usize) -> PyResult<()> {
        if index >= self.size() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Index {} out of range",
                index
            )));
        }
        self.index = index;
        Ok(())
    }

    #[pyo3(name = "__call__")]
    pub fn py_call(
        &self,
        py: Python<'_>,
        x: PyReadonlyArray1<Float>,
    ) -> PyResult<Py<PyArray1<Float>>> {
        self.call_h(x.as_array())
            .map(|result| result.into_pyarray(py).to_owned())
    }

    #[pyo3(name = "to_measurement_box")]
    pub fn py_to_measurement_box(&self) -> PyResult<MeasurementFunctionBox> {
        Ok(self.to_measurement_box())
    }
}
