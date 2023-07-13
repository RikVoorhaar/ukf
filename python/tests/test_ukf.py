# %%
import pytest
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import filterpy.kalman.UKF
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF_Py
from ukf import UnscentedKalmanFilter, SigmaPoints
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


def relative_error(a: np.ndarray, b: np.ndarray, rtol: float = 1e-4) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + rtol))


def compare_kalman_filters(
    kf_py: UKF_Py,
    kf_rs: UnscentedKalmanFilter,
):
    errors = {
        "x": relative_error(kf_py.x, kf_rs.x),
        "z": relative_error(kf_py.z, kf_rs.z),
        "sigmas_f": relative_error(kf_py.sigmas_f, kf_rs.sigmas_f.T),
        "sigmas_h": relative_error(kf_py.sigmas_h, kf_rs.sigmas_h.T),
        "Q": relative_error(kf_py.Q, kf_rs.Q),
        "R": relative_error(kf_py.R, kf_rs.R),
        "K": relative_error(kf_py.K, kf_rs.K),
        "P": relative_error(kf_py.P, kf_rs.P),
    }
    total_error = sum(errors.values())
    return errors, total_error


class CombinedFilter:
    def __init__(self, ukf_rs: UnscentedKalmanFilter, ukf_py: UKF_Py) -> None:
        self.ukf_rs = ukf_rs
        self.ukf_py = ukf_py

    def update(self, z: np.ndarray) -> None:
        self.ukf_rs.update(z)
        self.ukf_py.update(z)

    def predict(self, dt: float) -> None:
        self.ukf_rs.predict(dt)
        self.ukf_py.predict(dt=dt)

    def total_error(self) -> float:
        _, total_error = compare_kalman_filters(self.ukf_py, self.ukf_rs)
        return total_error

    def all_errors(self) -> dict[str, float]:
        errors, _ = compare_kalman_filters(self.ukf_py, self.ukf_rs)
        return errors



@pytest.mark.parametrize("dim_z", [1, 2, 3])
@pytest.mark.parametrize("alpha", [1, 2, 3])
@pytest.mark.parametrize("beta", [1, 2, 3])
@pytest.mark.parametrize("kappa", [0, 1, 2])
def test_constant_speed_model(dim_z, alpha, beta, kappa):
    dim_x = dim_z * 2

    def hx(x: np.ndarray) -> np.ndarray:
        if x.shape != (dim_x,):
            raise ValueError("x must have shape (dim_x,)")

        return x[:dim_z].astype(np.float32)  # just forget speed

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        if x.shape != (dim_x,):
            raise ValueError("x must have shape (dim_x,)")

        F = np.eye(dim_x)
        F[:dim_z, dim_z:] = dt * np.eye(dim_z)

        return (F @ x).astype(np.float32)

    sigma_points_py = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
    sigma_points_rs = SigmaPoints.merwe(dim_x, alpha, beta, kappa)
    kalman_filter_rs = UnscentedKalmanFilter(dim_x, dim_z, hx, fx, sigma_points_rs)
    kalman_filter_py = filterpy.kalman.UKF.UnscentedKalmanFilter(
        dim_x, dim_z, 1.0, hx, fx, sigma_points_py
    )
    ukf_comb = CombinedFilter(kalman_filter_rs, kalman_filter_py)

    for _ in range(10):
        z = np.random.normal(size=dim_z).astype(np.float32)
        dt = np.random.uniform(0, 2)

        ukf_comb.update(z)
        ukf_comb.predict(dt)

    assert ukf_comb.total_error() < 1e-4


@pytest.mark.parametrize("dim_z", [1, 2, 3])
@pytest.mark.parametrize("dim_x", [1, 2, 3])
@pytest.mark.parametrize("alpha", [1, 2, 3])
@pytest.mark.parametrize("beta", [1, 2, 3])
@pytest.mark.parametrize("kappa", [0, 1, 2])
def test_random_matrix_model(dim_z, dim_x, alpha, beta, kappa):
    np.random.seed(np.mod(hash((dim_z, dim_x, alpha, beta, kappa)), 2**32-1))
    sigma_points = SigmaPoints.merwe(dim_x, alpha, beta, kappa)

    H_mat = np.random.normal(size=(dim_z,dim_x))
    F_mat = np.random.normal(size=(dim_x,dim_x))

    def hx(x: np.ndarray) -> np.ndarray:
        if x.shape != (dim_x,):
            raise ValueError("x must have shape (dim_x,)")
        return (H_mat @ x).astype(np.float32)

    def fx(x: np.ndarray, dt: float) -> np.ndarray:
        if x.shape != (dim_x,):
            raise ValueError("x must have shape (dim_x,)")

        return (F_mat @ x).astype(np.float32)

    kalman_filter_rs = UnscentedKalmanFilter(dim_x, dim_z, hx, fx, sigma_points)
    sigma_points_py = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
    kalman_filter_py = filterpy.kalman.UKF.UnscentedKalmanFilter(
        dim_x, dim_z, 1.0, hx, fx, sigma_points_py
    )
    ukf_comb = CombinedFilter(kalman_filter_rs, kalman_filter_py)

    for _ in range(10):
        z = np.random.normal(size=dim_z).astype(np.float32)
        dt = np.random.uniform(0, 2)

        ukf_comb.update(z)
        ukf_comb.predict(dt)

    assert ukf_comb.total_error() < 1e-3
