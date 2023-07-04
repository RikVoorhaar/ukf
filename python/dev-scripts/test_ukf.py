# %%
import os
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import filterpy.kalman.UKF
from filterpy.kalman.unscented_transform import unscented_transform
from ukf import UnscentedKalmanFilter, SigmaPoints
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)


dim_x = 4  # dimension of state space
dim_z = 2  # dimension of measurement space

sigma_points = SigmaPoints.merwe(dim_x, 1, 2, 0)


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


def compare_kalman_filters(
    kf_py: filterpy.kalman.UKF.UnscentedKalmanFilter,
    kf_rs: UnscentedKalmanFilter,
):
    errors = {
        "x": np.linalg.norm(kf_py.x - kf_rs.x),
        "z": np.linalg.norm(kf_py.z - kf_rs.z),
        "sigmas_f": np.linalg.norm(kf_py.sigmas_f - kf_rs.sigmas_f.T),
        "sigmas_h": np.linalg.norm(kf_py.sigmas_h - kf_rs.sigmas_h.T),
        "Q": np.linalg.norm(kf_py.Q - kf_rs.Q),
        "R": np.linalg.norm(kf_py.R - kf_rs.R),
        "K": np.linalg.norm(kf_py.K - kf_rs.K),
        "P": np.linalg.norm(kf_py.P - kf_rs.P),
    }
    total_error = sum(errors.values())
    return errors, total_error


kalman_filter_rs = UnscentedKalmanFilter(dim_x, dim_z, hx, fx, sigma_points)
dir(kalman_filter_rs)
random_z1 = np.random.normal(size=dim_z).astype(np.float32)
random_z2 = np.random.normal(size=dim_z).astype(np.float32)

kalman_filter_rs.update(random_z1)
kalman_filter_rs.predict(1)
print(kalman_filter_rs.x)
kalman_filter_rs.update(random_z2)
kalman_filter_rs.predict(1)
print(kalman_filter_rs.x)
kalman_filter_rs.P


# %%
sigma_points_py = MerweScaledSigmaPoints(dim_x, 1, 2, 0)
kalman_filter_py = filterpy.kalman.UKF.UnscentedKalmanFilter(
    dim_x, dim_z, 1.0, hx, fx, sigma_points_py
)
kalman_filter_py.update(random_z1)
kalman_filter_py.predict(dt=1.0)
print(kalman_filter_py.x)
kalman_filter_py.update(random_z2)
kalman_filter_py.predict(dt=1.0)
print(kalman_filter_py.x)


# %%
ut_rs = kalman_filter_rs.unscented_transform(
    kalman_filter_rs.sigmas_f, kalman_filter_rs.Q
)
ut_py = unscented_transform(
    kalman_filter_rs.sigmas_f.T,
    kalman_filter_rs.sigma_points.wm,
    kalman_filter_rs.sigma_points.wc,
    kalman_filter_rs.Q,
)
ut_rs, ut_py

cv_rs = kalman_filter_rs.cross_variance(kalman_filter_rs.x, kalman_filter_rs.z)
cv_py = kalman_filter_py.cross_variance(
    kalman_filter_rs.x,
    kalman_filter_rs.z,
    kalman_filter_rs.sigmas_f.T,
    kalman_filter_rs.sigmas_h.T,
)
np.linalg.norm(cv_py - cv_rs)

# %%
U = np.array(
    [
        [-2044.7402, 0.48989728, 15.818511, 2.6613257e-8],
        [0.4898973, 8.017494, -0.0034020653, 1.5625],
        [15.81851, -0.0034020646, 6.4634995, -1.8481427e-10],
        [2.6613256e-8, 1.5625, -1.848143e-10, 6.5625],
    ]
)
np.linalg.eigvalsh(kalman_filter_rs.P)
# %%
kalman_filter_rs.cross_variance(kalman_filter_rs.x, kalman_filter_rs.z)
# %%

cross_var_py = kalman_filter_py.cross_variance(
    kalman_filter_rs.x,
    kalman_filter_rs.z,
    kalman_filter_rs.sigmas_f.T,
    kalman_filter_rs.sigmas_h.T,
)
cross_var_rust = kalman_filter_rs.cross_variance(kalman_filter_rs.x, kalman_filter_rs.z)

x = np.random.normal(size=dim_x).astype(np.float32)
P = np.random.normal(size=(dim_x, dim_x)).astype(np.float32)
P = P @ P.T

np.linalg.norm(sigma_points_py.sigma_points(x, P) - sigma_points(x, P))
# %%
print(np.linalg.norm(cross_var_py - cross_var_rust))
# %%


def cross_var1(x, z, sigmas_f, sigmas_h, Wc):
    """
    Compute cross variance of the state `x` and measurement `z`.
    """

    Pxz = np.zeros((sigmas_f.shape[1], sigmas_h.shape[1]))
    N = sigmas_f.shape[0]
    for i in range(N):
        dx = sigmas_f[i] - x
        dz = sigmas_h[i] - z
        Pxz += Wc[i] * np.outer(dx, dz)
    return Pxz


cv1 = cross_var1(
    kalman_filter_rs.x,
    kalman_filter_rs.z,
    kalman_filter_rs.sigmas_f.T,
    kalman_filter_rs.sigmas_h.T,
    kalman_filter_rs.sigma_points.wc,
)


def cross_var2(x, z, sigmas_f, sigmas_h, Wc):
    L = sigmas_f - x

    print("L before\n", L)
    for k in range(len(Wc)):
        L[k] *= Wc[k]
    R = sigmas_h - z
    print("L after\n", L)
    print("0-")
    print(L.shape)
    print(R.shape)
    return L.T @ R


cv2 = cross_var2(
    kalman_filter_rs.x,
    kalman_filter_rs.z,
    kalman_filter_rs.sigmas_f.T,
    kalman_filter_rs.sigmas_h.T,
    kalman_filter_rs.sigma_points.wc,
)
# %%
# Todo: now cross variance works correctly. I still need to verify if the rest of the code
# works as expected.
