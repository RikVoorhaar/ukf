# %%
import os
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
import filterpy.kalman.UKF
from ukf import UnscentedKalmanFilter, SigmaPoints
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)


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


kalman_filter = UnscentedKalmanFilter(dim_x, dim_z, hx, fx, sigma_points)

dir(kalman_filter)
kalman_filter.update(np.random.normal(size=dim_z).astype(np.float32))
kalman_filter.predict(1)
kalman_filter.update(np.random.normal(size=dim_z).astype(np.float32))
kalman_filter.predict(1)
print(kalman_filter.x, kalman_filter.x_post, kalman_filter.x_prior)
print(kalman_filter.P, kalman_filter.P_post, kalman_filter.P_prior)
# %%
U = np.array(
    [
        [-2044.7402, 0.48989728, 15.818511, 2.6613257e-8],
        [0.4898973, 8.017494, -0.0034020653, 1.5625],
        [15.81851, -0.0034020646, 6.4634995, -1.8481427e-10],
        [2.6613256e-8, 1.5625, -1.848143e-10, 6.5625],
    ]
)
np.linalg.eigvalsh(kalman_filter.P)
# %%
kalman_filter.cross_variance(kalman_filter.x, kalman_filter.z)
# %%
sigma_points_py = MerweScaledSigmaPoints(dim_x, 1, 2, 0)
kalman_filter_py = filterpy.kalman.UKF.UnscentedKalmanFilter(
    dim_x, dim_z, 1.0, hx, fx, sigma_points_py
)

cross_var_py = kalman_filter_py.cross_variance(
    kalman_filter.x, kalman_filter.z, kalman_filter.sigmas_f.T, kalman_filter.sigmas_h.T
)
cross_var_rust = kalman_filter.cross_variance(kalman_filter.x, kalman_filter.z)

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
    kalman_filter.x,
    kalman_filter.z,
    kalman_filter.sigmas_f.T,
    kalman_filter.sigmas_h.T,
    kalman_filter.sigma_points.wc,
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
    kalman_filter.x,
    kalman_filter.z,
    kalman_filter.sigmas_f.T,
    kalman_filter.sigmas_h.T,
    kalman_filter.sigma_points.wc,
)
# %%
# Todo: now cross variance works correctly. I still need to verify if the rest of the code
# works as expected.