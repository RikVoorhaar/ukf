# %%
import os
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
# %%
"""
It now runs update and predict without error! Very good news. Next step is to compare it
to filterpy and verify that it actually does what it's supposed to do.
"""