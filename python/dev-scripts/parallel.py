# %%
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ukf_pyrs import (
    UKF,
    FirstOrderTransitionFunction,
    SigmaPoints,
    measurement_function,
    transition_function,
    UKFParallel,
)
from ukf_pyrs.pinhole_camera import CameraProjector, PinholeCamera

N_points = 5000
point_a = np.array([-50, 0, 0])
point_b = np.array([0, 120, 130])
point_c = np.array([10, -10, 10]) * 2
t = np.linspace(0, 1, N_points)
dt = t[1] - t[0]
points = (1 - t)[:, None] * point_a[None, :] + t[:, None] * point_b[None, :]
points += (np.cos(t * np.pi) ** 2)[:, None] * point_c[None, :]
points = points.astype(np.float32)
np.random.seed(0)
rand_scale = 1
points_rand = (points + np.random.randn(*points.shape) * rand_scale).astype(np.float32)

cam1 = PinholeCamera.from_params(
    camera_position=np.array([50, 100, 0]),
    lookat_target=np.array([0, 0, 0]),
    fov_x_degrees=90,
    resolution=np.array([640, 480]),
)
cam2 = PinholeCamera.from_params(
    camera_position=np.array([-50, 80, 0]),
    lookat_target=np.array([0, 0, 0]),
    fov_x_degrees=90,
    resolution=np.array([640, 480]),
)
proj_points_obs1 = cam1.project(np.ascontiguousarray(points_rand[0::2])).reshape(-1, 2)
proj_points1 = cam1.project(points).reshape(-1, 2)
proj_points_obs2 = cam2.project(np.ascontiguousarray(points_rand[1::2])).reshape(-1, 2)
proj_points2 = cam2.project(points).reshape(-1, 2)

# %%


cam1 = PinholeCamera.from_params(
    camera_position=np.array([50, 100, 0]),
    lookat_target=np.array([0, 0, 0]),
    fov_x_degrees=90,
    resolution=np.array([640, 480]),
)
cam2 = PinholeCamera.from_params(
    camera_position=np.array([-50, 80, 0]),
    lookat_target=np.array([0, 0, 0]),
    fov_x_degrees=90,
    resolution=np.array([640, 480]),
)
hx_rust = CameraProjector([cam1.to_rust(), cam2.to_rust()])
fx_rust = FirstOrderTransitionFunction(3)
sigma_points = SigmaPoints.merwe(6, 0.5, 2, -2)

kalman_filters = []
n_filters = 10
for _ in range(n_filters):
    kalman_filter = UKF(6, 2, hx_rust, fx_rust, sigma_points)
    kalman_filter.Q = np.diag([1] * 3 + [3e3] * 3).astype(np.float32)
    kalman_filter.Q *= 1e-2
    kalman_filter.R = np.diag([1, 1]).astype(np.float32) * 1e0
    kalman_filter.P = np.diag([1e0] * 3 + [1e0] * 3).astype(np.float32)
    kalman_filters.append(kalman_filter)

parallel_ukf = UKFParallel(kalman_filters)


time_begin = perf_counter()
predictions_list = []
for p1, p2 in zip(
    proj_points_obs1.astype(np.float32), proj_points_obs2.astype(np.float32)
):
    parallel_ukf.update_measurement_context([0]*n_filters)
    parallel_ukf.predict(dt)

    parallel_ukf.update([p1 + i for i in range(n_filters)])
    predictions_list.append(parallel_ukf.x)

    # parallel_ukf.update_measurement_context([1]*n_filters)
    # hx_rust.select_camera(1)
    # parallel_ukf.predict(dt)
    # parallel_ukf.update([p2 + i for i in range(n_filters)])
    # predictions_list.append(parallel_ukf.x)
time_end = perf_counter()

print(f"Time elapsed: {time_end - time_begin:0.4f}s")
