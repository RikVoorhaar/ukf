# %%
"""
Our test problem. We have a skeleton consisting of n_keypoints.

These are all somewhere in space, for now random is ok. 

They are then observed by a calibrated camera, and mapped to a 2d screen. For 
this we can just make up an intrinsic and extrinsic matrix; it doesn't really
matter. Although it would be nice to have a method that can create them from an
fov value. 
"""
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
)
from ukf_pyrs.pinhole_camera import CameraProjector, PinholeCamera

point_a = np.array([-50, 0, 0])
point_b = np.array([0, 120, 130])
point_c = np.array([10, -10, 10]) * 2
t = np.linspace(0, 1, 50)
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
cam1.project(points).reshape(-1, 2) - cam1.world_to_screen(points)
print(
    np.linalg.norm(
        cam1.world_to_screen_fast(points) - cam1.project(points).reshape(-1, 2)
    )
)

# %%
plt.plot(*proj_points_obs1.T, ".")
plt.plot(*proj_points1.T, ".-")
plt.show()

proj_points_obs2 = cam2.project(np.ascontiguousarray(points_rand[1::2])).reshape(-1, 2)
proj_points2 = cam2.project(points).reshape(-1, 2)
plt.plot(*proj_points_obs2.T, ".")
plt.plot(*proj_points2.T, ".-")
plt.show()
# %%

dim_x = 3
dim_z = 2


@measurement_function
def hx(x: np.ndarray, cam_id: int) -> np.ndarray:
    if cam_id == 0:
        return cam1.world_to_screen_single(x)
    elif cam_id == 1:
        return cam2.world_to_screen_single(x)
    else:
        return np.zeros(2, dtype=np.float32)


@transition_function
def fx(x: np.ndarray, dt: float) -> np.ndarray:
    return x


sigma_points = SigmaPoints.merwe(dim_x, 0.5, 2, 0)
kalman_filter = UKF(dim_x, dim_z, hx, fx, sigma_points)

kalman_filter.Q *= 1e2
kalman_filter.R *= 1
kalman_filter.P *= 1000
predictions_list = []
for p1, p2 in zip(
    proj_points_obs1.astype(np.float32), proj_points_obs2.astype(np.float32)
):
    hx.context = 0
    kalman_filter.update(p1)
    kalman_filter.predict(dt)
    predictions_list.append(kalman_filter.x)

    hx.context = 1
    kalman_filter.update(p2)
    kalman_filter.predict(dt)
    predictions_list.append(kalman_filter.x)

predictions = np.array(predictions_list)

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.title("Camera 1 reprojected tracking")
plt.plot(*cam1.project(np.array(predictions)).reshape(-1, 2).T, ".-", label="Predicted")
plt.plot(*proj_points_obs1.T, ".", label="Observed")
plt.plot(*proj_points1.T, "-", label="True")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Camera 2 reprojected tracking")
plt.plot(*cam2.project(np.array(predictions)).reshape(-1, 2).T, ".-", label="Predicted")
plt.plot(*proj_points_obs2.T, ".", label="Observed")
plt.plot(*proj_points2.T, "-", label="True")
plt.legend()

plt.subplot(2, 2, 3)
plt.title("Tracking error")
tracking_errors = np.linalg.norm(points - predictions, axis=1)
tracking_errors_rand = np.linalg.norm(points_rand - predictions, axis=1)
plt.plot(
    tracking_errors,
    label="Unperturbed",
)
plt.plot(
    tracking_errors_rand,
    label="Perturbed",
)
plt.ylim(0, np.mean(tracking_errors) * 2)
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Tracking error (smoothened)")
sigma = 5
plt.plot(
    gaussian_filter1d(tracking_errors, sigma),
    label="Unperturbed",
)
plt.plot(
    gaussian_filter1d(tracking_errors_rand, sigma),
    label="Perturbed",
)
plt.ylim(0, np.mean(tracking_errors) * 2)
plt.legend()
plt.show()


# %%
"""Now let's do a first order model"""

dim_x = 6
dim_z = 2

float_type = np.float32


@measurement_function
def hx_first_order(x: np.ndarray, cam_id: int) -> np.ndarray:
    pos = x[:3]
    if cam_id == 0:
        return cam1.world_to_screen_single(pos)
    elif cam_id == 1:
        return cam2.world_to_screen_single(pos)
    else:
        return np.zeros(2, dtype=float_type)


@transition_function
def fx_first_order(x: np.ndarray, dt: float) -> np.ndarray:
    pos = x[:3]
    vel = x[3:]
    return np.concatenate([pos + vel * dt, vel])


sigma_points = SigmaPoints.merwe(dim_x, 0.5, 2, -2)
hx_rust = CameraProjector([cam1.to_rust(), cam2.to_rust()])

fx_rust = FirstOrderTransitionFunction(3)
kalman_filter = UKF(dim_x, dim_z, hx_rust, fx_rust, sigma_points)
# kalman_filter = UKF(dim_x, dim_z, hx_first_order, fx_first_order, sigma_points)


kalman_filter.Q = np.diag([1] * 3 + [3e3] * 3).astype(float_type)
kalman_filter.Q *= 1e-2
kalman_filter.R = np.diag([1, 1]).astype(float_type) * 1e0
kalman_filter.P = np.diag([1e0] * 3 + [1e0] * 3).astype(float_type)
predictions_list = []
for p1, p2 in zip(
    proj_points_obs1.astype(float_type), proj_points_obs2.astype(float_type)
):
    # hx_first_order.context = 0
    hx_rust.select_camera(0)
    kalman_filter.predict(dt)
    kalman_filter.update(p1)
    predictions_list.append(kalman_filter.x)

    # hx_first_order.context = 1
    hx_rust.select_camera(1)
    kalman_filter.predict(dt)
    kalman_filter.update(p2)
    predictions_list.append(kalman_filter.x)

predictions = np.array(predictions_list)  # type: ignore
pos_predictions = predictions[:, :3]

plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.title("Camera 1 reprojected tracking")
plt.plot(
    *cam1.project(np.array(pos_predictions)).reshape(-1, 2).T, ".-", label="Predicted"
)
plt.plot(*proj_points_obs1.T, ".", label="Observed")
plt.plot(*proj_points1.T, "-", label="True")
plt.legend()

plt.subplot(2, 2, 2)
plt.title("Camera 2 reprojected tracking")
plt.plot(
    *cam2.project(np.array(pos_predictions)).reshape(-1, 2).T, ".-", label="Predicted"
)
plt.plot(*proj_points_obs2.T, ".", label="Observed")
plt.plot(*proj_points2.T, "-", label="True")
plt.legend()


plt.subplot(2, 2, 3)
plt.title("Tracking error (smoothened)")
sigma = 5
tracking_errors = np.linalg.norm(points - pos_predictions, axis=1)
tracking_errors_rand = np.linalg.norm(points_rand - pos_predictions, axis=1)
plt.plot(
    gaussian_filter1d(tracking_errors, sigma),
    label="Unperturbed",
)
plt.plot(
    gaussian_filter1d(tracking_errors_rand, sigma),
    label="Perturbed",
)
plt.ylim(0, np.mean(tracking_errors) * 2)
plt.legend()

plt.subplot(2, 2, 4)
plt.title("Tracking of velocity")
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
names = ["x", "y", "z"]
for i, v in enumerate(predictions[:, 3:].T):
    plt.plot(v, color=colors[i], label=names[i])

velocity = np.diff(points, axis=0) / dt
for i, v in enumerate(velocity.T):
    plt.plot(v, color=colors[i], linestyle="--")
plt.legend()
plt.show()

# %%
"""Now let's look at the Rust version. Is it much faster?"""


def time_python_version():
    sigma_points = SigmaPoints.merwe(dim_x, 0.5, 2, -2)
    kalman_filter = UKF(dim_x, dim_z, hx_first_order, fx_first_order, sigma_points)

    kalman_filter.Q = np.diag([1] * 3 + [3e3] * 3).astype(float_type)
    kalman_filter.Q *= 1e-2
    kalman_filter.R = np.diag([1, 1]).astype(float_type) * 1e0
    kalman_filter.P = np.diag([1e0] * 3 + [1e0] * 3).astype(float_type)
    predictions_list = []
    for p1, p2 in zip(
        proj_points_obs1.astype(float_type), proj_points_obs2.astype(float_type)
    ):
        hx_first_order.context = 0
        kalman_filter.predict(dt)
        kalman_filter.update(p1)
        predictions_list.append(kalman_filter.x)

        hx_first_order.context = 1
        kalman_filter.predict(dt)
        kalman_filter.update(p2)
        predictions_list.append(kalman_filter.x)


time_before = perf_counter()
for _ in range(100):
    time_python_version()
time_after = perf_counter()
print(f"Python version took {time_after - time_before:.3f}s")
# %%


fx_rust(np.arange(6, dtype=np.float32), 3)
hx_rust(np.arange(6, dtype=np.float32))


def time_rust_version():
    hx_rust = CameraProjector([cam1.to_rust(), cam2.to_rust()])
    fx_rust = FirstOrderTransitionFunction(3)
    sigma_points = SigmaPoints.merwe(dim_x, 0.5, 2, -2)
    kalman_filter = UKF(dim_x, dim_z, hx_rust, fx_rust, sigma_points)

    kalman_filter.Q = np.diag([1] * 3 + [3e3] * 3).astype(float_type)
    kalman_filter.Q *= 1e-2
    kalman_filter.R = np.diag([1, 1]).astype(float_type) * 1e0
    kalman_filter.P = np.diag([1e0] * 3 + [1e0] * 3).astype(float_type)
    predictions_list = []
    for p1, p2 in zip(
        proj_points_obs1.astype(float_type), proj_points_obs2.astype(float_type)
    ):
        kalman_filter.predict(dt)
        kalman_filter.update(p1)
        predictions_list.append(kalman_filter.x)

        kalman_filter.predict(dt)
        kalman_filter.update(p2)
        predictions_list.append(kalman_filter.x)


time_before = perf_counter()
for _ in range(100):
    time_rust_version()
time_after = perf_counter()
print(f"Rust version took {time_after - time_before:.3f}s")
