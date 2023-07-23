# %%
"""
Our test problem. We have a skeleton consisting of n_keypoints.

These are all somewhere in space, for now random is ok. 

They are then observed by a calibrated camera, and mapped to a 2d screen. For 
this we can just make up an intrinsic and extrinsic matrix; it doesn't really
matter. Although it would be nice to have a method that can create them from an
fov value. 
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.common import Q_discrete_white_noise
from scipy.ndimage import gaussian_filter1d
from ukf import UnscentedKalmanFilter, SigmaPoints, MeasurementFunction


class PerspectiveCamera:
    def __init__(
        self,
        rotation_rodrigues: np.ndarray,
        position: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        self.rvec = -rotation_rodrigues
        self.rmat = cv2.Rodrigues(self.rvec)[0]
        self.tvec = (self.rmat @ position).reshape(3, 1)
        self.camera_matrix = camera_matrix

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Project a set of 3d points to 2d image coordinates.
        """
        return cv2.projectPoints(
            points, self.rvec, self.tvec, self.camera_matrix, None
        )[0]

    @classmethod
    def from_params(
        cls,
        camera_position: np.ndarray,
        lookat_target,
        fov_x_degrees,
        resolution,
        up=None,
    ):
        if up is None:
            up = np.array([0, 1, 0], dtype=np.float32)

        up = np.array([0, 1, 0], dtype=np.float32)
        rot = rotation_from_lookat(camera_position, lookat_target, up)
        camera_matrix = camera_matrix_from_fov_resolution(
            fov_x_degrees, np.array([640, 480])
        )

        return cls(
            rotation_rodrigues=rot,
            position=camera_position,
            camera_matrix=camera_matrix,
        )


def camera_matrix_from_fov_resolution(
    fov_degrees: float, resolution: np.ndarray
) -> np.ndarray:
    """
    Create a camera matrix from a field of view and a resolution.
    """
    aspect_ratio = resolution[0] / resolution[1]  # width / height
    fov = np.deg2rad(fov_degrees)
    fx = resolution[0] / (2 * np.tan(fov / 2))
    fy = fx / aspect_ratio
    return np.array([[fx, 0, resolution[0] / 2], [0, fy, resolution[1] / 2], [0, 0, 1]])


def rotation_from_lookat(
    position: np.ndarray, target: np.ndarray, up: np.ndarray
) -> np.ndarray:
    """
    Create a rotation matrix from a position, target and up vector.
    """
    z = (position - target) / np.linalg.norm(position - target)
    x = np.cross(up, z) / np.linalg.norm(np.cross(up, z))
    y = np.cross(z, x)
    rot_mat = np.stack([x, y, z], axis=1).astype(np.float32)
    return cv2.Rodrigues(rot_mat)[0]


point_a = np.array([-50, 0, 0])
point_b = np.array([0, 120, 130])
point_c = np.array([10, -10, 10]) * 2
t = np.linspace(0, 1, 50)
dt = t[1] - t[0]
points = (1 - t)[:, None] * point_a[None, :] + t[:, None] * point_b[None, :]
points += (np.cos(t * 2*np.pi) ** 2)[:, None] * point_c[None, :]
points = points.astype(np.float32)
np.random.seed(0)
rand_scale = 1
points_rand = (points + np.random.randn(*points.shape) * rand_scale).astype(np.float32)

cam1 = PerspectiveCamera.from_params(
    camera_position=np.array([50, 100, 0]),
    lookat_target=np.array([0, 0, 0]),
    fov_x_degrees=90,
    resolution=np.array([640, 480]),
)
cam2 = PerspectiveCamera.from_params(
    camera_position=np.array([-50, 80, 0]),
    lookat_target=np.array([0, 0, 0]),
    fov_x_degrees=90,
    resolution=np.array([640, 480]),
)

proj_points_obs1 = cam1.project(np.ascontiguousarray(points_rand[0::2])).reshape(-1, 2)
proj_points1 = cam1.project(points).reshape(-1, 2)
plt.plot(*proj_points_obs1.T, ".")
plt.plot(*proj_points1.T, ".-")
plt.show()

proj_points_obs2 = cam2.project(np.ascontiguousarray(points_rand[1::2])).reshape(-1, 2)
proj_points2 = cam2.project(points).reshape(-1, 2)
plt.plot(*proj_points_obs2.T, ".")
plt.plot(*proj_points2.T, ".-")
plt.show()
# %%
# %%

dim_x = 3
dim_z = 2


def hx(x: np.ndarray, cam_id: int) -> np.ndarray:
    if cam_id == 0:
        return cam1.project(x.reshape(1, 3)).reshape(2)
    elif cam_id == 1:
        return cam2.project(x.reshape(1, 3)).reshape(2)
    else:
        return np.zeros(2, dtype=np.float32)


h = MeasurementFunction(hx, [])


def fx(x: np.ndarray, dt: float) -> np.ndarray:
    return x


sigma_points = SigmaPoints.merwe(dim_x, 0.5, 2, 0)
kalman_filter = UnscentedKalmanFilter(dim_x, dim_z, h, fx, sigma_points)

kalman_filter.Q *= 1e2
kalman_filter.R *= 1
kalman_filter.P *= 1000
predictions_list = []
for p1, p2 in zip(
    proj_points_obs1.astype(np.float32), proj_points_obs2.astype(np.float32)
):
    h.context = 0
    kalman_filter.update(p1)
    kalman_filter.predict(dt)
    predictions_list.append(kalman_filter.x)

    h.context = 1
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


contexts = []


def hx_first_order(x: np.ndarray, cam_id: int) -> np.ndarray:
    contexts.append(cam_id)
    x = x.astype(float_type)
    pos = x[:3].reshape(1, 3)
    if cam_id == 0:
        return cam1.project(pos).reshape(2).astype(float_type)
    elif cam_id == 1:
        return cam2.project(pos).reshape(2).astype(float_type)
    else:
        return np.zeros(2, dtype=float_type)


h_first_order = MeasurementFunction(hx_first_order, 0)


def fx_first_order(x: np.ndarray, dt: float) -> np.ndarray:
    pos = x[:3]
    vel = x[3:]
    return np.concatenate([pos + vel * dt, vel])


sigma_points = SigmaPoints.merwe(dim_x, 0.5, 2, -2)
kalman_filter = UnscentedKalmanFilter(
    dim_x, dim_z, h_first_order, fx_first_order, sigma_points
)


# kalman_filter.Q = Q_discrete_white_noise(
#     2, dt=dt, var=1e6, block_size=3, order_by_dim=False
# ).astype(float_type)
kalman_filter.Q = np.diag([1] * 3 + [3e3] * 3).astype(float_type)
kalman_filter.Q *= 1e-2
kalman_filter.R = np.diag([1, 1]).astype(float_type) * 1e0
kalman_filter.P = np.diag([1e0] * 3 + [1e0] * 3).astype(float_type)
predictions_list = []
for p1, p2 in zip(
    proj_points_obs1.astype(float_type), proj_points_obs2.astype(float_type)
):
    h_first_order.context = 0
    kalman_filter.predict(dt)
    kalman_filter.update(p1)
    predictions_list.append(kalman_filter.x)

    h_first_order.context = 1
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
plt.title("Tracking error (velocity)")
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

kalman_filter.Q = Q_discrete_white_noise(
    2, dt=dt, var=1, block_size=3, order_by_dim=False
).astype(np.float32)
plt.matshow(kalman_filter.Q)
