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


point_a = np.array([0, 0, 0])
point_b = np.array([0, 120, 130])
t = np.linspace(0, 1, 100)
points = (1 - t)[:, None] * point_a[None, :] + t[:, None] * point_b[None, :]
np.random.seed(0)
points_rand = points + np.random.randn(*points.shape) * 3

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

proj_points_obs1 = cam1.project(points_rand).reshape(-1, 2)
proj_points1 = cam1.project(points).reshape(-1, 2)
plt.plot(*proj_points_obs1.T, ".")
plt.plot(*proj_points1.T, ".-")
plt.show()

proj_points_obs2 = cam2.project(points_rand).reshape(-1, 2)
proj_points2 = cam2.project(points).reshape(-1, 2)
plt.plot(*proj_points_obs2.T, ".")
plt.plot(*proj_points2.T, ".-")
plt.show()

# %%

dim_x = 3
dim_z = 2


contexts = []


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


sigma_points = SigmaPoints.merwe(dim_x, 1, 2, 0)
kalman_filter = UnscentedKalmanFilter(dim_x, dim_z, h, fx, sigma_points)

predictions = []
for p1, p2 in zip(
    proj_points_obs1.astype(np.float32), proj_points_obs2.astype(np.float32)
):
    h.context = 0
    kalman_filter.update(p1)
    kalman_filter.predict(1.0)
    predictions.append(kalman_filter.x)

    h.context = 1
    kalman_filter.update(p2)
    kalman_filter.predict(1.0)
    predictions.append(kalman_filter.x)

# %%
plt.plot(*cam1.project(np.array(predictions)).reshape(-1, 2).T, ".", label="pred")
plt.plot(*proj_points_obs1.T, ".", label="obs")
plt.plot(*proj_points1.T, ".-", label="true")
plt.legend()
plt.show()

plt.plot(*cam2.project(np.array(predictions)).reshape(-1, 2).T, ".", label="pred")
plt.plot(*proj_points_obs2.T, ".", label="obs")
plt.plot(*proj_points2.T, ".-", label="true")
plt.legend()
plt.show()
