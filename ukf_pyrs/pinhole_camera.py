# %%
from functools import cached_property

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

from ukf_pyrs import (
    CameraProjector,
    Rust_PinholeCamera,
)


class PinholeCamera:
    def __init__(
        self,
        rotation_rodrigues: np.ndarray,
        position: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> None:
        self.rvec = -rotation_rodrigues
        self.rmat = cv2.Rodrigues(self.rvec)[0]
        self.tvec = (self.rmat @ position).reshape(3, 1)
        self.intrinsic_matrix = camera_matrix

    def project(self, points: np.ndarray) -> np.ndarray:
        """
        Project a set of 3d points to 2d image coordinates.
        """
        return cv2.projectPoints(
            points, self.rvec, self.tvec, self.intrinsic_matrix, None
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
        camera_matrix = camera_matrix_from_fov_resolution(fov_x_degrees, resolution)

        return cls(
            rotation_rodrigues=rot,
            position=camera_position,
            camera_matrix=camera_matrix,
        )

    @cached_property
    def inverse_extrinsic_matrix(self) -> np.ndarray:
        """
        Get the extrinsic matrix of the camera.
        """
        M_extr = np.eye(4)
        M_extr[:3, :3] = self.rmat.T
        M_extr[:3, 3] = -(self.rmat.T @ self.tvec.squeeze())
        return M_extr

    @cached_property
    def extrinsic_matrix(self) -> np.ndarray:
        """
        Get the inverse extrinsic matrix of the camera.
        """
        M_extr = np.eye(4)
        M_extr[:3, :3] = self.rmat
        M_extr[:3, 3] = self.tvec.squeeze()
        return M_extr

    def world_to_camera(self, points: np.ndarray) -> np.ndarray:
        """
        Transform a set of points from world coordinates to camera coordinates.
        """
        M_extr = self.extrinsic_matrix
        points = points @ M_extr[:3, :3].T + M_extr[:3, 3]
        return points

    def camera_to_screen(self, points: np.ndarray) -> np.ndarray:
        """
        Transform a set of points from camera coordinates to screen coordinates.
        """
        screen_projective = self.intrinsic_matrix @ points.T
        screen_projective /= screen_projective[2, :]
        return np.ascontiguousarray(screen_projective[:2, :].T)

    @cached_property
    def fundamental_matrix_translation(self) -> tuple[np.ndarray, np.ndarray]:
        M_extr = self.extrinsic_matrix
        Mat = (M_extr[:3, :3].T @ self.intrinsic_matrix.T).copy()
        t = (M_extr[:3, 3] @ self.intrinsic_matrix.T).copy()
        return Mat, t

    def world_to_screen_fast(self, points: np.ndarray) -> np.ndarray:
        M, t = self.fundamental_matrix_translation
        points = points @ M + t
        points /= points[:, 2:]
        return points[:, :2]

    def world_to_screen_single(self, point: np.ndarray) -> np.ndarray:
        M, t = self.fundamental_matrix_translation
        point = point @ M + t
        return np.array([point[0] / point[2], point[1] / point[2]], np.float32)

    def world_to_screen(self, points: np.ndarray) -> np.ndarray:
        """
        Transform a set of points from world coordinates to screen coordinates.
        """
        return self.camera_to_screen(self.world_to_camera(points))

    def to_rust(self) -> Rust_PinholeCamera:
        return Rust_PinholeCamera(
            self.extrinsic_matrix.astype(np.float32),
            self.intrinsic_matrix.astype(np.float32),
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
cams = [cam1.to_rust(), cam2.to_rust()]
projector = CameraProjector(cams)
point = np.random.normal(size=3).astype(np.float32)
projector.select_camera(1)
projector(point) - cam2.world_to_screen_single(point)
# %%
