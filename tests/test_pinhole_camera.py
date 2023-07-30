# %%
from ukf_pyrs.pinhole_camera import PinholeCamera, CameraProjector
import numpy as np
import pytest

# %%


@pytest.mark.parametrize(
    "cam1_position", [np.array([50, 100, 0]), np.array([-30, -10, 12])]
)
@pytest.mark.parametrize(
    "cam2_position", [np.array([-50, 80, 0]), np.array([-100, 20, 40])]
)
@pytest.mark.parametrize(
    "lookat_target1", [np.array([0, 0, 0]), np.array([10, 20, 30])]
)
@pytest.mark.parametrize(
    "lookat_target2", [np.array([0, 0, 0]), np.array([-5, -30, 50])]
)
@pytest.mark.parametrize("fov_x_degrees1", [90, 45])
@pytest.mark.parametrize("fov_x_degrees2", [90, 80])
def test_camera_projector_project(
    cam1_position,
    cam2_position,
    lookat_target1,
    lookat_target2,
    fov_x_degrees1,
    fov_x_degrees2,
):
    cam1 = PinholeCamera.from_params(
        camera_position=cam1_position,
        lookat_target=lookat_target1,
        fov_x_degrees=fov_x_degrees1,
        resolution=np.array([640, 480]),
    )
    cam2 = PinholeCamera.from_params(
        camera_position=cam2_position,
        lookat_target=lookat_target2,
        fov_x_degrees=fov_x_degrees2,
        resolution=np.array([640, 480]),
    )
    cams = [cam1.to_rust(), cam2.to_rust()]

    num_fails = 0
    num_fails = 0
    projector = CameraProjector(cams)
    rtol = 1e-4
    for _ in range(100):
        point = np.random.normal(size=3).astype(np.float32)
        projector.select_camera(0)
        if np.linalg.norm(projector(point) - cam1.world_to_screen_single(point)) > rtol:
            num_fails += 1
        projector.select_camera(1)
        if np.linalg.norm(projector(point) - cam2.world_to_screen_single(point)) > rtol:
            num_fails += 1

    assert num_fails < 10


def test_camera_projector_project_multiple():
    cam1 = PinholeCamera.from_params(
        camera_position=np.array([50, 100, 0]),
        lookat_target=np.array([0, 0, 0]),
        fov_x_degrees=90,
        resolution=np.array([640, 480]),
    )

    N = 20
    for _ in range(100):
        points = np.random.normal(size=(N, 3)).astype(np.float32)
        assert (
            np.linalg.norm(
                cam1.project(points).reshape(-1, 2) - cam1.world_to_screen_fast(points)
            )
            / N
            < 1e-4
        )
