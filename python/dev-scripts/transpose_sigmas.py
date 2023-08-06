from ukf_pyrs import UKF, SigmaPoints, measurement_function, transition_function
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG)

dim_z = 2
dim_x = dim_z * 2
alpha, beta, kappa = (1, 1, 0)

@measurement_function(dim_z)
def hx(x: np.ndarray) -> np.ndarray:
    if x.shape != (dim_x,):
        raise ValueError("x must have shape (dim_x,)")

    return x[:dim_z].astype(np.float32)  # just forget speed

@transition_function
def fx(x: np.ndarray, dt: float) -> np.ndarray:
    if x.shape != (dim_x,):
        raise ValueError(f"x has shape {x.shape}, expected shape {(dim_x,)=}")

    F = np.eye(dim_x)
    F[:dim_z, dim_z:] = dt * np.eye(dim_z)

    return (F @ x).astype(np.float32)


sigma_points_rs = SigmaPoints.merwe(dim_x, alpha, beta, kappa)
ukf = UKF(
    dim_x, dim_z, hx, fx, sigma_points_rs
)
for _ in range(10):
    z = np.random.normal(size=dim_z).astype(np.float32)
    dt = np.random.uniform(0, 2)

    ukf.update(z)
    ukf.predict(dt)