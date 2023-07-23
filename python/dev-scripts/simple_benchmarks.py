# %%
from filterpy.kalman.sigma_points import MerweScaledSigmaPoints
from filterpy.kalman.UKF import UnscentedKalmanFilter as UKF_Py
from ukf_pyrs import UnscentedKalmanFilter, SigmaPoints, constant_speed_ukf
from numba import jit
import numpy as np
from time import perf_counter

dim_z = 10
alpha = 1
beta = 2
kappa = 0
dim_x = dim_z * 2


def hx(x: np.ndarray) -> np.ndarray:
    return x[:dim_z]

F = np.eye(dim_x, dtype=np.float32)
def fx(x: np.ndarray, dt: float) -> np.ndarray:
    F[:dim_z, dim_z:] = dt * np.eye(dim_z)
    return F @ x

def fx_fast(x: np.ndarray, dt: float) -> np.ndarray:
    new_x = x.copy()
    new_x[:dim_z] += dt * x[dim_z:]
    return new_x


sigma_points_py = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
sigma_points_rs = SigmaPoints.merwe(dim_x, alpha, beta, kappa)
kalman_filter_rs = UnscentedKalmanFilter(dim_x, dim_z, hx, fx_fast, sigma_points_rs)
kalman_filter_py = UKF_Py(dim_x, dim_z, 1.0, hx, fx_fast, sigma_points_py)


N = 1000
z_vals = np.random.randn(N, dim_z).astype(np.float32)
dt_vals = np.random.uniform(0, 2, size=N).astype(np.float32)

t0 = perf_counter()
for z,dt in zip(z_vals, dt_vals):
    kalman_filter_rs.update(z)
    kalman_filter_rs.predict(dt)
t1 = perf_counter()

print(f"RS: {t1 - t0}")

t0 = perf_counter()
for z,dt in zip(z_vals, dt_vals):
    kalman_filter_py.update(z)
    kalman_filter_py.predict(dt)
t1 = perf_counter()

print(f"filterpy: {t1 - t0}")
# %%
kalman_filter_rs2 = constant_speed_ukf(dim_z, sigma_points_rs)
t0 = perf_counter()
for z,dt in zip(z_vals, dt_vals):
    kalman_filter_rs2.update(z)
    kalman_filter_rs2.predict(dt)
t1 = perf_counter()

print(f"pure rs: {t1 - t0}")

# %%
%%prun 

for z,dt in zip(z_vals, dt_vals):
    kalman_filter_rs.update(z)
    kalman_filter_rs.predict(dt)

# %%
@jit(nopython=True)
def hx_jit(x: np.ndarray) -> np.ndarray:
    return x[:dim_z]

@jit(nopython=True)
def fx_jit(x: np.ndarray, dt: float) -> np.ndarray:
    F = np.eye(dim_x, dtype=np.float32)
    F[:dim_z, dim_z:] = dt * np.eye(dim_z)
    return F @ x


sigma_points_py = MerweScaledSigmaPoints(dim_x, alpha, beta, kappa)
sigma_points_rs = SigmaPoints.merwe(dim_x, alpha, beta, kappa)
kalman_filter_rs = UnscentedKalmanFilter(dim_x, dim_z, hx_jit, fx_jit, sigma_points_rs)
kalman_filter_py = UKF_Py(dim_x, dim_z, 1.0, hx_jit, fx_jit, sigma_points_py)


N = 100
z_vals = np.random.randn(N, dim_z).astype(np.float32)

t0 = perf_counter()
for z in z_vals:
    kalman_filter_rs.update(z)
    kalman_filter_rs.predict(1.0)
t1 = perf_counter()

print(f"RS: {t1 - t0}")

t0 = perf_counter()
for z in z_vals:
    kalman_filter_py.update(z)
    kalman_filter_py.predict(1.0)
t1 = perf_counter()

print(f"filterpy: {t1 - t0}")

# %%
fx_jit(np.random.randn(dim_x).astype(np.float32), 1.0)