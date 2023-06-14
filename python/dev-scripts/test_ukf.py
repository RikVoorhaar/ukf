# %%
from ukf import MerweSigmaPoints
import numpy as np

n = 2

sigma_points = MerweSigmaPoints(2, alpha=1, beta=2, kappa=0)

sigma_points.Wm

x = np.arange(n).astype(np.float32)  # np.random.normal(size=n)
P = np.diag(np.arange(1, n + 1)).astype(np.float32)
sigma_points.sigma_points(x, P)