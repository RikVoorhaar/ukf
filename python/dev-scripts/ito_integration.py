# %%
"""I want to verify some identities of martingales / ito stuff numerically"""



import numpy as np
import matplotlib.pyplot as plt

# %%
"""
Is the variance of the ito integral \int_0^t W(s)ds = t^3 / 3?
"""

def sample_wiener_paths(time: float, steps: int, n_paths: int):
    """Sample a wiener path at the given time and number of steps."""
    dt = time / steps
    return np.cumsum(
        np.random.normal(scale=np.sqrt(dt), size=(steps, n_paths)), axis=0
    )

t = 5
n_steps = 1000
n_paths = 10000
dt = t / n_steps
t_values = np.linspace(0, t, n_steps)
paths = sample_wiener_paths(t, n_steps, n_paths)

np.mean(np.trapz(paths, t_values, axis=0))
np.var(np.trapz(paths, t_values, axis=0)) / (t**3 / 3)

