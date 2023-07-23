# %%
from ukf_pyrs import measurement_function, transition_function
import numpy as np


# %%


@transition_function
def f(x: np.ndarray, dt: float, context=None) -> np.ndarray:
    if context is not None:
        print(context)
    else:
        print("no context")
    return x * 2 + dt


@measurement_function("maybe")
def h(x: np.ndarray, context=None) -> np.ndarray:
    if context is not None:
        print(context)
    else:
        print("no context")
    return x * 2


f(np.arange(2, dtype=np.float32), 0.1)
h(np.arange(2, dtype=np.float32))

# %%
