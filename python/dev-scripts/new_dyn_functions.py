# %%
from ukf import PythonMeasurementFunction
import numpy as np


def f(x: np.ndarray, context=None) -> np.ndarray:
    if context is not None:
        print(context)
    else:
        print("no context")
    return x * 2


pmf = PythonMeasurementFunction(f)
pmf.context = "omg"

pmf(np.array([1, 2, 3], dtype=np.float32))

# %%
