# %%
from ukf_pyrs import MeasurementFunction, TransitionFunction
import numpy as np


def f(x: np.ndarray, dt: float, context=None) -> np.ndarray:
    if context is not None:
        print(context)
    else:
        print("no context")
    return x * 2 + dt


def h(x: np.ndarray, context=None) -> np.ndarray:
    if context is not None:
        print(context)
    else:
        print("no context")
    return x * 2


pmf = MeasurementFunction(h, 3)
ptf = TransitionFunction(f, 3)
# pmf.context = "omg"

pmf(np.array([1, 2, 3], dtype=np.float32))
ptf(np.array([1, 2, 3], dtype=np.float32), -0.1)

# %%


def measurement_function(arg=None):
    if callable(arg):
        func = arg
        return MeasurementFunction(func)

    def decorator(func):
        return MeasurementFunction(func, arg)

    return decorator


@measurement_function("context")
def h_m(x: np.ndarray, context=None) -> np.ndarray:
    print(context)
    return x


h_m(np.arange(2, dtype=np.float32))
