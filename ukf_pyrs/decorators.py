from ukf_pyrs import MeasurementFunction, TransitionFunction
from typing import Any


def measurement_function(output_dim: int, context: Any | None = None):
    # if callable(arg):
    #     func = arg
    #     return MeasurementFunction(func)

    def decorator(func):
        return MeasurementFunction(func, output_dim, context)

    return decorator


def transition_function(arg=None):
    if callable(arg):
        func = arg
        return TransitionFunction(func)

    def decorator(func):
        return TransitionFunction(func, arg)

    return decorator
