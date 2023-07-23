from ukf_pyrs import MeasurementFunction, TransitionFunction


def measurement_function(arg=None):
    if callable(arg):
        func = arg
        return MeasurementFunction(func)

    def decorator(func):
        return MeasurementFunction(func, arg)

    return decorator


def transition_function(arg=None):
    if callable(arg):
        func = arg
        return TransitionFunction(func)

    def decorator(func):
        return TransitionFunction(func, arg)

    return decorator
