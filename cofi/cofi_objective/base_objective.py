from cofi.cofi_objective.base_forward import LinearFittingFwd
from .model_params import Model
from .base_forward import BaseForward

from typing import Protocol, Union
from numbers import Number
import numpy as np


class _ObjectiveCallable(Protocol):
    def __call__(self, *args: Number) -> Number: ...


class BaseObjective:
    """Base class for all problem setup in CoFI.

    All objective problems must be sub-classes of this class and implements two methods:
    1. __init__
    2. misfit()
    """

    def __init__(self, func: _ObjectiveCallable):
        self.objective = func

    def misfit(self, model: Model):
        """
        Misfit value: try to optimise this value by lowering it
        """
        return self.objective(*model.values())

    def jacobian(self, model: Union(Model, np.array)): # TODO (with Jax maybe)
        raise NotImplementedError("This is a TOOD task, or to be implemented by subclasses")

    def gradient(self, model: Union(Model, np.array)):
        raise NotImplementedError("This is a TOOD task, or to be implemented by subclasses")

    def hessian(self, model: Union(Model, np.array)):
        raise NotImplementedError("This is a TOOD task, or to be implemented by subclasses")

    def log_posterior(self, model: Union(Model, np.array)):
        raise NotImplementedError("This is a TOOD task, or to be implemented by subclasses")


class DataBasedObjective(BaseObjective):
    """
    General class holder for objective functions that are calculated from data misfit

    feed the data into constructor, and specify a misfit function
    """

    def __init__(self, X, Y, forward: BaseForward, distance):
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"Numbers of data points don't match between X ({X.shape}) and Y ({Y.shape})")

        self.X = X
        self.Y = Y
        self.forward = forward
        self.n_params = forward.model_dimension()

        if isinstance(forward, LinearFittingFwd):
            self.linear = True

        # distance can be a function or a string
        if isinstance(distance, function):
            self.distance = distance
        elif isinstance(distance, str):
            self.distance_name = distance
            # TODO - define the actual distance functions
            # if distance == 'l2':
            #     pass
            # else:
            #     pass

        # TODO self.objective = ???

    def misfit(self, model: Model):
        if self.distance_name:
            raise NotImplementedError(
                "distance functions specified by str not implemented yet"
            )

        return super().misfit(model)
