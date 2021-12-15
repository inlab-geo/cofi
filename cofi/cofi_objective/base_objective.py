from . import Model
from .base_forward import BaseForward, LinearFittingFwd

from typing import Callable, Protocol, Union
from numbers import Number
import numpy as np


class _ObjectiveCallable(Protocol):
    def __call__(self, *args: Number) -> Number:
        ...


class BaseObjective:
    """Base class for all problem setup in CoFI.

    All objective problems must be sub-classes of this class and implements two methods:
    1. __init__
    2. misfit(model: Union[Model, np.ndarray]) -> Number

    Optional implementations (depending on solvers):
    - jacobian(model: Union[Model, np.ndarray])
    - gradient(model: Union[Model, np.ndarray])
    - hessian(model: Union[Model, np.ndarray])
    - residuals(model: Union[Model, np.ndarray])
    - log_posterior(model: Union[Model, np.ndarray])
    - data_x()
    - data_y()
    - initial_model()
    - n_params()
    """

    def __init__(self, func: _ObjectiveCallable):
        self._objective = func

    def misfit(self, model: Model):
        """
        Misfit value: try to optimise this value by lowering it
        """
        return self._objective(*model.values())

    def gradient(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "This is a TOOD task, or to be implemented by subclasses"
        )

    def hessian(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "This is a TOOD task, or to be implemented by subclasses"
        )

    def residual(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "This is a TOOD task, or to be implemented by subclasses"
        )

    def jacobian(self, model: Union[Model, np.ndarray]):  # TODO (with Jax maybe)
        raise NotImplementedError(
            "This is a TOOD task, or to be implemented by subclasses"
        )

    def log_posterior(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "This is a TOOD task, or to be implemented by subclasses"
        )


class LeastSquareObjective(BaseObjective):
    """
    General class holder for objective functions that are calculated from data misfit

    Feed the data into constructor, and least squares misfit will be generated automatically
    """

    def __init__(self, X, Y, forward: Union[BaseForward, Callable]):
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Numbers of data points don't match between X ({X.shape}) and Y"
                f" ({Y.shape})"
            )

        self.X = X
        self.Y = Y
        if not isinstance(forward, BaseForward):
            forward = BaseForward(forward)
        self.forward = forward
        self.n_params = forward.model_dimension()

        if isinstance(forward, LinearFittingFwd):
            self.linear = True

    def misfit(self, model: Model):
        residual = self.residual(model)
        return residual @ residual

    def residual(self, model: Model):
        predicted_Y = np.apply_along_axis(self.forward.solve_curried(model), 1, self.X)
        return predicted_Y - self.Y
