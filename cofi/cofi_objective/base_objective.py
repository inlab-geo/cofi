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
    - jacobian(model: Union[Model, np.ndarray]), the Jacobian of the forward function
    - gradient(model: Union[Model, np.ndarray]), the gradient of the misfit function
    - hessian(model: Union[Model, np.ndarray]), the Hessian of the misfit function
    - residual(model: Union[Model, np.ndarray]), the difference between observed and predicted data
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
            "`gradient` is required in the solving approach but you haven't"
            " implemented it"
        )

    def hessian(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "`hessian` is required in the solving approach but you haven't"
            " implemented it"
        )

    def residual(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "`residual` is required in the solving approach but you haven't"
            " implemented it"
        )

    def jacobian(self, model: Union[Model, np.ndarray]):  # TODO (with Jax maybe)
        raise NotImplementedError(
            "`jacobian` is required in the solving approach but you haven't"
            " implemented it"
        )

    def log_posterior(self, model: Union[Model, np.ndarray]):
        raise NotImplementedError(
            "`log_posterior` is required in the solving approach but you haven't"
            " implemented it"
        )

    def data_x(self):
        raise NotImplementedError(
            "`data_x` is required in the solving approach but you haven't"
            " implemented it"
        )

    def data_y(self):
        raise NotImplementedError(
            "`data_y` is required in the solving approach but you haven't"
            " implemented it"
        )

    def initial_model(self):
        raise NotImplementedError(
            "`initial_model` is required in the solving approach but you haven't"
            " implemented it"
        )

    def n_params(self):
        raise NotImplementedError(
            "`n_params` is required in the solving approach but you haven't"
            " implemented it"
        )


class LeastSquareObjective(BaseObjective):
    """
    General class holder for objective functions that are calculated from data misfit

    Feed the data into constructor, and least squares misfit will be generated automatically
    """

    def __init__(
        self,
        X,
        Y,
        forward: Union[BaseForward, Callable],
        initial_model: Union[Model, np.ndarray, list] = None,
    ):
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(
                f"Numbers of data points don't match between X ({X.shape}) and Y"
                f" ({Y.shape})"
            )

        self.X = X
        self.Y = np.expand_dims(Y, axis=1)
        if not isinstance(forward, BaseForward):
            forward = BaseForward(forward)
        self.forward = forward
        self.nparams = forward.model_dimension()

        if initial_model:
            self.prior = (
                initial_model.values()
                if isinstance(initial_model, Model)
                else np.asanyarray(initial_model)
            )
        else:
            self.prior = np.zeros(self.nparams)

    def misfit(self, model: Union[Model, np.ndarray]):
        # TODO weight to the data? (ref: inversion textbook p31)
        residual = self.residual(model)
        return residual.T @ residual

    def residual(self, model: Union[Model, np.ndarray]):
        if isinstance(model, Model):
            model = model.values()
        X = np.expand_dims(self.X, axis=1)
        model = np.squeeze(model)
        predicted_Y = np.apply_along_axis(self.forward.solve_curried(model), 1, X)
        return np.squeeze(predicted_Y - self.Y)

    def data_x(self):
        return self.X

    def data_y(self):
        return self.Y

    def n_params(self):
        return self.nparams

    def initial_model(self):
        return self.prior

    def params_size(self):
        return len(self.prior)