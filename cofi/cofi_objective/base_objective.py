from . import Model
from .base_forward import BaseForward, LinearFittingFwd

from typing import Callable, Union
from numbers import Number
import numpy as np


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
    - params_size()
    """

    def __init__(self, func: Callable[[np.ndarray], Number]=None):
        self._objective = func

    def misfit(self, model: Union[Model, np.ndarray]):
        """
        Misfit value: try to optimise this value by lowering it
        """
        model = np.asanyarray(model.values() if isinstance(model, Model) else model)
        return self._objective(model)

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

    def params_size(self):
        raise NotImplementedError(
            "`params_size` is required in the solving approach but you haven't"
            " implemented it"
        )

    def setMisfit(self, misfit_func: Callable[[Union[Model, np.ndarray]], Number]):
        self.misfit = misfit_func

    def setGradient(
        self, gradient_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.gradient = gradient_func

    def setHessian(
        self, hessian_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.hessian = hessian_func

    def setResidual(
        self, residual_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.residual = residual_func

    def setJacobian(
        self, jacobian_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.jacobian = jacobian_func

    def setDataX(self, data_x: np.ndarray):
        self.data_x = lambda _: data_x

    def setDataY(self, data_y: np.ndarray):
        self.data_y = lambda _: data_y

    def setInitialModel(self, initial_model: Union[Model, np.ndarray]):
        self.initial_model = (
            lambda _: initial_model.values()
            if isinstance(initial_model, Model)
            else initial_model
        )


class LeastSquareObjective(BaseObjective):
    """
    General class holder for objective functions that are calculated from data misfit

    Feed the data into constructor, and least squares misfit will be generated automatically.
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
                f"Numbers of data points don't match between X:{X.shape} and Y:"
                f"{Y.shape}"
            )

        self.X = X
        self.Y = np.expand_dims(Y, axis=1)
        if not isinstance(forward, BaseForward):
            forward = BaseForward(forward, X.shape[1])
        self.forward = forward
        self.nparams = forward.model_dimension()

        if initial_model is not None:
            self.initial_model = np.asanyarray(
                initial_model.values()
                if isinstance(initial_model, Model)
                else initial_model
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
        X = np.expand_dims(self.X, axis=1) if len(self.X.shape) == 1 else self.X
        model = np.squeeze(model)
        predicted_Y = np.apply_along_axis(self.forward.calc_curried(model), 1, X)
        return np.squeeze(predicted_Y - self.Y)

    def data_x(self):
        return self.X

    def data_y(self):
        return self.Y

    def initial_model(self):
        return self.prior

    def params_size(self):
        return len(self.prior)


class LinearFittingObjective(LeastSquareObjective):
    def __init__(
        self,
        X,
        Y,
        nparams,
        design_matrix: Callable = None,
        forward: LinearFittingFwd = None,
        initial_model: Union[Model, np.ndarray, list] = None,
    ):
        self.calc_design_matrix = design_matrix
        if forward and hasattr(forward, "design_matrix"):
            self.calc_design_matrix = forward.design_matrix
        elif forward is None:
            if design_matrix is None:
                raise ValueError(
                    "Please specify at least one of design_matrix and forward"
                )
            forward = LinearFittingFwd(nparams, design_matrix)

        super().__init__(X, Y, forward, initial_model)

    def design_matrix(self):
        if hasattr(self, "_design_matrix"):
            return self._design_matrix
        else:
            self._design_matrix = self.calc_design_matrix(self.X)
            return self._design_matrix

    def gradient(self, model: Union[Model, np.ndarray]):
        return np.squeeze(self.jacobian(model).T @ self.residual(model))

    def hessian(self, model: Union[Model, np.ndarray]):
        g = self.design_matrix()
        return g.T @ g

    def jacobian(self, model: Union[Model, np.ndarray]):
        return self.design_matrix()
