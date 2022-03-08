from typing import Callable, Union
from numbers import Number

import numpy as np

from . import Model, BaseForward, LinearForward


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

    def __init__(self, func: Callable[[np.ndarray], Number] = None):
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

    def set_misfit(self, misfit_func: Callable[[Union[Model, np.ndarray]], Number]):
        self.misfit = misfit_func

    def set_gradient(
        self, gradient_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.gradient = gradient_func

    def set_hessian(
        self, hessian_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.hessian = hessian_func

    def set_residual(
        self, residual_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.residual = residual_func

    def set_jacobian(
        self, jacobian_func: Callable[[Union[Model, np.ndarray]], np.ndarray]
    ):
        self.jacobian = jacobian_func

    def set_data_X(self, data_x: np.ndarray):
        self.data_x = lambda _: data_x

    def set_data_Y(self, data_y: np.ndarray):
        self.data_y = lambda _: data_y

    def set_initial_model(self, initial_model: Union[Model, np.ndarray]):
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


class LinearObjective(LeastSquareObjective):
    def __init__(
        self,
        X,
        Y,
        nparams,
        basis_function: Callable = None,
        forward: LinearForward = None,
        initial_model: Union[Model, np.ndarray, list] = None,
    ):
        self.basis_function = basis_function
        if forward and hasattr(forward, "basis_function"):
            self.basis_function = forward.basis_function
        elif forward is None:
            if basis_function is None:
                raise ValueError(
                    "Please specify at least one between `basis_function` and `forward`"
                )
            forward = LinearForward(nparams, basis_function)

        super().__init__(X, Y, forward, initial_model)

    def basis_matrix(self):
        if hasattr(self, "_basis_matrix"):
            return self._basis_matrix
        else:
            self._basis_matrix = self.basis_function(self.X)
            return self._basis_matrix

    def gradient(self, model: Union[Model, np.ndarray]):
        return np.squeeze(self.jacobian(model).T @ self.residual(model))

    def hessian(self, model: Union[Model, np.ndarray]):
        g = self.basis_matrix()
        return g.T @ g

    def jacobian(self, model: Union[Model, np.ndarray]):
        return self.basis_matrix()
