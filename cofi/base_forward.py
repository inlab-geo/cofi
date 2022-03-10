from numbers import Number
from typing import Callable, Union

import numpy as np

from . import Model


class BaseForward:
    def __init__(self, forward: Callable, nparams: Number):
        self._forward = forward
        self.nparams = nparams

    def calc(self, model: Union[Model, np.ndarray], data_X) -> np.ndarray:
        data_X = np.asanyarray(data_X)
        return self._forward(model, data_X)

    def calc_curried(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        def calc_with_model(data_X):
            return self.calc(model, data_X)

        return calc_with_model

    def basis_function(
        self, data_X
    ):  # only solver targeting linear forward will call this
        raise NotImplementedError(
            "Linear solvers should have 'LinearForward' as forward solver or"
            " implements basis_function method"
        )

    def model_dimension(self):
        return self.nparams


class LinearForward(BaseForward):
    def __init__(self, nparams, basis_function=None):
        self.nparams = nparams
        if basis_function:
            self.basis_function = basis_function
        else:
            self.basis_function = lambda X: X

    def calc(self, model: Union[Model, np.ndarray], data_X):
        self.nparams = model.length() if isinstance(model, Model) else len(model)
        basis_matrix = self.basis_function(data_X)
        if self.nparams != basis_matrix.shape[1]:
            raise ValueError(
                f"Parameters count ({self.nparams}) doesn't match the shape of"
                f" basis_matrix {basis_matrix.shape} in linear curve forward fitting"
            )

        if isinstance(model, Model):
            model = model.values()
        return _linear_forward(model, basis_matrix)


def _linear_forward(model: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    return basis_matrix @ model


class PolynomialForward(LinearForward):
    def __init__(self, order: int = None):
        if order:
            self.nparams = order + 1

    def calc(
        self, model: Union[Model, np.ndarray], data_x
    ):  # put here to avoid confusion
        return super().calc(model, data_x)

    def basis_function(self, x):
        """
        This is invoked by calc(model, x) from superclass prior to solving.
        The polynomial transformation happens here
        """
        x = np.asanyarray(x)
        ncolumns = 1 if len(x.shape) == 1 else x.shape[1]
        if ncolumns != 1:
            raise ValueError(
                "Shape of x should be a column vertex in the context of polynomial"
                f" fitting, however got shape {x.shape}"
            )

        try:
            x = np.squeeze(x)
            basis_matrix = np.array([x**o for o in range(self.nparams)]).T
            if len(basis_matrix.shape) == 1:
                basis_matrix = np.expand_dims(basis_matrix, axis=0)
            return basis_matrix
        except AttributeError:
            raise ValueError(
                "Please specify the order of linear curve fitting by either passing it"
                " through the constructor or passing in a model through the 'calc'"
                " method"
            )
