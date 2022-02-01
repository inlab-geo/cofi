from . import Model

import numpy as np
from typing import Callable, Union
from numbers import Number


class BaseForward:
    def __init__(self, forward: Callable, nparams: Number):
        self._forward = forward
        self.nparams = nparams

    def calc(self, model: Union[Model, np.ndarray], X) -> np.ndarray:
        X = np.asanyarray(X)
        return self._forward(model, X)

    def calc_curried(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        def calc_with_model(X):
            return self.calc(model, X)

        return calc_with_model

    def design_matrix(self, X):  # only solver targeting linear forward will call this
        raise NotImplementedError(
            "Linear solvers should have 'LinearFittingFwd' as forward solver or"
            " implements design_matrix method"
        )

    def model_dimension(self):
        return self.nparams


class LinearFittingFwd(BaseForward):
    def __init__(self, nparams, design_matrix=None):
        self.nparams = nparams
        if design_matrix:
            self.design_matrix = design_matrix
        else:
            self.design_matrix = lambda X: X

    def calc(self, model: Union[Model, np.ndarray], X):
        self.nparams = model.length() if isinstance(model, Model) else len(model)
        X = self.design_matrix(X)
        if self.nparams != X.shape[1]:
            raise ValueError(
                f"Parameters count ({self.nparams}) doesn't match X shape"
                f" {X.shape} in linear curve forward fitting"
            )

        if isinstance(model, Model):
            model = model.values()
        return self._forward(model, X)

    def _forward(self, model: np.ndarray, X: np.ndarray) -> np.ndarray:
        return X @ model


class PolynomialFittingFwd(LinearFittingFwd):
    def __init__(self, order: int = None):
        if order:
            self.nparams = order + 1

    def calc(self, model: Union[Model, np.ndarray], x):  # put here to avoid confusion
        return super().calc(model, x)

    def design_matrix(self, x):
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
            X = np.array([x ** o for o in range(self.nparams)]).T
            if len(X.shape) == 1:
                X = np.expand_dims(X, axis=0)
            return X
        except AttributeError:
            raise ValueError(
                "Please specify the order of linear curve fitting by either passing it through the constructor or passing in a model through the 'calc' method"
            )
