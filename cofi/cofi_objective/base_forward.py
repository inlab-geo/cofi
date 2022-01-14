from . import Model

import numpy as np
from typing import Callable, Union


class BaseForward:
    def __init__(self, forward: Callable):
        self._forward = forward

    def solve(self, model: Union[Model, np.ndarray], X) -> np.ndarray:
        X = np.asanyarray(X)
        return self._forward(model, X)

    def solve_curried(self, model: Union[Model, np.ndarray]) -> np.ndarray:
        def solve_with_model(X):
            return self.solve(model, X)

        return solve_with_model

    def design_matrix(self, X):  # only solver targeting linear forward will call this
        raise NotImplementedError(
            "Linear solvers should have 'LinearFittingFwd' as forward solver or"
            " implements design_matrix method"
        )


class LinearFittingFwd(BaseForward):
    def __init__(self, params_count, design_matrix=None):
        self.params_count = params_count
        if design_matrix:
            self.design_matrix = design_matrix
        else:
            self.design_matrix = lambda X: X

    def solve(self, model: Union[Model, np.ndarray], X):
        self.params_count = model.length() if isinstance(model, Model) else len(model)
        X = self.design_matrix(X)
        if self.params_count != X.shape[1]:
            raise ValueError(
                f"Parameters count ({self.params_count}) doesn't match X shape"
                f" {X.shape} in linear curve forward fitting"
            )

        if isinstance(model, Model):
            model = model.values()
        return self._forward(model, X)

    def _forward(self, model: np.ndarray, X: np.ndarray) -> np.ndarray:
        return X @ model

    def model_dimension(self):
        return self.params_count


class PolynomialFittingFwd(LinearFittingFwd):
    def __init__(self, order: int = None):
        if order:
            self.params_count = order + 1

    def solve(self, model: Union[Model, np.ndarray], x):  # put here to avoid confusion
        return super().solve(model, x)

    def design_matrix(self, x):
        """
        This is invoked by solve(model, x) from superclass prior to solving.
        The polynomial transformation happens here
        """
        x = np.asanyarray(x)
        if len(x.shape) != 1:
            raise ValueError(
                "Shape of x should be a 1-D array in the context of polynomial"
                f" fitting, however got shape {x.shape}"
            )

        if self.params_count is None:
            raise ValueError("Please specify the order of linear curve fitting")
        X = np.array([x ** o for o in range(self.params_count)]).T
        return X
