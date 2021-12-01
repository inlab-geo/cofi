from typing import Callable
from .model_params import Model

import numpy as np


class BaseForward:
    def __init__(self, forward: Callable):
        self._forward = forward

    def solve(self, model: Model, X) -> np.ndarray:
        X = np.asanyarray(X)
        return self._forward(model, X)

    def get_G(self, X):  # only solver targeting linear forward will call this
        raise NotImplementedError(
            "Linear solvers should have 'LinearFittingFwd' as forward solver or"
            " implements get_G method"
        )


class LinearFittingFwd(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Model, X):
        self.params_count = model.length()
        X = self.get_G(X)
        if self.params_count != X.shape[1]:
            raise ValueError(
                f"Parameters count ({self.params_count}) doesn't match X shape"
                f" {X.shape} in linear curve forward fitting"
            )

        return self._forward(model, X)

    def _forward(self, model: Model, X: np.ndarray) -> np.ndarray:
        return X @ model.values()

    def get_G(self, X, order=None):
        X = np.asanyarray(X)
        return X

    def model_dimension(self):
        return self.params_count


class PolynomialFittingFwd(LinearFittingFwd):
    def __init__(self):
        pass

    def solve(self, model: Model, x):  # put here to avoid confusion
        return super().solve(model, x)

    def get_G(self, x):
        """
        this is invoked by solve(model, x) from superclass prior to solving
        polynomial transformation happens here
        """
        x = np.asanyarray(x)
        if len(x.shape) != 1:
            raise ValueError(
                "Shape of x should be a 1-D array in the context of polynomial"
                f" fitting, however got shape {x.shape}"
            )

        X = np.array([x ** o for o in range(self.params_count)]).T
        return X
