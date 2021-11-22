from typing import Callable
from .model_params import Model, Parameter

import numpy as np


class BaseForward:
    def __init__(self, forward: Callable):
        self.forward = forward

    def solve(self, model: Model, X) -> np.array:
        if isinstance(X, list):
            X = np.array(X)
        return self.forward(model, X)


class LinearFittingFwd(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Model, X):
        X = self.get_X(X, model.length())
        if model.length() != X.shape[1]:
            raise ValueError(
                f"Parameters count ({model.length()}) doesn't match X shape {X.shape} in linear curve forward fitting"
            )

        return self.forward(model, X)

    def forward(self, model: Model, X: np.array) -> np.array:
        return X @ model.values()

    def get_X(self, X, order=None):
        if isinstance(X, list):
            X = np.array(X)
        return X


class PolynomialFittingFwd(LinearFittingFwd):
    def __init__(self):
        pass

    def get_X(self, x, order: int):
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) != 1:
            raise ValueError(
                f"Shape of x should be a 1-D array in the context of polynomial fitting, however got shape {x.shape}"
            )

        X = np.array([x ** o for o in range(order)]).T
        return X


class FourierFittingFwd(LinearFittingFwd):
    def __init__(self):
        pass

    def solve(self, model: Model, x, domain_length=1.0):
        # argument checking
        if model.length() % 2 == 0:
            raise ValueError(
                f"Fourier basis requires odd number of model parameters, however got {model.length()}"
            )
        if domain_length <= 0:
            raise ValueError(
                f"Argument 'domain_length' must be positive, however got {domain_length}"
            )
        if not np.all(0 <= x) and np.all(x <= domain_length):
            raise ValueError(
                f"Fourier basis requires all sample points to be in within given range: (0, {domain_length})"
            )

        # TODO (reference inverseionCourse.curveFitting lines 35-43)
        raise NotImplementedError("Fourier curve fitting #TODO")
