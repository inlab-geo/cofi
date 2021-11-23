from typing import Callable
from .model_params import Model

import numpy as np


class BaseForward:
    def __init__(self, forward: Callable):
        self._forward = forward

    def solve(self, model: Model, X) -> np.array:
        if isinstance(X, list):
            X = np.array(X)
        return self._forward(model, X)

    def get_G(self, X):     # only solver targeting linear forward will call this
        raise NotImplementedError("Linear solvers should have 'LinearFittingFwd' as forward solver or implements get_G method")


class LinearFittingFwd(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Model, X):
        self.params_count = model.length()
        X = self.get_G(X)
        if self.params_count != X.shape[1]:
            raise ValueError(
                f"Parameters count ({self.params_count}) doesn't match X shape {X.shape} in linear curve forward fitting"
            )

        return self._forward(model, X)

    def _forward(self, model: Model, X: np.array) -> np.array:
        return X @ model.values()

    def get_G(self, X, order=None):
        if isinstance(X, list):
            X = np.array(X)
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
        if isinstance(x, list):
            x = np.array(x)
        if len(x.shape) != 1:
            raise ValueError(
                f"Shape of x should be a 1-D array in the context of polynomial fitting, however got shape {x.shape}"
            )

        X = np.array([x ** o for o in range(self.params_count)]).T
        return X 


class FourierFittingFwd(LinearFittingFwd):
    def __init__(self):
        pass

    def solve(self, model: Model, x):  # put here to avoid confusion
        return super().solve(model, x)

    def get_G(self, x, domain_length=1.0):
        """ 
        this is invoked by solve(model, x) from superclass prior to solving
        Fourier transformation happens here, so the rest work is only linear
        """
        # argument checking
        if self.params_count % 2 == 0:
            raise ValueError(
                f"Fourier basis requires odd number of model parameters, however got {self.params_count}"
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
