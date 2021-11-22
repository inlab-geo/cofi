from .model_params import Model, Parameter

import numpy as np


class BaseForward:
    def __init__(self, forward: function):
        self.forward = forward

    def solve(self, model: Model, X) -> np.array:
        if isinstance(X, list): X = np.array(X)
        return self.forward(model, X)


class LinearFittingFwd(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Model, X):
        if isinstance(X, list): X = np.array(X)
        if model.length() != X.shape[1]:
            raise ValueError(f"Parameters count {model.length()} doesn't match X shape {X.shape} in linear curve forward fitting")
        
        return self.forward(model, X)

    def forward(model: Model, X: np.array) -> np.array:
        return X @ model.values()


class PolynomialFittingFwd(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Model, x):
        if isinstance(x, list): x = np.array(x)
        if len(x.shape) != 1:
            raise ValueError(f"Shape of x should be a 1-D array in the context of polynomial fitting, however got shape {x.shape}")

        return self.forward(model, x)

    def forward(model: Model, x: np.array) -> np.array:
        order = model.length()
        X = np.array([x ** o for o in range(order)]).T
        return X @ model.values()


class FourierFittingFwd(BaseForward):
    def __init__(self):
        pass

    def solve(self, model: Model, x, domain_length = 1.):
        if model.length() % 2 == 0:
            raise ValueError(f"Fourier basis requires odd number of model parameters, however got {model.length()}")
        if domain_length <= 0:
            raise ValueError(f"Argument 'domain_length' must be positive, however got {domain_length}")
        if not np.all(0<=x) and np.all(x<=domain_length):
            raise ValueError(f"Fourier basis requires all sample points to be in within given range: (0, {domain_length})")
        
        raise NotImplementedError("#TODO")

