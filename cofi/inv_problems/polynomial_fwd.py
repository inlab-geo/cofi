import numpy as np

from . import BaseForward


class PolynomialForward(BaseForward):
    def __init__(self, data_x):
        self.data_x = data_x
    
    def __call__(self, model):
        return np.polynomial.Polynomial(model)(self.data_x)
