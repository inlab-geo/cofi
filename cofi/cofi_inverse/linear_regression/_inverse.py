from cofi.cofi_forward.model_params import Model
from cofi.cofi_inverse import BaseInverse
from cofi.cofi_forward import BaseObjectiveFunction, Model, Parameter

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression(BaseInverse):
    def __init__(self, initial_model: Model, objective: BaseObjectiveFunction):
        self.prior = initial_model
        self.objective = objective
        self.forward = objective.forward

    def solve(self) -> Model:
        G = self.forward.get_G(self.objective.X)
        Y = self.objective.Y
        GTG = G.T @ G
        # TODO regularisation handling? prior model? (ref: inverseionCourse.curveFitting) 
        # TODO return posterior covariance? (ref: inverseionCourse.curveFitting)
        estimated_model = np.linalg.inv(GTG) @ (G.T @ (Y - G @ self.prior.values()))
        # TODO estimate value
