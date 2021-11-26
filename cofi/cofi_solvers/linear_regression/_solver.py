from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import BaseObjective, Model

import numpy as np


class LinearRegression(BaseSolver):
    def __init__(self, initial_model: Model, objective: BaseObjective):
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
