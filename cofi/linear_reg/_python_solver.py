import numpy as np

from .. import Model, BaseSolver, LeastSquareObjective
from ._utils import warn_normal_equation


class LRNormalEquation(BaseSolver):
    def __init__(self, objective: LeastSquareObjective):
        self.objective = objective

    def solve(self, reg_eps_squared=None) -> Model:
        warn_normal_equation()

        G = self.objective.basis_matrix()
        Y = self.objective.data_y()
        # TODO regularisation handling? prior model? (ref: inverseionCourse.curveFitting)
        # TODO return posterior covariance? (ref: inverseionCourse.curveFitting)
        if reg_eps_squared is None:
            res = np.linalg.inv(G.T @ G) @ G.T @ Y
        else:  # Tikhonov-regularised solution
            res = (
                np.linalg.inv(G.T @ G + reg_eps_squared * np.eye(G.shape[1])) @ G.T @ Y
            )
        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model
