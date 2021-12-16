from cofi.cofi_objective.base_objective import LeastSquareObjective
from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import LeastSquareObjective, Model

import numpy as np
from warnings import warn


class SimpleLinearRegression(BaseSolver):
    def __init__(self, objective: LeastSquareObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn("You are using linear regression formula solver, please note that this is only for small scale of data")

        G = self.objective.basis_transform(self.objective.data_x())
        Y = self.objective.data_y()
        # TODO regularisation handling? prior model? (ref: inverseionCourse.curveFitting)
        # TODO return posterior covariance? (ref: inverseionCourse.curveFitting)
        res = np.linalg.inv(G.T @ G) @ (G.T @ Y)
        model = Model(
            **dict(
                [("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)]
            )
        )
        return model
