from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import LeastSquareObjective, Model
from ._cpp_solver_lib import solve as cpp_solve

import numpy as np
from warnings import warn

class LRNormalEquationCpp(BaseSolver):
    def __init__(self, objective: LeastSquareObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn(
            "You are using linear regression formula solver, please note that this is"
            " only for small scale of data"
        )

        G = self.objective.design_matrix()
        Y = self.objective.data_y()

        res = cpp_solve(G.shape[1], G.shape[0], G, Y)
        # res = np.linalg.inv(G.T @ G) @ (G.T @ Y)

        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model