from cofi.cofi_solvers import BaseSolver
from cofi.cofi_objective import LeastSquareObjective, Model
from ._f77_solver_lib import solve as f77_solve

import numpy as np
from warnings import warn

class LRNormalEquationF77(BaseSolver):
    def __init__(self, objective: LeastSquareObjective):
        self.objective = objective

    def solve(self) -> Model:
        warn(
            "You are using linear regression formula solver, please note that this is"
            " only for small scale of data"
        )

        G = self.objective.design_matrix()
        Y = self.objective.data_y()

        res = f77_solve(G.shape[1], G.shape[0], np.asfortranarray(G), np.asfortranarray(Y))

        model = Model(
            **dict([("p" + str(index[0]), val) for (index, val) in np.ndenumerate(res)])
        )
        return model